from typing import Dict, Tuple
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from segmentation_models_pytorch.losses import LovaszLoss
import random
from copy import deepcopy
from tqdm import tqdm
from composer import MetaLoader, UPDataset
from utils import Dir, Scheduler, save_config
from config import preprocess_config, model_config, transforms, scheduler_config


class Trainer:
    """
    고정된 seed에서
    특정 모델을
    그룹화한 metadata에 대해
    지정한 group에 대해 학습하되
    fdname에 결과를 저장하며
    그 과정을 로깅하는
    학습기
    """
    def __init__(
            self,
            check: bool,
            seed: int,
            model: str,
            architecture,   # architecture: torch.nn.Module,
            mdname: str,
            group: str,
            fdname: str,
            logger: logging.Logger,
    ):
        self.check = check
        self.seed = seed
        self.model = model
        self.architecture = architecture
        self.mdname = mdname
        self.group = group
        self.fdname = fdname
        self.logger = logger

        self.dirbox = Dir(fdname)

    def set_seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.logger.info('Seed Fixed.')

    @staticmethod
    def _train_valid_split(valid_size, seed, check, meta: pd.DataFrame) \
            -> (pd.DataFrame, pd.DataFrame):
        indices = np.arange(len(meta))

        train_idx, valid_idx = train_test_split(indices,
                                                test_size=valid_size,
                                                random_state=seed)
        train_meta = meta.iloc[train_idx].reset_index(drop=True)
        valid_meta = meta.iloc[valid_idx].reset_index(drop=True)

        if check:
            train_meta = train_meta.iloc[:30]
            valid_meta = valid_meta.iloc[:5]

        return train_meta, valid_meta

    def _get_dataloader(self, train_package: Tuple, valid_package: Tuple, batch_size, testmode):
        train_dataset = UPDataset(meta=train_package[0],
                                  group=self.group,
                                  transforms=train_package[1],
                                  testmode=testmode)
        valid_dataset = UPDataset(meta=valid_package[0],
                                  group=self.group,
                                  transforms=valid_package[1],
                                  testmode=testmode)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

        return train_loader, valid_loader

    @staticmethod
    def _train(train_loader, model, criterion, optimizer, device, desc):
        model.train()

        running_loss = 0.
        pbar = tqdm(train_loader, total=len(train_loader), desc=f'Training | {desc}', leave=False)
        for images, masks in pbar:
            images, masks = images.to(device).float(), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(y_pred=outputs.float(), y_true=masks.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        return train_loss

    @staticmethod
    def _valid(valid_loader, model, criterion, monitor, device, desc, scheduler):
        model.eval()

        running_loss, running_metric = 0., 0.
        pbar = tqdm(valid_loader, total=len(valid_loader), desc=f'Validation | {desc}', leave=False)
        for images, masks in pbar:
            images, masks = images.to(device).float(), masks.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(y_pred=outputs.float(), y_true=masks.float())
                # metric = custom_metric(outputs, masks)

                running_loss += loss.item()
                # running_metric += metric.item()
        valid_loss = running_loss / len(valid_loader)
        # valid_metric = running_metric / len(valid_loader)

        if hasattr(scheduler, 'cycle_momentum'):
            scheduler.step()
        else:
            pass

        if monitor == 'loss':
            return valid_loss, None
        # else:
        #     return valid_loss, valid_metric

    def _update(
            self,
            model, monitor,
            epoch, running, best,
            scoreboard, best_weight,    # weight_save_path,
            patience,
    ):
        sota_updated = False
        if (monitor == 'loss' and running < best) or (monitor != 'loss' and running > best):
            sota_updated = True

        if sota_updated:
            self.logger.info(f'Sota updated on epoch {epoch} | {best:.5f} -> {running:.5f}')

            # best = running.cpu().numpy().item()
            best = running
            best_weight = deepcopy(model.state_dict())
            # torch.save(best_weight, weight_save_path)

            scoreboard['epoch'].append(epoch)
            scoreboard['best'].append(best)
            # scoreboard['weight_path'].append(weight_save_path)

            patience = 0

            return best, patience, scoreboard, best_weight
        else:
            patience += 1
            return best, patience, scoreboard, best_weight

    def _save_result(self, scoreboard: Dict = None):
        if scoreboard is None:
            items = [f'<{self.fdname} for Group {self.group}>',
                     'Preprocessing', 'Model', 'Transforms', 'Scheduler']
            configs = [None, preprocess_config, model_config, transforms['train'], scheduler_config]
            modes = len(configs) * ['a']
            modes[0] = 'w'
            for item, config, mode in zip(items, configs, modes):
                save_config(self.fdname, f'{self.group}_hparams', 'txt', item, config, mode)
            self.logger.info('Hyperparameter settings is saved.')
        else:
            scoreboard = pd.DataFrame(scoreboard)
            save_path = self.dirbox.get_save_path(sname=f'{self.group}_scoreboard', suf='csv')
            scoreboard.to_csv(save_path, index=False)
            self.logger.info('Scoreboard is saved.')

    def _save_best_weight(self, best_weight, epoch):
        save_path = self.dirbox.get_save_path(sname=f'best_weight_epoch{epoch}', suf='pth')
        torch.save(best_weight, save_path)
        self.logger.info(f'Best weight from epoch {epoch} is saved.')

    def train_valid(
            self,
            valid_size,
            monitor,
            epochs,
            batch_size,
            lr,
            early_stopping_rounds,
            sch: str
    ):
        self.set_seed(self.seed)
        self._save_result()

        meta = MetaLoader(mode='train', fdname=self.fdname, mdname=self.mdname).load_meta()
        train_meta, valid_meta = Trainer._train_valid_split(valid_size, self.seed, self.check, meta)
        train_trans, valid_trans = transforms['train'], transforms['test']
        train_package, valid_package = (train_meta, train_trans), (valid_meta, valid_trans)
        train_loader, valid_loader = self._get_dataloader(train_package, valid_package, batch_size, testmode=False)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.architecture(**model_config[self.model])
        model.to(device)
        criterion = LovaszLoss(mode='binary')
        optimizer = optim.Adam(model.parameters(), lr=lr)
        _scheduler = Scheduler(optimizer, self.logger)
        scheduler = _scheduler.get_scheduler(sch)(**scheduler_config[sch])

        best = np.inf if monitor == 'loss' else 0.
        best_weight = deepcopy(model.state_dict())
        patience = 0
        scoreboard = {'epoch': [], 'best': [],
                      # 'weight_path': []
                      }
        for epoch in range(1, epochs + 1):
            desc = f'{self.fdname} Group {self.group} | Epoch {epoch}'
            train_loss = Trainer._train(train_loader, model, criterion, optimizer, device, desc)
            valid_loss, valid_metric = Trainer._valid(valid_loader, model, criterion, monitor, device, desc, scheduler)
            running = valid_loss if monitor == 'loss' else valid_metric
            if hasattr(scheduler, 'patience'):  # ReduceLROnPlateau 고려
                scheduler.step(running)
            elif hasattr(scheduler, 'cycle_momentum'):  # CycleLR, OneCycleLR 고려
                pass
            elif scheduler is None:
                pass
            else:
                scheduler.step()

            msg = f'Epoch {epoch}/{epochs} | train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}'
            # msg = f'{self.fdname} Group {self.group} | {msg}'
            if valid_metric is not None:
                msg += f', {monitor}: {valid_metric:.5f}'
            self.logger.info(msg)
            _scheduler.log_lr(scheduler, epoch)

            best, patience, scoreboard, best_weight = self._update(model, monitor, epoch, running, best, scoreboard, patience, best_weight)
            if patience == early_stopping_rounds:
                self.logger.info(f'Early stopping triggered on epoch {epoch - patience}.')
                self._save_best_weight(best_weight, epoch)
                self._save_result(scoreboard)

                break

        self._save_best_weight(best_weight, epoch)
        self._save_result(scoreboard)
