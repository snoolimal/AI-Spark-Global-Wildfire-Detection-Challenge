import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import joblib
from tqdm import tqdm
from composer import MetaLoader, UPDataset
from utils import Dir
from config import model_config, transforms


class Predictor:
    def __init__(
            self,
            model: torch.nn.Module,
            architecture,   # architecture: torch.nn.Module,
            mdname: str,
            group: str,
            fdname: str,
            threshold: int,
            logger: logging.Logger,
    ):
        self.model = model
        self.architecture = architecture
        self.mdname = mdname
        self.group = group
        self.fdname = fdname
        self.threshold = threshold
        self.logger = logger
        self.dirbox = Dir(fdname)

    def predict(self, batch_size=1):
        best_weight_path = self.dirbox.save_dir.glob('*.pth')[0]
        best_weight = torch.load(str(best_weight_path))

        test_meta = MetaLoader('test', self.fdname, self.mdname).load_meta()
        test_trans = transforms['test']
        test_dataset = UPDataset(meta=test_meta,
                                 group=self.group,
                                 transforms=test_trans,
                                 testmode=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.architecture(**model_config[self.model])
        model.to(device)

        y_pred_dict = {}
        pbar = tqdm(test_loader, total=len(test_loader),
                    desc=f'{self.fdname} Group {self.group} | Inference | using {Path(best_weight_path).stem}')
        for name, image in pbar:
            image = image.to(device).float()

            with torch.no_grad():
                output = model(image)  # [B, C=1, H, W]
                # output = output.permute(0, 2, 3, 1)

            output = output.cpu().numpy()
            # y_pred = np.where(output[0, :, :, 0] > configs['threshold'], 1, 0)
            y_pred = np.where(output[0, 0, :, :] > self.threshold, 1, 0)
            y_pred = y_pred.astype(np.uint8)
            y_pred_dict[name[0]] = y_pred

        self.logger.info('Inference is Done.')
        save_path = self.dirbox.get_save_path(sname='submission', suf='pkl')
        joblib.dump(y_pred_dict, save_path)
