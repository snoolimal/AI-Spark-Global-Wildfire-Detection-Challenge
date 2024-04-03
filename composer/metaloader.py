from typing import Dict
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from composer.scaler import Scaler
from utils import check_mode, Dir, setup_logger, Loader
from config import preprocess_config


class MetaLoader(Loader):
    def __init__(
            self,
            mode: str,
            fdname: str,
            mdname: str,
            config: Dict = preprocess_config
    ):
        super().__init__(mode)
        self.mdname = mdname
        self.config = config
        self.logger = setup_logger(fdname)
        self.debugger = setup_logger(fdname, debug=True)
        self.D = Dir(fdname)

    def load_base_meta(self, count: bool = False) -> pd.DataFrame:
        img_dir = Dir.get_root_dir('data') / f'{self.mode}_img'
        img_paths = [p for p in img_dir.glob('*.tif')]
        img_names = [p.name for p in img_paths]
        img_nums = np.array([p.split('_')[-1].split('.')[0] for p in img_names]).astype(int)
        meta = pd.DataFrame(
            dict(num=img_nums, img_name=img_names, img_path=img_paths)
        )

        if self.mode == 'train':
            mask_dir = Dir.get_root_dir('data') / 'train_mask'
            mask_paths = [p for p in mask_dir.glob('*.tif')]
            mask_names = [p.stem for p in mask_paths]
            mask_nums = np.array([p.split('_')[-1] for p in mask_names]).astype(int)

            mask_meta = pd.DataFrame(
                dict(mask_name=mask_names, mask_path=mask_paths, mask_num=mask_nums)
            )
            meta = pd.concat(objs=[meta, mask_meta], axis=1)

            if count:
                ones = [super().load_mask(num).flatten().sum() for num in tqdm(mask_nums)]
                meta['one'] = ones

        meta = meta.sort_values(by='num').reset_index(drop=True)

        return meta

    def mark_sonnal(self, meta: pd.DataFrame) -> pd.DataFrame:
        """손날치기 이미지를 마킹한다."""
        out_list = []
        pbar = tqdm(meta['num'].values, desc=f'{self.mode.upper()} | Marking sliced images.', leave=False)
        for n in pbar:
            is_zeros = 0
            for c in range(9):
                is_zero = super().load_tif(n, c).flatten().min() == 0
                is_zeros += is_zero

            if is_zeros == 9:
                out_list.append(n)

        meta['stat'] = -1
        meta.loc[~meta['num'].isin(out_list), 'stat'] = 0

        return meta

    def mark_size(self, meta: pd.DataFrame) -> pd.DataFrame:
        """불의 크기를 마킹한다. (그룹화)"""
        meta_for_group = meta[meta['stat'] == 0]  # 손날치기는 이미 걸렀고...
        S = Scaler(self.mode, channels=self.config['channels'])

        pbar = tqdm(meta_for_group['num'].values, desc=f'{self.mode.upper()} | Marking the size of fire. (Grouping)')
        for num in pbar:
            scaled = S.scale_for_group(num)
            removed = Scaler.remove_scape(scaled, self.config['weight'])
            output = np.where(removed > self.config['cutoff'], 1, 0)     # cutoff보다 크면 발화점(1), 작으면 정상(0)
            stats = output.flatten().sum()
            meta.loc[meta['num'] == num, 'stat'] = stats

        conditions = [
            meta['stat'] < 0,
            meta['stat'].between(0, self.config['binning'][0]),
            meta['stat'].between(self.config['binning'][0] + 1, self.config['binning'][1]),
            meta['stat'] > self.config['binning'][1]
        ]
        groups = ['out', 'small', 'medium', 'large']
        meta['group'] = np.select(conditions, groups)

        meta.insert(0, 'stat', meta.pop('stat'))
        meta.insert(0, 'group', meta.pop('group'))

        return meta

    def load_meta(self, save=False):
        check_mode(self.mode)
        save_dir = Dir.get_root_dir(return_type='meta')
        sname = f'{self.mode}_{self.mdname}'
        save_path = self.D.get_save_path(sdir=save_dir, sname=sname, suf='csv')
        if Path(save_path).exists():
            meta = pd.read_csv(save_path)
            if save:
                self.debugger.warning(f'{self.mdname} already exists.')
            else:
                return meta
        else:
            meta = self.load_base_meta()
            meta = self.mark_sonnal(meta)
            meta = self.mark_size(meta)
            if save:
                meta.to_csv(save_path, index=False)
                self.logger.info(
                    f'Metadata {sname}.csv is saved.'
                )
            else:
                return meta
