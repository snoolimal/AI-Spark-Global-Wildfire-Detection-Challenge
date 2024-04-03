from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import albumentations
from torch.utils.data import Dataset, DataLoader
from composer.scaler import Scaler
from utils import Dir, Loader
from config import preprocess_config


class UPDataset(Dataset):
    def __init__(
            self,
            meta: pd.DataFrame,
            group: str,
            transforms: albumentations.Compose = None,
            testmode: bool = False,
            config: Dict = preprocess_config,
    ):
        self.meta = meta[meta['group'] == group].reset_index(drop=True)
        self.transforms = transforms
        self.mode = 'train' if not testmode else 'test'
        self.testmode = testmode
        self.config = config

    def __len__(self):
        return len(self.meta)

    def get_input_image(self, num: int) -> np.ndarray:
        """6번 채널 + 6번 채널 + [robust 스케일링 -> 지형지물 제거 채널]

        :return: HWC
        """
        channels, input_percentiles, weight = self.config['channels'], self.config['input_percentiles'], self.config['weight']
        scaler = Scaler(self.mode, channels=channels)

        six = scaler.load_tif(num, channel=6)

        scaled = scaler.scale_input(num, input_percentiles)
        removed = scaler.remove_scape(scaled, weight)
        removed = removed[:, :, np.newaxis]

        input_image = np.concatenate((six, six, removed), axis=2)

        return input_image

    def __getitem__(self, item):
        """

        :param item:
        :return:
            - image: (H, W, C=3) array
            - mask: (H, W, C=1) array
        """
        loader = Loader(self.mode)
        num = self.meta.loc[item, 'num']
        image = self.get_input_image(num)   # HWC

        # Training or Validation
        if not self.testmode:
            mask = loader.load_mask(num)
            mask = mask[:, :, np.newaxis]   # HWC

            if self.transforms is not None:
                transformed = self.transforms(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']

            return image, mask
        # Testing
        else:
            name = self.meta.loc[item, 'img_name']

            if self.transforms is not None:
                image = self.transforms(image=image)['image']

            return name, image

    @classmethod
    def check(cls):
        from config import parse_args
        mdname = parse_args().mdname
        meta_path = Dir.get_root_dir('meta') / f'train_{mdname}.csv'
        meta = pd.read_csv(meta_path)

        DS = cls(meta, group='small')
        img, msk = DS.__getitem__(0)[0], DS.__getitem__(0)[1]
        print(img.shape, msk.shape)     # (H=256, W=256, C=3), (H=256, W=256, C=1)

        DL = DataLoader(DS, batch_size=16)
        for x, y in DL:
            print(x.shape, y.shape)     # [B=16, C=256, H=256, W=3], [B=16, C=256, H=256, W=1]
            break


# UPDataset.check()
