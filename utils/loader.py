from __future__ import annotations
from typing import Sequence
from numpy.typing import NDArray
import numpy as np
import tifffile as tiff
from pathlib import Path
from utils.dir import Dir
from utils.utils import check_mode


class Loader:
    def __init__(self, mode: str, data_dir: Path = Dir.get_root_dir('data')):
        check_mode(mode)
        self.data_dir = data_dir
        self.mode = mode

    def load_mask(self, num: int) -> NDArray[int, ...]:
        """
        0이나 1로 구성된 마스킹 이미지를 반환한다.
        발화점은 1, 정상점은 0이다.

        :return: HW
        """
        path = self.data_dir / 'train_mask', f'train_mask_{num}.tif'
        mask = tiff.imread(str(path))
        return mask

    def load_tif(self, num: int, channel: int | Sequence = None) -> NDArray[int, ...]:
        """
        반사율을 픽셀값으로 갖는 이미지를 반환한다.
        반사율은 [0, 1]도, [0, 255]의 범위도 아니고 그냥 지멋대로다.

        :return: HWC
        """
        path = self.data_dir / f'{self.mode}_img' / f'{self.mode}_img_{num}.tif'
        tif = tiff.imread(str(path))

        if channel is not None:
            tif = tif[:, :, channel]
            if isinstance(channel, int):
                tif = tif[:, :, np.newaxis]

        return tif