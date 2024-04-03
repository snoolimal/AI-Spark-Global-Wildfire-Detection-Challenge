from __future__ import annotations
from typing import Tuple
import numpy as np
import logging
from utils import Loader


class Scaler(Loader):
    """
    이 버전의 스케일러 클래스는 "동일한 채널들"에 다양한 스케일링을 먹이는 경우에 적합하다.
    그러므로 생성자에서 채널을 받음으로써 인스턴스를 생성할 때 다룰 채널을 지정하는 과정이 자연스럽다.
    서로 다른 채널을 선택해 각기 다른 스케일링을 수행한다면
    스케일러 클래스의 생성자가 아닌 각각의 메소드에 채널을 지정하는 파라미터를 넣어 주어
    여러 커스텀 스케일 함수를 담고 있는 스케일러 클래스에서 선택한 채널용 스케일링 함수가 적용되도록 만드는 것이 자연스럽다.
    """
    def __init__(
            self,
            mode: str,
            channels: Tuple[int, ...] | int,    # Tuple[int, int, int] | int = (4, 5, 6)
            debugger: logging.Logger = None
    ):
        super().__init__(mode)
        self.channels = channels
        self.debugger = debugger

    def scale_for_group(self, num: int) -> np.ndarray:
        """그룹화를 위한 스케일러.

        이미지를 채널별로 minmax 스케일링하여 값을 [0, 1]에 가둔다.

        :return: HWC
        """
        tif = super().load_tif(num, self.channels)

        mms = np.empty(tif.shape)   # minmax scaled
        for c in range(tif.shape[-1]):  # 채널별로 스케일링
            schn = tif[:, :, c]    # single channel
            minimum, maximum = schn.min(), schn.max()
            schns = (schn - minimum) / (maximum - minimum)     # single channel scaled
            mms[:, :, c] = schns

        return mms

    def scale_input(self, num: int, input_percentiles: Tuple[int, int, int],):
        """입력 이미지의 스케일러.

        입력 이미지를 채널별로 robust 스케일링한다.

        :return: HWC
        """
        tif = super().load_tif(num, self.channels)
        rbs = np.empty(tif.shape)   # robust scaled
        percentiles = np.sort(input_percentiles)
        for c in range(tif.shape[-1]):  # 채녈별로 스케일링
            schn = tif[:, :, c]     # single channel
            plb, pmed, pub = tuple([np.percentile(schn, p) for p in percentiles])
            if pub - plb == 0:
                # self.debugger.warning(f'{mode} image {num} channel {c} | pub - plb = {pub - plb}')
                non_zero_pxs = schn[schn != 0]
                plb, pmed, pub = tuple(np.percentile(non_zero_pxs, p) for p in schn)

            schns = (schn - pmed) / (pub - plb)
            rbs[:, :, c] = schns

        return rbs

    @staticmethod
    def remove_scape(scaled: np.ndarray, weight: float) -> np.ndarray:
        """
        :return: HW
        """
        removed = scaled[:, :, 2] - (scaled[:, :, 1] * weight) - (scaled[:, :, 0] * (1 - weight))
        return removed