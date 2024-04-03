from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class UN3SK(nn.Module):
    def __init__(self, add_channels: int | None):
        super().__init__()
        self.lw = True if add_channels is None else False
        self.keep_size = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pooling layer

    @staticmethod
    def h_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    @staticmethod
    def enc_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
        )

    @staticmethod
    def upsamp_block(scale_factor, mode,
                      in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    def enc_to_dec(self, enc_lv, dec_lv, encoder_channels, kernel_size=3, stride=1, padding=1):
        in_channels = encoder_channels[enc_lv]
        out_channels = encoder_channels[dec_lv] if self.lw else encoder_channels[0]

        if enc_lv == dec_lv:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
        else:
            pool_size = 2 ** (dec_lv - enc_lv)
            return nn.Sequential(
                nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )

    @staticmethod
    def make_dec_block(h, *args, dim=1):
        return h(torch.cat(args, dim))

    @classmethod
    def check(cls, model_name: str, bchw: Tuple[int, int, int, int] = (10, 3, 256, 256), check_size: bool = True):
        from config import model_config
        model = cls(**model_config[model_name]).to('cuda')
        x = torch.randn(bchw).to('cuda')
        output = model(x).detach()

        if check_size:
            print(output.size())  # [B, num_classes=1, H, W]
        else:
            # print(model)
            return output

    @staticmethod
    def lw_is_available(encoder_channels: Tuple[int, int, int]):
        """층이 3개인 UNet 구조에 적절한 모델 하이퍼퍼라미터가 설정되었는지 확인한다.

        :param encoder_channels: encoder의
            [0]: 0층에서의 입력 채널 수
            [1]: 1층에서의 입력 채널 수
            [2]: 2층에서의 입력 채널 수는 반드시 0층의 채널 수의 4배가 되어야 한다.
        """
        if not len(encoder_channels) == 3:
            raise RuntimeError(
                'This Skeleton is only for 3-level UNet architecture.'
            )
        if not encoder_channels[-1] == (encoder_channels[0] * 4):
            raise RuntimeError(
                'Number of input channel of encoder Lv.Max(2) must be same with 4 * its Lv.0'
            )
