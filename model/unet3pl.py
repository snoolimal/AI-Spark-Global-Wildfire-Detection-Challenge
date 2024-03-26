from __future__ import annotations
from typing import List
import torch.nn as nn
from model.utils import UNUtils


class UN3PL(UNUtils):
    """Lightweighted UNet+++

    디코더의 업샘플링을 3번만 한다.
    """
    def __init__(
            self,
            in_channels: int,
            encoder_channels: List[int, int, int],
            upsamp_mode: str,
            num_classes: int,
            add_channels=None
    ):
        super().__init__(add_channels)
        UNUtils._lw_is_available(encoder_channels)

        ## H Block
        self.h1 = nn.Sequential(
            UNUtils._h_block(encoder_channels[2] * 3, encoder_channels[2]),
            UNUtils._h_block(encoder_channels[2], encoder_channels[2])
        )
        self.h2 = nn.Sequential(
            UNUtils._h_block(encoder_channels[1] * 2 + encoder_channels[0], encoder_channels[1] * 2 + encoder_channels[0]),
            UNUtils._h_block(encoder_channels[1] * 2 + encoder_channels[0], encoder_channels[2])
        )
        self.h3 = nn.Sequential(
            UNUtils._h_block(encoder_channels[2], encoder_channels[2]),
            UNUtils._h_block(encoder_channels[2], encoder_channels[2])
        )

        ## Encoder Block
        self.enc0 = UNUtils._enc_block(in_channels, encoder_channels[0])
        self.enc1 = UNUtils._enc_block(encoder_channels[0], encoder_channels[1])
        self.enc2 = UNUtils._enc_block(encoder_channels[1], encoder_channels[2])

        ## Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        ## Connection
        self.enc0_dec0 = self._enc_to_dec(enc_lv=0, dec_lv=0, encoder_channels=encoder_channels)
        self.enc0_dec1 = self._enc_to_dec(enc_lv=0, dec_lv=1, encoder_channels=encoder_channels)
        self.enc0_dec2 = self._enc_to_dec(enc_lv=0, dec_lv=2, encoder_channels=encoder_channels)
        self.enc1_dec1 = self._enc_to_dec(enc_lv=1, dec_lv=1, encoder_channels=encoder_channels)
        self.enc1_dec2 = self._enc_to_dec(enc_lv=1, dec_lv=2, encoder_channels=encoder_channels)
        self.enc2_dec2 = self._enc_to_dec(enc_lv=2, dec_lv=2, encoder_channels=encoder_channels)

        ## Upsampling Block
        upsamp_config = {'mode': upsamp_mode, 'in_channels': encoder_channels[2], 'out_channels': encoder_channels[0]}
        self.dec_up2 = UNUtils._upsamp_block(scale_factor=2, **upsamp_config)
        self.dec_up4 = UNUtils._upsamp_block(scale_factor=4, **upsamp_config)
        self.enc2_up4 = UNUtils._upsamp_block(scale_factor=4, **upsamp_config)

        ## Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], num_classes, **self.keep_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc0 = self.enc0(x)
        pool1 = self.pool(enc0)

        enc1 = self.enc1(pool1)
        pool2 = self.pool(enc1)

        enc2 = self.enc2(pool2)

        dec2 = UN3PL._concat_input(self.h1,
                                   self.enc0_dec2(enc0),
                                   self.enc1_dec2(enc1),
                                   self.enc2_dec2(enc2))
        dec1 = UN3PL._concat_input(self.h2,
                                   self.enc0_dec1(enc0),
                                   self.enc1_dec1(enc1),
                                   self.dec_up2(dec2))
        dec0 = UN3PL._concat_input(self.h3,
                                   self.enc0_dec0(enc0),
                                   self.dec_up2(dec1),
                                   self.dec_up4(dec2),
                                   self.enc2_up4(enc2))   # 4배

        out = self.segmentation_head(dec0)

        return out


# UN3PL.check('unet3pl')    # [B=10, num_classes=1, H=256, W=256]
