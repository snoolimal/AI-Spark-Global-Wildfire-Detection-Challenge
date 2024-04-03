from typing import Tuple
import torch.nn as nn
from models.skeleton import UN3SK


class UN3PL(UN3SK):
    """Lightweighted UNet+++

    경량화된 3층의 UNet+++를 구현한다.
    """

    def __init__(
            self,
            in_channels: int,
            encoder_channels: Tuple[int, int, int],
            upsamp_mode: str,
            num_classes: int,
            add_channels=None,
    ):
        super().__init__(add_channels)
        UN3SK.lw_is_available(encoder_channels)

        ## H Block
        self.h1 = nn.Sequential(
            UN3SK.h_block(encoder_channels[2] * 3, encoder_channels[2]),
            UN3SK.h_block(encoder_channels[2], encoder_channels[2])
        )
        self.h2 = nn.Sequential(
            UN3SK.h_block(encoder_channels[1] * 2 + encoder_channels[0],
                          encoder_channels[1] * 2 + encoder_channels[0]),
            UN3SK.h_block(encoder_channels[1] * 2 + encoder_channels[0],
                          encoder_channels[2]),
        )
        self.h3 = nn.Sequential(
            UN3SK.h_block(encoder_channels[2], encoder_channels[2]),
            UN3SK.h_block(encoder_channels[2], encoder_channels[2]),
        )

        ## Encoder Block
        self.enc0 = UN3SK.enc_block(in_channels, encoder_channels[0])
        self.enc1 = UN3SK.enc_block(encoder_channels[0], encoder_channels[1])
        self.enc2 = UN3SK.enc_block(encoder_channels[1], encoder_channels[2])

        ## Connection
        self.enc0_dec0 = self.enc_to_dec(enc_lv=0, dec_lv=0, encoder_channels=encoder_channels)
        self.enc0_dec1 = self.enc_to_dec(enc_lv=0, dec_lv=1, encoder_channels=encoder_channels)
        self.enc0_dec2 = self.enc_to_dec(enc_lv=0, dec_lv=2, encoder_channels=encoder_channels)
        self.enc1_dec1 = self.enc_to_dec(enc_lv=1, dec_lv=1, encoder_channels=encoder_channels)
        self.enc1_dec2 = self.enc_to_dec(enc_lv=1, dec_lv=2, encoder_channels=encoder_channels)
        self.enc2_dec2 = self.enc_to_dec(enc_lv=2, dec_lv=2, encoder_channels=encoder_channels)

        ## Upsampling Block
        upsamp_config = {'mode': upsamp_mode, 'in_channels': encoder_channels[2], 'out_channels': encoder_channels[0]}
        self.dec_up2 = UN3SK.upsamp_block(scale_factor=2, **upsamp_config)
        self.dec_up4 = UN3SK.upsamp_block(scale_factor=4, **upsamp_config)
        self.enc2_up4 = UN3SK.upsamp_block(scale_factor=4, **upsamp_config)

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

        dec2 = UN3SK.make_dec_block(self.h1,
                                    self.enc0_dec2(enc0),
                                    self.enc1_dec2(enc1),
                                    self.enc2_dec2(enc2))
        dec1 = UN3SK.make_dec_block(self.h2,
                                    self.enc0_dec1(enc0),
                                    self.enc1_dec1(enc1),
                                    self.dec_up2(dec2))
        dec0 = UN3SK.make_dec_block(self.h3,
                                    self.enc0_dec0(enc0),
                                    self.dec_up2(dec1),
                                    self.dec_up4(dec2),
                                    self.enc2_up4(enc2))  # 4배

        out = self.segmentation_head(dec0)

        return out


# UN3PL.check('unet3pl')  # [B=10, num_classes=1, H=256, W=256]
