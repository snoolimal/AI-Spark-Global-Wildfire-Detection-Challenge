from typing import Tuple
import torch.nn as nn
from models.skeleton import UN3SK


class UN3P(UN3SK):
    """Standard UNet+++

    5층짜리 기본 UNet+++를 구현한다.
    """
    def __init__(
            self,
            in_channels: int,
            encoder_channels: Tuple[int, int, int, int, int],
            upsamp_mode: str,
            num_classes: int,
            add_channels: int
    ):
        super().__init__(add_channels)

        ## H Block
        self.h = UN3SK.h_block(add_channels, add_channels)

        ## Encoder Block
        self.enc0 = UN3SK.enc_block(in_channels, encoder_channels[0])
        self.enc1 = UN3SK.enc_block(encoder_channels[0], encoder_channels[1])
        self.enc2 = UN3SK.enc_block(encoder_channels[1], encoder_channels[2])
        self.enc3 = UN3SK.enc_block(encoder_channels[2], encoder_channels[3])
        self.enc4 = UN3SK.enc_block(encoder_channels[3], encoder_channels[4])

        ## Connection
        self.enc0_dec0 = self.enc_to_dec(enc_lv=0, dec_lv=0, encoder_channels=encoder_channels)
        self.enc0_dec1 = self.enc_to_dec(enc_lv=0, dec_lv=1, encoder_channels=encoder_channels)
        self.enc0_dec2 = self.enc_to_dec(enc_lv=0, dec_lv=2, encoder_channels=encoder_channels)
        self.enc0_dec3 = self.enc_to_dec(enc_lv=0, dec_lv=3, encoder_channels=encoder_channels)
        self.enc1_dec1 = self.enc_to_dec(enc_lv=1, dec_lv=1, encoder_channels=encoder_channels)
        self.enc1_dec2 = self.enc_to_dec(enc_lv=1, dec_lv=2, encoder_channels=encoder_channels)
        self.enc1_dec3 = self.enc_to_dec(enc_lv=1, dec_lv=3, encoder_channels=encoder_channels)
        self.enc2_dec2 = self.enc_to_dec(enc_lv=2, dec_lv=2, encoder_channels=encoder_channels)
        self.enc2_dec3 = self.enc_to_dec(enc_lv=2, dec_lv=3, encoder_channels=encoder_channels)
        self.enc3_dec3 = self.enc_to_dec(enc_lv=3, dec_lv=3, encoder_channels=encoder_channels)

        ## Upsampling Block
        upsamp_config = {'mode': upsamp_mode, 'in_channels': add_channels, 'out_channels': encoder_channels[0]}
        self.dec_up2 = UN3SK.upsamp_block(scale_factor=2, **upsamp_config)
        self.dec_up4 = UN3SK.upsamp_block(scale_factor=4, **upsamp_config)
        self.dec_up8 = UN3SK.upsamp_block(scale_factor=8, **upsamp_config)

        upsamp_config['in_channels'] = encoder_channels[4]
        self.enc4_up2 = UN3SK.upsamp_block(scale_factor=2, **upsamp_config)
        self.enc4_up4 = UN3SK.upsamp_block(scale_factor=4, **upsamp_config)
        self.enc4_up8 = UN3SK.upsamp_block(scale_factor=8, **upsamp_config)
        self.enc4_up16 = UN3SK.upsamp_block(scale_factor=16, **upsamp_config)

        ## Segmentation Head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(add_channels, num_classes, **self.keep_size),
            nn.Sigmoid()    # LovaszLoss는 softmax(binary라면 sigmoid겠지)를 취한 후의 예측값에 사용
        )

    def forward(self, x):
        enc0 = self.enc0(x)
        pool1 = self.pool(enc0)

        enc1 = self.enc1(pool1)
        pool2 = self.pool(enc1)

        enc2 = self.enc2(pool2)
        pool3 = self.pool(enc2)

        enc3 = self.enc3(pool3)
        pool4 = self.pool(enc3)

        enc4 = self.enc4(pool4)

        dec3 = UN3SK.make_dec_block(self.h,
                                    self.enc0_dec3(enc0),
                                    self.enc1_dec3(enc1),
                                    self.enc2_dec3(enc2),
                                    self.enc3_dec3(enc3),
                                    self.enc4_up2(enc4))
        dec2 = UN3SK.make_dec_block(self.h,
                                    self.enc0_dec2(enc0),
                                    self.enc1_dec2(enc1),
                                    self.enc2_dec2(enc2),
                                    self.dec_up2(dec3),
                                    self.enc4_up4(enc4))
        dec1 = UN3SK.make_dec_block(self.h,
                                    self.enc0_dec1(enc0),
                                    self.enc1_dec1(enc1),
                                    self.dec_up2(dec2),
                                    self.dec_up4(dec3),
                                    self.enc4_up8(enc4))
        dec0 = UN3SK.make_dec_block(self.h,
                                    self.enc0_dec0(enc0),
                                    self.dec_up2(dec1),
                                    self.dec_up4(dec2),
                                    self.dec_up8(dec3),
                                    self.enc4_up16(enc4))

        out = self.segmentation_head(dec0)

        return out


# UN3P.check('unet3p')    # [B=10, num_classes=1, H=256, W=256]
