import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet2P(nn.Module):
    def __init__(self, **config):
        super().__init__()

        unet2p = smp.UnetPlusPlus(**config)

        self.encoder = unet2p.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = unet2p.decoder
        self.segmentation_head = unet2p.segmentation_head

    def forward(self, x):
        encoded = self.encoder(x)  # feature maps from freezed encoder
        decoded = self.decoder(*encoded)
        output = self.segmentation_head(decoded)    # logits

        return output

    @classmethod
    def check(cls, bchw=(10, 3, 256, 256), check_size=True):
        model = cls().to('cuda')
        x = torch.randn(bchw).to('cuda')
        output = model(x).detach()

        if check_size:
            print(output.size())    # [B, 1, H, W]
        else:
            return output


UNet2P.check()  # [B=10, C=1, H=256, W=256]
