import albumentations as A
from albumentations.pytorch import ToTensorV2


model_config = {
    'unet2p': {
        'encoder_name': 'resnet18',
        'encoder_weights': 'imagenet',
        'encoder_depth': 3,
        'decoder_channels': [128, 64, 32],
        'decoder_attention_type': 'scse',
        'in_channels': 3,
        'classes': 1,
        'activation': None,
    },
    'unet3pl': {
        'in_channels': 3,
        'encoder_channels': [16, 32, 64],   # 64 = 16 * "4"
        'upsamp_mode': 'bilinear',   # mode for upsampling
        'num_classes': 1,
    },
    'unet3p': {
        'in_channels': 3,
        'encoder_channels': [64, 128, 256, 512, 1024],
        'add_channels': 320,
        'upsamp_mode': 'bilinear',
        'num_classes': 1,
    }
}

transforms = {
    'train': A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2(transpose_mask=True)
    ]),
    'test': A.Compose([
        ToTensorV2(transpose_mask=True)
    ])
}
