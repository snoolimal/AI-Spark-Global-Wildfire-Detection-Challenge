import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.argparse import parse_args


preprocess_config = {
    'input_percentiles': (50, 80, 20),
    'channels': (4, 5, 6),
    'weight': 0.8,
    'cutoff': 0.2,
    'binning': (4, 12),
}

model_config = {
    'un2p': {
        'encoder_name': 'resnet18',
        'encoder_weights': 'imagenet',
        'encoder_depth': 3,
        'decoder_channels': (128, 64, 32),
        'decoder_attention_type': 'scse',
        'in_channels': 3,
        'classes': 1,
        'activation': None,
    },
    'un3pl': {
        'in_channels': 3,
        'encoder_channels': (16, 32, 64),   # 64 = 16 * "4"
        'upsamp_mode': 'bilinear',   # mode for upsampling
        'num_classes': 1,
    },
    'un3p': {
        'in_channels': 3,
        'encoder_channels': (64, 128, 256, 512, 1024),
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

args = parse_args()
scheduler_config = {
    'LLR': {
        'factor': 0.95
    },
    'MLR': {
        'factor': 0.95,
    },
    'SLR': {
        'step_size': 5,
        'gamma': 0.1,
    },
    'MSLR': {
        'milestones': np.arange(0, 100, 10)[1:],
        'gamma': 0.1
    },
    'ELR': {
        'gamma': 0.1
    },
    'CALR': {
        'T_max': 10,
        'eta_min': 1e-6
    },
    'RLROP': {
        'mode': 'min' if args.monitor == 'loss' else 'max',
        'threshold_mode': 'rel',
        'factor': 0.05,
        'patience': 5,
        'threshold': 0.0,
        'min_lr': 1e-6,
        'eps': 1e-6
    },
    'CLR': {
        'mode': 'triangular',   # 'triangular' or 'trianguler2' or 'exp_range'
        'base_lr': 1e-6,
        'max_lr': args.lr * args.div,
        'step_size_up': args.epoch // 2,
        'step_size_down': args.epoch // 2,
        'gamma': 0.95,      # exp_range에 대해서만 유효
        'scale_fn': None,   # exp_range면 커스텀 스케일링 함수 필요
        'scale_mode': 'cycle',
        'cycle_momentum': True,
        'base_momentum': 0.8,
        'max_momentum': 0.9
    },
    'OCLR': {
        'max_lr': args.lr * args.div,
        'epochs': args.epochs,
        'steps_per_epoch': args.batch_size,
        'pct_s': 1 / args.div,
        'anneal_strategy': 'cos',
        'div_factor': 10,   # initial_lr = max_lr / div_factor
        'final_div_factor': 100,    # max_lr = initial_lr / final_div_factor
        'cycle_momentum': True,
        'base_momentum': 0.85,
        'max_momentum': 0.95
    },
    'CAWR': {
        'T_0': args.div,
        'T_mult': args.div,
        'eta_min': 1e-6
    }
}


def lr_lambda(epoch):
    pass


def scale_fn():
    pass