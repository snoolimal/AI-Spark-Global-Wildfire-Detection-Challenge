from __future__ import annotations
from typing import Tuple
import torch
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Wildfire Segmentation of Multispectral Satellite Image')

    # Base
    parser.add_argument('--check', action='store_true', help='check whether code runs without error.')
    parser.add_argument('--group', type=str, default='small', choices=['small', 'medium', 'large'],
                        help='size of wildfire.')   # 지정 안하면 None (전체)
    parser.add_argument('--folder_name', type=str, required=True, help='folder where result is saved.')
    parser.add_argument('--seed', type=int, default=42)

    # Preprocessing (Grouping)
    parser.add_argument('--cutoff', type=float, default=0.2, help='cutoff for stats for grouping.')
    parser.add_argument('--binning', nargs='+', default=[4, 12], help='bins for grouping.')

    # Model
    parser.add_argument('--model', type=str, choices=['unet2p', 'unet3plw'])   # 지정 안하면 None (전체)

    # Training
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--monitor', type=str, default='loss', help="'loss' or score_name.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stopping_rounds', type=int, default=11)

    # Scheduler
    parser.add_argument('--factor', type=float, default=0.1, help='reducing lr: new_lr = lr * factor.')
    parser.add_argument('--div', type=int, default=2, help='for scheduler.')
    parser.add_argument('--verbose', action='store_false')

    # Inference
    parser.add_argument('--threshold', type=float, default=0.75, help='threshold for final binary inference.')

    args = parser.parse_args()

    # Scheduler
    mode, patience, min_lr = _get_scheduler_config(args)
    setattr(args, 'mode', mode)
    setattr(args, 'patience', patience)
    setattr(args, 'min_lr', min_lr)
    setattr(args, 'factor', 0.1)
    setattr(args, 'verbose', False)

    if args.check:
        args.epochs = 2

    return args


def _get_scheduler_config(args: argparse.Namespace) -> Tuple[str, int, float]:
    """
    :return: mode, patience, min_lr
    """
    mode = 'min' if args.monitor == 'loss' else 'max'
    patience = args.early_stopping_rounds // args.div
    min_lr = args.lr * (0.1 ** args.div)

    return mode, patience, min_lr
