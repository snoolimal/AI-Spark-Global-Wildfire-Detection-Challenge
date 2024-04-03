import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Wildfire Segmentation of Multispectral Satellite Image')

    # Directory
    parser.add_argument('--mdname', type=str, default='meta', help='name of metadata file.')
    parser.add_argument('--fdname', type=str, default='temp', help='folder where result is saved.')

    # Version
    parser.add_argument('--check', action='store_true', help='check wheter code runs without error.')
    parser.add_argument('--group', type=str, nargs='+', default='small', choices=['small', 'medium', 'large', 'out'],
                        help="size of wildfire, target group of inference. select one or more among 'small', 'medium', 'large', 'out'")
    parser.add_argument('--model', type=str, nargs='+', default='un3pl',
                        help="select one or more among 'un3pl', 'un3p', 'un2p'.")
    parser.add_argument('--seed', type=int, default=42)

    # Training
    parser.add_argument('--valid_size', type=float, default=0.2)
    parser.add_argument('--monitor', type=str, default='loss', help="'loss' or score_name.")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--div', type=int, default=10, help='div factor for cycle scheduler.')
    parser.add_argument('--early_stopping_rounds', type=int, default=11)
    parser.add_argument('--sch', type=str, default='RLROP',
                        choices=['LLR', 'MLR', 'SLR', 'MSLR', 'ELR', 'CALR', 'RLROP', 'CLR', 'OCLR', 'CAWR'],
                        help='scheduler. select one among choies. access to config in module components-scheduler_config')

    #  Inference
    parser.add_argument('--threshold', type=float, default=0.75, help='threshold for final binary inference.')

    args = parser.parse_args()

    if args.check:
        args.epochs = 2

    args.early_stopping_rounds = args.epochs // args.div + 2

    return args