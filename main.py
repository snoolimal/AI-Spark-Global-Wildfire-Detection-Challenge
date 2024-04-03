from composer import MetaLoader
from models import UN2P, UN3PL, UN3P
from core import Trainer, Predictor
from utils import setup_logger, main_flow
from config import parse_args


def main():
    args = parse_args()

    check = args.check
    seed = args.seed
    threshold = args.threshold
    groups, models = main_flow(args.group), main_flow(args.model)
    mdname, fdname = args.mdname, args.fdname
    logger, debugger = setup_logger(fdname), setup_logger(fdname, debug=True)

    valid_size = args.valid_size
    monitor = args.monitor
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    early_stopping_rounds = args.early_stopping_rounds
    sch = args.sch

    for mode in ('train', 'test'):
        metaloader = MetaLoader(mode, fdname, mdname)
        metaloader.load_meta(save=True)

    for group in groups:
        for model in models:
            if model == 'un2p':
                architecture = UN2P
            elif model == 'un3pl':
                architecture = UN3PL
            elif model == 'un3p':
                architecture = UN3P
            trainer = Trainer(check, seed, model, architecture, mdname, group, fdname, logger)
            trainer.train_valid(valid_size, monitor, epochs, batch_size, lr, early_stopping_rounds, sch)
            predictor = Predictor(model, architecture, mdname, group, fdname, threshold, logger)
            predictor.predict(batch_size)


if __name__ == '__main__':
    main()