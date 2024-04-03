import logging
from utils.dir import Dir


def setup_logger(fdname: str, debug: bool = False) -> logging.Logger:
    name = 'train_log' if not debug else 'debug_log'
    name = f'{fdname}_' + name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)   # DEBUG-INFO-WARNING-ERROR-CRITICAL

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(fmt='%(levelname)s: %(message)s | %(asctime)s',
                                          datefmt='%H:%M:%S %m-%d')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    D = Dir(fdname)
    save_dir = Dir.get_root_dir('log')
    save_path = D.get_save_path(sdir=save_dir, sname=name, suf='txt')
    file_handler = logging.FileHandler(save_path)
    if not debug:
        file_handler.setLevel(logging.INFO)
        file_handler.addFilter(_InfoFilter())
    else:
        file_handler.setLevel(logging.WARNING)
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


class _InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO