import logging


def setup_logger(save_path: str, name: str, debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # DEBUG-INFO-WARNING-ERROR-CRITICAL

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(fmt='%(levelname)s: %(message)s | %(asctime)s',
                                          datefmt='%H:%M:%S %m-%d')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

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

