from typing import Iterable
from utils.dir import Dir


def check_mode(mode: str):
    if mode not in ['train', 'test']:
        raise RuntimeError("Parameter 'mode' must be either 'train' or 'test'.")
    # assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."


def save_config(fdname: str, sname: str, suf: str,
                item: str, config: Iterable = None, mode: str = 'a'):
    save_path = Dir(fdname).get_save_path(sname, suf)
    with open(save_path, mode) as file:
        file.write('\n' + item + '\n')

        if config is not None:
            if isinstance(config, dict):
                for key, value in config.items():
                    line = f'{key}: {value}\n'
                    file.write(line)
            else:
                for line in config:
                    file.write(str(line) + '\n')


def main_flow(obj) -> Iterable:
    if isinstance(obj, str):
        return [obj]
    else:
        return obj