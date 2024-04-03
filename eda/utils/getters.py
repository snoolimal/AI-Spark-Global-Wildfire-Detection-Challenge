from __future__ import annotations
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm

# MODE = 'train'  # 'train' or 'test
DATA_DIR = Path(__file__).resolve().parents[2] / 'dataset'      # C:\Users\soono\OneDrive\__Projects__\AI-S~e\dataset
SAVE_DIR = Path(__file__).resolve().parents[1] / 'meta'     # C:\Users\soono\OneDrive\__Projects__\AI-S~e\eda\meta
                                                        # 이걸 Path.cwd().parnet로 하면 eda 폴더의 ipynb에서는 ~eda로 읽힘
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _get_tif(num: int, channel: int | list | tuple | np.ndarray = None, mode='train') -> np.ndarray:
    """시각화를 위해 HWC의 array를 반환한다. (C는 1 or 3)

    :return: HWC
    """
    assert mode in ['train', 'test'], "Parameter must be either 'train' or 'test'."
    # path = Path.cwd().parent / 'dataset' / 'train_img' / f'train_img_{num}.tif'   # 요렇게 하면 여기 PyCharm에서는 이 함수 사용 불가능
    path = DATA_DIR / f'{mode}_img' / f'{mode}_img_{num}.tif'
    tif = tiff.imread(str(path))

    if channel is not None:     # channel이 None이면 (256, 256, 10)의 전체 tif 로드
        tif = tif[:, :, channel]

        if not isinstance(channel, (list, tuple, np.ndarray)):
            tif = tif[:, :, np.newaxis]

    return tif


def get_tif(num, channel: int | list | tuple = None, mode='train'):
    """CHW가 정배다. (C는 1 or 3)

    :return: CHW
    """
    assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

    path = DATA_DIR / f'{mode}_img' / f'{mode}_img_{num}.tif'
    tif = tiff.imread(str(path))

    if channel is not None:
        tif = tif[:, :, channel]

        if not isinstance(channel, (list, tuple)):
            tif = tif[np.newaxis, :, :]
        else:
            tif = tif.transpose(2, 0, 1)

    return tif


def _get_mask(num, mode='train'):
    assert mode in ['train', 'test'], "Parameter 'mode' must be either 'train' or 'test'."

    path = DATA_DIR / f'{mode}_mask' / f'{mode}_mask_{num}.tif'
    mask = tiff.imread(str(path))

    return mask


def get_mask_meta(also_tpos=True):
    mask_dir = DATA_DIR / 'train_mask'
    mask_paths = [p for p in mask_dir.glob('*.tif')]

    mask_meta = pd.DataFrame(index=range(len(mask_paths)))
    mask_meta_dict = dict()
    for i, path in enumerate(tqdm(mask_paths, desc=f'running get_mask_meta() | also_tpos={also_tpos}')):
        mask = tiff.imread(str(path))

        name = str(path.stem).split('_')[-1]
        mask_meta.loc[i, 'num'] = name
        mask_meta.loc[i, 'one'] = int(mask.flatten().sum())

        if also_tpos:
            row_idx, col_idx = mask.nonzero()
            target_pos = list(zip(row_idx, col_idx))
            mask_meta_dict[name] = target_pos

    mask_meta['num'] = mask_meta['one'].astype(int)
    mask_meta = mask_meta.sort_values(by='one').reset_index(drop=True)

    if also_tpos:
        int_keys_dict = {int(key): value for key, value in mask_meta_dict.items()}
        tpos = dict(sorted(int_keys_dict.items()))

        return mask_meta, tpos
    else:
        return mask_meta


def get_target_val(channel, tpos):
    """

    :param channel: channel
    :param tpos: get_mask_data()의 tpos (target_pos.pkl)
    """
    tval = dict()
    pbar = tqdm(tpos.items())
    for i, (key, value) in enumerate(pbar):
        pbar.set_description(f'running get_target_val() | channel {channel}')

        tif = get_tif(i, channel)
        val_list = []
        for h, w in value:
            val = tif[:, h, w].item()
            val_list.append(val)
        tval[key] = val_list

    return tval


def get_out_meta(meta):
    out_list = []
    for i in tqdm(range(len(meta)), desc='running get_out_list()'):
        is_zeros = 0

        for c in range(9):
            is_zero = get_tif(i, c).flatten().min() == 0
            is_zeros += is_zero

        if is_zeros == 9:
            out_list.append(i)

    out = pd.DataFrame()
    out['num'] = meta['one']
    out['one'] = out_list

    return out