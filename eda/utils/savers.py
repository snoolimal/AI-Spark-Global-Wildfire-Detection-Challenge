from __future__ import annotations
from numpy.typing import NDArray
import joblib
from eda.utils.getters import *


def save_mask_meta(also_tpos=True):
    meta = get_mask_meta(also_tpos)

    if isinstance(meta, tuple):
        mask_meta, tpos = meta
        mask_meta.to_csv(str(SAVE_DIR / 'mask_meta.csv'), index=False)
        joblib.dump(tpos, str(SAVE_DIR / 'target_pos.pkl'))
    else:
        meta.to_csv(str(SAVE_DIR / 'mask_meta.csv'), index=False)


def save_target_val(channel: list | NDArray[np.int64] = np.arange(0, 10)):
    tpos_path = SAVE_DIR / 'target_pos.pkl'
    if tpos_path.exists():
        tpos = joblib.load(str(tpos_path))
    else:
        _, tpos = get_mask_meta()

    for c in channel:
        tval = get_target_val(channel=c, tpos=tpos)
        joblib.dump(tval, str(SAVE_DIR / f'target_val_{c}.pkl'))


def save_out_meta():
    meta_path = SAVE_DIR / 'mask_meta.csv'
    if meta_path.exists():
        meta = pd.read_csv(str(meta_path))
        out_meta = get_out_meta(meta)
    else:
        meta = get_mask_meta(also_tpos=False)
        out_meta = get_out_meta(meta)

    out_meta.to_csv(str(SAVE_DIR / 'out_meta.csv'), index=False)
