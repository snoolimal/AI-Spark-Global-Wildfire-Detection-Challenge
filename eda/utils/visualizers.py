import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from eda.utils.getters import _get_tif, _get_mask


def vis1(num, channel=None, figsize=(4, 4)):
    if channel is None:
        tif = _get_tif(num, (3, 2, 1))
        tif = rescale_intensity(tif, in_range=(tif.min(), tif.max()), out_range=(0, 1))
        tif = np.array(tif)     # tif는 이미 rescale_intensity()를 타면 array일 텐데,
                                # 여기서 한번 더 변환 안 해주면 plt.imshow()에서
                                # Expected type 'Union[_SupportsArray[dtype], _NestedSequence[_SupportsArray[dtype]], bool, int, float, complex, str, bytes, _NestedSequence[Union[bool, int, float, complex, str, bytes]], Image]', got 'array.pyi' instead의 warning
    else:
        tif = _get_tif(num, channel)
        tif = rescale_intensity(tif, in_range=(tif.min(), tif.max()), out_range=(0, 255))
        tif = np.array(tif)

    fig = plt.figure(figsize=figsize)
    plt.imshow(tif, cmap='gray')
    plt.title(f'Train {num}')
    plt.axis('off')
    plt.show()


def vis2(num, nrows=3, ncols=4):
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 12))

    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]

            idx = row * 4 + col
            if idx <= 9:
                channel = idx
                cimg = _get_tif(num, channel)
                cimg = rescale_intensity(cimg, in_range=(cimg.min(), cimg.max()), out_range=(0, 1))

                ax.imshow(cimg, cmap='gray')
                ax.set_title(f'channel {channel}', fontsize=10)
            elif idx == 10:
                rgb = _get_tif(num, (3, 2, 1))
                rgb = rescale_intensity(rgb, in_range=(rgb.min(), rgb.max()), out_range=(0, 1))
                ax.imshow(rgb)
                ax.set_title('RGB', fontsize=10)
            elif idx == 11:
                mask = _get_mask(num)
                ax.imshow(mask, cmap='gray')
                ax.set_title(f'Mask ({mask.flatten().sum()}/{(mask.shape[0] * mask.shape[1])})', fontsize=10)
            else:
                ax.set_visible(False)

            ax.axis('off')

    plt.suptitle(f'Train {num}')
    plt.tight_layout()
    plt.show()



