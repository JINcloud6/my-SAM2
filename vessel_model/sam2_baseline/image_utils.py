import os

import numpy as np
from PIL import Image


def get_slice(volume, axis, curr_idx, box):
    if axis == 0:
        sl = volume[curr_idx, box[1]:box[2], box[3]:box[4]]
    elif axis == 1:
        sl = volume[box[1]:box[2], curr_idx, box[3]:box[4]]
    else:
        sl = volume[box[1]:box[2], box[3]:box[4], curr_idx]
    return sl


def to_uint8_rgb(slice_2d: np.ndarray) -> np.ndarray:
    sl = slice_2d
    if sl.size == 0:
        return None
    if sl.dtype != np.uint8:
        sl = sl.astype(np.float32)
        mn = float(np.min(sl))
        mx = float(np.max(sl))
        if mx <= mn + 1e-6:
            sl = np.zeros_like(sl, dtype=np.uint8)
        else:
            sl = (sl - mn) / (mx - mn + 1e-6) * 255.0
            sl = np.clip(sl, 0, 255).astype(np.uint8)
    rgb = np.stack([sl] * 3, axis=-1)
    return rgb


def write_jpeg_frames(frames_rgb: list[np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(out_dir):
        fp = os.path.join(out_dir, fn)
        if os.path.isfile(fp):
            os.remove(fp)

    for i, fr in enumerate(frames_rgb):
        path = os.path.join(out_dir, f"{i:05d}.jpg")
        Image.fromarray(fr).save(path, quality=95, subsampling=0)
