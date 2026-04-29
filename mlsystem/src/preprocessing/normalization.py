from __future__ import annotations

import numpy as np


def normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    arr[~np.isfinite(arr)] = 0.0
    out = np.zeros_like(arr, dtype="float32")
    for band in range(arr.shape[0]):
        data = arr[band]
        valid = data[data != 0]
        if valid.size < 16:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[band] = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return out
