from __future__ import annotations

import os

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
        sample = _percentile_sample(valid)
        lo, hi = np.percentile(sample, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[band] = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return out


def _percentile_sample(values: np.ndarray) -> np.ndarray:
    max_samples = int(os.getenv("INFERENCE_ENGINE_NORMALIZE_MAX_SAMPLES", "65536"))
    if max_samples <= 0 or values.size <= max_samples:
        return values
    step = max(1, values.size // max_samples)
    return values[::step][:max_samples]
