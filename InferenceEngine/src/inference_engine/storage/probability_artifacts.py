from __future__ import annotations

from pathlib import Path

import numpy as np


def write_probability_uint8(path: str | Path, prob_uint8: np.ndarray) -> None:
    """Write a per-tile probability artifact without ZIP overhead.

    Existing plans still use a .npz suffix, so write through a file handle to
    keep the planned path unchanged while storing fast .npy payload bytes.
    Readers in this module support both legacy NPZ and this NPY-in-place format.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        np.save(handle, np.asarray(prob_uint8, dtype=np.uint8))


def load_probability_array(path: str | Path, *, probability_band: str | None = None, as_float: bool = True) -> tuple[np.ndarray, str]:
    payload = np.load(str(path))
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            band = _select_npz_band(payload, probability_band)
            array = payload[band]
            return _normalize_array(array, band=band, as_float=as_float), band
        finally:
            payload.close()
    array = np.asarray(payload)
    band = "prob_uint8" if array.dtype == np.uint8 else "prob"
    return _normalize_array(array, band=band, as_float=as_float), band


def probability_artifact_info(path: str | Path, *, probability_band: str | None = None) -> tuple[int, int, str]:
    array, band = load_probability_array(path, probability_band=probability_band, as_float=False)
    height, width = array.shape[:2]
    return int(height), int(width), band


def _select_npz_band(payload: np.lib.npyio.NpzFile, probability_band: str | None) -> str:
    if probability_band and probability_band in payload.files:
        return probability_band
    if "prob_uint8" in payload.files:
        return "prob_uint8"
    if "prob" in payload.files:
        return "prob"
    if payload.files:
        return str(payload.files[0])
    raise ValueError("Probability artifact has no arrays")


def _normalize_array(array: np.ndarray, *, band: str, as_float: bool) -> np.ndarray:
    if not as_float:
        return np.asarray(array)
    if band == "prob_uint8" or array.dtype == np.uint8:
        return np.asarray(array, dtype=np.float32) / 255.0
    return np.asarray(array, dtype=np.float32)
