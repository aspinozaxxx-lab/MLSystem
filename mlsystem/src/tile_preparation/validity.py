from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rasterio.windows import Window


@dataclass
class ValidDataMaskResult:
    mask: np.ndarray
    source: str

    @property
    def valid_pixel_share(self) -> float:
        total = int(self.mask.size)
        return float(np.count_nonzero(self.mask)) / total if total else 0.0


def read_valid_data_mask(
    ds: Any,
    window: Any,
    *,
    mode: str = "auto",
    nodata_values: list[int | float] | None = None,
) -> np.ndarray:
    return read_valid_data_mask_with_source(ds, window, mode=mode, nodata_values=nodata_values).mask


def read_valid_data_mask_with_source(
    ds: Any,
    window: Any,
    *,
    mode: str = "auto",
    nodata_values: list[int | float] | None = None,
) -> ValidDataMaskResult:
    raster_window = _window(window)
    mode = str(mode or "auto").lower()
    if mode == "dataset_mask":
        return ValidDataMaskResult(_dataset_mask(ds, raster_window), "dataset_mask")
    if mode == "alpha":
        return ValidDataMaskResult(_alpha_mask(ds, raster_window), "alpha")
    if mode == "nodata":
        return ValidDataMaskResult(_nodata_mask(ds, raster_window, nodata_values), "nodata")
    if mode == "nonzero_any":
        return ValidDataMaskResult(_nonzero_mask(ds, raster_window, any_band=True), "nonzero_any")
    if mode == "nonzero_all":
        return ValidDataMaskResult(_nonzero_mask(ds, raster_window, any_band=False), "nonzero_all")
    if mode != "auto":
        raise ValueError(f"Unsupported valid pixel mode: {mode}")

    for source, fn in (
        ("dataset_mask", lambda: _dataset_mask(ds, raster_window)),
        ("alpha", lambda: _alpha_mask(ds, raster_window)),
        ("nodata", lambda: _nodata_mask(ds, raster_window, nodata_values)),
    ):
        try:
            mask = fn()
        except Exception:  # noqa: BLE001
            continue
        if mask.any() and not mask.all():
            return ValidDataMaskResult(mask, source)

    nonzero = _nonzero_mask(ds, raster_window, any_band=True)
    if nonzero.any():
        return ValidDataMaskResult(nonzero, "nonzero_any")
    try:
        mask = _dataset_mask(ds, raster_window)
        return ValidDataMaskResult(mask, "dataset_mask")
    except Exception:  # noqa: BLE001
        shape = (int(raster_window.height), int(raster_window.width))
        return ValidDataMaskResult(np.zeros(shape, dtype="uint8"), "empty")


def _dataset_mask(ds: Any, window: Window) -> np.ndarray:
    try:
        return (ds.dataset_mask(window=window, boundless=True) > 0).astype("uint8")
    except ValueError:
        return (ds.dataset_mask(window=window) > 0).astype("uint8")


def _alpha_mask(ds: Any, window: Window) -> np.ndarray:
    alpha_bands = _alpha_band_indexes(ds)
    if not alpha_bands:
        raise ValueError("dataset has no alpha band")
    try:
        alpha = ds.read(alpha_bands[0], window=window, boundless=True, fill_value=0)
    except ValueError:
        alpha = ds.read(alpha_bands[0], window=window)
    return (alpha > 0).astype("uint8")


def _nodata_mask(ds: Any, window: Window, nodata_values: list[int | float] | None) -> np.ndarray:
    if nodata_values is not None:
        try:
            arr = ds.read(window=window, boundless=True, fill_value=0)
        except ValueError:
            arr = ds.read(window=window)
        valid = np.ones(arr.shape[1:], dtype=bool)
        for value in nodata_values:
            valid &= ~np.all(arr == value, axis=0)
        return valid.astype("uint8")
    try:
        masks = ds.read_masks(window=window, boundless=True)
    except ValueError:
        masks = ds.read_masks(window=window)
    if masks.size == 0:
        raise ValueError("dataset has no masks")
    return np.all(masks > 0, axis=0).astype("uint8")


def _nonzero_mask(ds: Any, window: Window, *, any_band: bool) -> np.ndarray:
    try:
        arr = ds.read(window=window, boundless=True, fill_value=0)
    except ValueError:
        arr = ds.read(window=window)
    if arr.size == 0:
        return np.zeros((int(window.height), int(window.width)), dtype="uint8")
    if any_band:
        return np.any(arr != 0, axis=0).astype("uint8")
    return np.all(arr != 0, axis=0).astype("uint8")


def _alpha_band_indexes(ds: Any) -> list[int]:
    try:
        from rasterio.enums import ColorInterp

        return [idx + 1 for idx, value in enumerate(ds.colorinterp or []) if value == ColorInterp.alpha]
    except Exception:  # noqa: BLE001
        return []


def _window(window: Any) -> Window:
    if isinstance(window, Window):
        return window
    return Window(int(window.x), int(window.y), int(window.width), int(window.height))
