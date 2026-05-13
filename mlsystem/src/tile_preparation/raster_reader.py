from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.windows import Window

from .config import TilePreparationConfig
from .records import TileSampleRecord


def read_rgb_window(ds: Any, record: TileSampleRecord, *, bands: list[int] | None = None) -> np.ndarray:
    usable_bands = bands or list(range(1, min(3, int(ds.count)) + 1))
    usable_bands = [band for band in usable_bands if 1 <= int(band) <= int(ds.count)]
    if not usable_bands:
        raise ValueError(f"{record.scene_id} has no readable bands")
    arr = ds.read(
        usable_bands,
        window=Window(int(record.x), int(record.y), int(record.width), int(record.height)),
        boundless=True,
        fill_value=0,
    )
    return to_rgb_uint8(arr)


def read_training_image(ds: Any, record: TileSampleRecord, config: TilePreparationConfig) -> np.ndarray:
    rgb = read_rgb_window(ds, record, bands=config.input_bands)
    if config.output_format == "hwc_uint8":
        return rgb
    data = rgb.astype("float32")
    if config.normalize:
        data = data / 255.0
    if config.output_format == "chw_float32":
        return np.transpose(data, (2, 0, 1)).astype("float32")
    if config.output_format == "hwc_float32":
        return data.astype("float32")
    raise ValueError(f"Unsupported output_format: {config.output_format}")


def to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected band-first raster array, got shape={arr.shape}")
    if arr.shape[0] == 1:
        rgb = np.repeat(arr[:1], 3, axis=0)
    elif arr.shape[0] == 2:
        rgb = np.concatenate([arr, arr[:1]], axis=0)
    else:
        rgb = arr[:3]
    if rgb.dtype == np.uint8:
        return np.transpose(rgb, (1, 2, 0)).copy()
    data = rgb.astype("float32")
    out = np.zeros_like(data, dtype="uint8")
    for idx in range(data.shape[0]):
        band = data[idx]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            lo, hi = float(valid.min()), float(valid.max() or 1.0)
        out[idx] = np.clip((band - lo) / max(1e-6, hi - lo) * 255.0, 0, 255).astype("uint8")
    return np.transpose(out, (1, 2, 0))
