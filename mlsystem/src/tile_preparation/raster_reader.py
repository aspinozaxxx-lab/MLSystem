from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.enums import Resampling
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
    arr = read_band_window(ds, record, bands=config.input_bands)
    return format_training_image(arr, config)


def format_training_image(
    arr: np.ndarray,
    config: TilePreparationConfig,
    *,
    scene_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if config.output_format == "hwc_uint8":
        return np.transpose(_normalize_to_uint8(arr), (1, 2, 0))
    data = arr.astype("float32", copy=False)
    if config.normalize:
        mode = str(getattr(config, "normalization_mode", "uint8_255") or "uint8_255").lower()
        if mode == "uint8_255":
            data = normalize_uint8_255(data)
        elif mode == "scene_percentile":
            data = normalize_band_first(data, scene_stats=scene_stats)
        elif mode == "tile_percentile":
            data = normalize_band_first(data)
        else:
            raise ValueError(f"Unsupported normalization_mode: {mode}")
    if config.output_format == "chw_float32":
        return data.astype("float32")
    if config.output_format == "hwc_float32":
        return np.transpose(data, (1, 2, 0)).astype("float32")
    raise ValueError(f"Unsupported output_format: {config.output_format}")


def read_band_window(ds: Any, record: TileSampleRecord, *, bands: list[int] | None = None) -> np.ndarray:
    usable_bands = bands or list(range(1, int(ds.count) + 1))
    usable_bands = [int(band) for band in usable_bands if 1 <= int(band) <= int(ds.count)]
    if not usable_bands:
        raise ValueError(f"{record.scene_id} has no readable bands")
    return ds.read(
        usable_bands,
        window=Window(int(record.x), int(record.y), int(record.width), int(record.height)),
        boundless=True,
        fill_value=0,
    )


def normalize_uint8_255(arr: np.ndarray) -> np.ndarray:
    data = arr.astype("float32", copy=False)
    data[~np.isfinite(data)] = 0.0
    return np.clip(data, 0.0, 255.0).astype("float32") / 255.0


def normalize_band_first(arr: np.ndarray, *, scene_stats: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    data = arr.astype("float32", copy=False)
    data[~np.isfinite(data)] = 0.0
    out = np.zeros_like(data, dtype="float32")
    if scene_stats is not None:
        lo_values, hi_values = scene_stats
        for band_index in range(data.shape[0]):
            lo = float(lo_values[band_index])
            hi = float(hi_values[band_index])
            if hi <= lo:
                hi = lo + 1.0
            out[band_index] = np.clip((data[band_index] - lo) / (hi - lo), 0.0, 1.0)
        return out
    for band_index in range(data.shape[0]):
        band = data[band_index]
        valid = band[band != 0]
        if valid.size < 16:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[band_index] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def compute_scene_percentile_stats(
    ds: Any,
    *,
    bands: list[int] | None = None,
    max_dimension: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    usable_bands = bands or list(range(1, int(ds.count) + 1))
    usable_bands = [int(band) for band in usable_bands if 1 <= int(band) <= int(ds.count)]
    if not usable_bands:
        raise ValueError("no readable bands for scene percentile stats")
    max_dimension = max(64, int(max_dimension))
    scale = max(1.0, max(float(ds.width), float(ds.height)) / float(max_dimension))
    out_height = max(1, int(round(float(ds.height) / scale)))
    out_width = max(1, int(round(float(ds.width) / scale)))
    arr = ds.read(
        usable_bands,
        out_shape=(len(usable_bands), out_height, out_width),
        resampling=Resampling.nearest,
        boundless=False,
    ).astype("float32", copy=False)
    arr[~np.isfinite(arr)] = 0.0
    lo_values = np.zeros((len(usable_bands),), dtype="float32")
    hi_values = np.ones((len(usable_bands),), dtype="float32")
    for band_index in range(arr.shape[0]):
        valid = arr[band_index][arr[band_index] != 0]
        if valid.size < 16:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        lo_values[band_index] = float(lo)
        hi_values[band_index] = float(hi)
    return lo_values, hi_values


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(normalize_band_first(arr) * 255.0, 0, 255).astype("uint8")


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
