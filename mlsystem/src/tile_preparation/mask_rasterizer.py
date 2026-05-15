from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box

from .geometry_index import filter_geometries_linear
from .records import TileSampleRecord, TileWindow


@dataclass
class MaskRasterizationResult:
    mask: np.ndarray
    raw_mask: np.ndarray
    geom_count: int
    raw_positive_pixels: int
    clipped_positive_pixels: int
    valid_pixel_share: float

    def __iter__(self):
        yield self.mask
        yield self.geom_count


def rasterize_mask_for_window(
    ds: Any,
    geometries: list[Any],
    window: TileWindow | TileSampleRecord,
    *,
    all_touched: bool = False,
    valid_mask: np.ndarray | None = None,
    geometry_index: Any | None = None,
    profile: dict[str, float] | None = None,
) -> MaskRasterizationResult:
    raster_window = Window(int(window.x), int(window.y), int(window.width), int(window.height))
    window_bounds = box(*ds.window_bounds(raster_window))
    query_started = time.perf_counter()
    if geometry_index is not None:
        selected = geometry_index.query(window_bounds)
    else:
        selected = filter_geometries_linear(geometries, window_bounds)
    _add_profile_value(profile, "geometry_index_query_sec", time.perf_counter() - query_started)
    burn_started = time.perf_counter()
    if not selected:
        raw_mask = np.zeros((int(window.height), int(window.width)), dtype="uint8")
    else:
        raw_mask = rasterize(
            [(geom, 1) for geom in selected],
            out_shape=(int(window.height), int(window.width)),
            transform=ds.window_transform(raster_window),
            fill=0,
            dtype="uint8",
            all_touched=bool(all_touched),
        )
        raw_mask = (raw_mask > 0).astype("uint8")
    _add_profile_value(profile, "rasterize_burn_sec", time.perf_counter() - burn_started)
    if valid_mask is not None:
        clip_started = time.perf_counter()
        valid = (np.asarray(valid_mask) > 0).astype("uint8")
        if valid.shape != raw_mask.shape:
            raise ValueError(f"valid_mask shape {valid.shape} does not match rasterized mask shape {raw_mask.shape}")
        mask = (raw_mask & valid).astype("uint8")
        valid_share = float(np.count_nonzero(valid)) / int(valid.size) if valid.size else 0.0
        _add_profile_value(profile, "valid_clip_sec", time.perf_counter() - clip_started)
    else:
        mask = raw_mask
        valid_share = 1.0
    return MaskRasterizationResult(
        mask=mask,
        raw_mask=raw_mask,
        geom_count=len(selected),
        raw_positive_pixels=int(np.count_nonzero(raw_mask)),
        clipped_positive_pixels=int(np.count_nonzero(mask)),
        valid_pixel_share=valid_share,
    )


def _add_profile_value(profile: dict[str, float] | None, name: str, seconds: float) -> None:
    if profile is not None:
        profile[str(name)] = float(profile.get(str(name), 0.0)) + float(seconds)


def positive_pixel_bbox(mask: np.ndarray, *, x_offset: int = 0, y_offset: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return (
        int(xs.min()) + int(x_offset),
        int(ys.min()) + int(y_offset),
        int(xs.max()) + 1 + int(x_offset),
        int(ys.max()) + 1 + int(y_offset),
    )
