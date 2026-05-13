from __future__ import annotations

from typing import Any

import numpy as np
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box

from .records import TileSampleRecord, TileWindow


def rasterize_mask_for_window(
    ds: Any,
    geometries: list[Any],
    window: TileWindow | TileSampleRecord,
    *,
    all_touched: bool = False,
) -> tuple[np.ndarray, int]:
    raster_window = Window(int(window.x), int(window.y), int(window.width), int(window.height))
    window_bounds = box(*ds.window_bounds(raster_window))
    selected = [geom for geom in geometries if geom.is_valid and not geom.is_empty and geom.intersects(window_bounds)]
    if not selected:
        return np.zeros((int(window.height), int(window.width)), dtype="uint8"), 0
    mask = rasterize(
        [(geom, 1) for geom in selected],
        out_shape=(int(window.height), int(window.width)),
        transform=ds.window_transform(raster_window),
        fill=0,
        dtype="uint8",
        all_touched=bool(all_touched),
    )
    mask = (mask > 0).astype("uint8")
    return mask, len(selected)


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
