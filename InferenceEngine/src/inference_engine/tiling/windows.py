from __future__ import annotations

from typing import Any

import numpy as np

from ..contracts import TileWindow


def origins(length: int, tile: int, stride: int) -> list[int]:
    if tile <= 0:
        raise ValueError("tile must be positive")
    stride = max(1, int(stride))
    if length <= tile:
        return [0]
    values = list(range(0, max(1, length - tile + 1), stride))
    edge = length - tile
    if values[-1] != edge:
        values.append(edge)
    return sorted(set(max(0, int(value)) for value in values))


def window_grid(width: int, height: int, tile: int, stride: int, scene_id: str = "") -> list[TileWindow]:
    xs = origins(width, tile, stride)
    ys = origins(height, tile, stride)
    windows: list[TileWindow] = []
    for y in ys:
        for x in xs:
            actual_w = min(tile, width - x)
            actual_h = min(tile, height - y)
            windows.append(
                TileWindow(
                    scene_id=scene_id,
                    x=int(x),
                    y=int(y),
                    width=int(actual_w),
                    height=int(actual_h),
                    tile_size=int(tile),
                    stride=int(stride),
                    is_edge=bool(x == 0 or y == 0 or x + actual_w >= width or y + actual_h >= height),
                    index=len(windows),
                )
            )
    return windows


def tile_insert_slices(
    x: int,
    y: int,
    actual_w: int,
    actual_h: int,
    scene_width: int,
    scene_height: int,
    *,
    patch_size: int,
    crop_mode: str = "full",
    center_size: int | None = None,
    context_bounds: int | None = None,
) -> dict[str, int]:
    if crop_mode != "center":
        crop_left = crop_top = crop_right = crop_bottom = 0
    else:
        margin = int(context_bounds if context_bounds is not None else max(0, (patch_size - int(center_size or patch_size)) // 2))
        crop_left = margin if x > 0 else 0
        crop_top = margin if y > 0 else 0
        crop_right = margin if x + actual_w < scene_width else 0
        crop_bottom = margin if y + actual_h < scene_height else 0
        if actual_w - crop_left - crop_right <= 0:
            crop_left = crop_right = 0
        if actual_h - crop_top - crop_bottom <= 0:
            crop_top = crop_bottom = 0

    crop_x0 = int(crop_left)
    crop_y0 = int(crop_top)
    crop_x1 = int(actual_w - crop_right)
    crop_y1 = int(actual_h - crop_bottom)
    insert_x = int(x + crop_x0)
    insert_y = int(y + crop_y0)
    insert_w = int(crop_x1 - crop_x0)
    insert_h = int(crop_y1 - crop_y0)
    return {
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "insert_x": insert_x,
        "insert_y": insert_y,
        "insert_width": insert_w,
        "insert_height": insert_h,
    }


def tile_weight_window(
    x: int,
    y: int,
    actual_w: int,
    actual_h: int,
    scene_width: int,
    scene_height: int,
    *,
    patch_size: int,
    center_size: int | None = None,
    context_bounds: int | None = None,
) -> np.ndarray:
    margin = int(context_bounds if context_bounds is not None else max(1, (patch_size - int(center_size or patch_size)) // 2))
    ramp = max(1, margin)
    wx = np.ones(actual_w, dtype="float32")
    wy = np.ones(actual_h, dtype="float32")
    if x > 0:
        n = min(ramp, actual_w)
        wx[:n] = np.linspace(0.05, 1.0, n, dtype="float32")
    if x + actual_w < scene_width:
        n = min(ramp, actual_w)
        wx[-n:] = np.minimum(wx[-n:], np.linspace(1.0, 0.05, n, dtype="float32"))
    if y > 0:
        n = min(ramp, actual_h)
        wy[:n] = np.linspace(0.05, 1.0, n, dtype="float32")
    if y + actual_h < scene_height:
        n = min(ramp, actual_h)
        wy[-n:] = np.minimum(wy[-n:], np.linspace(1.0, 0.05, n, dtype="float32"))
    return wy[:, None] * wx[None, :]


def window_to_legacy_tuple(window: TileWindow) -> tuple[int, int]:
    return window.x, window.y


def legacy_window_grid(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    return [window_to_legacy_tuple(window) for window in window_grid(width, height, tile, stride)]


def tile_insert_debug_properties(insert: dict[str, int], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {**(extra or {}), **insert, "expected_insert_bounds": [insert["insert_x"], insert["insert_y"], insert["insert_x"] + insert["insert_width"], insert["insert_y"] + insert["insert_height"]]}
