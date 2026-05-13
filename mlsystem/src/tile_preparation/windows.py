from __future__ import annotations

import math
from typing import Any

from .records import TileWindow


def axis_origins(length: int, tile_size: int, stride: int) -> list[int]:
    length = int(length)
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))
    if length <= tile_size:
        return [0]
    origins = list(range(0, max(1, length - tile_size + 1), stride))
    last = max(0, length - tile_size)
    if origins[-1] != last:
        origins.append(last)
    return origins


def expected_axis_count(length: int, tile_size: int, stride: int) -> int:
    if int(length) <= int(tile_size):
        return 1
    return int(math.ceil((int(length) - int(tile_size)) / max(1, int(stride)))) + 1


def generate_window_grid_for_scene(
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    *,
    scene_id: str = "",
) -> list[TileWindow]:
    windows: list[TileWindow] = []
    for y in axis_origins(height, tile_size, stride):
        for x in axis_origins(width, tile_size, stride):
            windows.append(
                TileWindow(
                    scene_id=scene_id,
                    x=int(x),
                    y=int(y),
                    width=int(tile_size),
                    height=int(tile_size),
                    tile_size=int(tile_size),
                    stride=max(1, int(stride)),
                    index=len(windows),
                )
            )
    return windows


def build_tiling_check(width: int, height: int, tile_size: int, stride: int, *, label: str, stride_factor: float = 1.0) -> dict[str, Any]:
    effective_stride = max(1, int(max(1, int(stride)) * max(0.000001, float(stride_factor))))
    windows = generate_window_grid_for_scene(width, height, tile_size, effective_stride, scene_id="tiling_check")
    xs = sorted({window.x for window in windows})
    ys = sorted({window.y for window in windows})
    expected_nx = expected_axis_count(width, tile_size, effective_stride)
    expected_ny = expected_axis_count(height, tile_size, effective_stride)
    out_of_bounds = [
        window
        for window in windows
        if window.x < 0 or window.y < 0 or window.x + window.width > width or window.y + window.height > height
    ]
    coverage_x = max((window.x + window.width for window in windows), default=0)
    coverage_y = max((window.y + window.height for window in windows), default=0)
    expected_total = expected_nx * expected_ny
    actual_total = len(windows)
    return {
        "stride_type": label,
        "stride_factor": float(stride_factor),
        "effective_stride": int(effective_stride),
        "expected_nx": int(expected_nx),
        "actual_nx": int(len(xs)),
        "expected_ny": int(expected_ny),
        "actual_ny": int(len(ys)),
        "expected_total": int(expected_total),
        "actual_total": int(actual_total),
        "coverage_x": int(coverage_x),
        "coverage_y": int(coverage_y),
        "last_x": int(xs[-1]) if xs else None,
        "last_y": int(ys[-1]) if ys else None,
        "out_of_bounds_windows": int(len(out_of_bounds)),
        "pass": bool(
            expected_nx == len(xs)
            and expected_ny == len(ys)
            and expected_total == actual_total
            and coverage_x == width
            and coverage_y == height
            and not out_of_bounds
        ),
    }
