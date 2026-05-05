from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from affine import Affine

from ..storage.local_io import write_json
from .contracts import PredictionTileInfo, ProcessingBlock, VectorizationPlan
from .tile_index import window_bounds


def build_processing_blocks(
    tiles: list[PredictionTileInfo],
    *,
    core_size_px: int,
    halo_px: int,
) -> list[ProcessingBlock]:
    blocks: list[ProcessingBlock] = []
    for tile in tiles:
        transform = Affine(*tile.transform)
        rows = max(1, math.ceil(tile.height / core_size_px))
        cols = max(1, math.ceil(tile.width / core_size_px))
        for row in range(rows):
            for col in range(cols):
                col_off = col * core_size_px
                row_off = row * core_size_px
                core_w = min(core_size_px, tile.width - col_off)
                core_h = min(core_size_px, tile.height - row_off)
                exp_col = max(0, col_off - halo_px)
                exp_row = max(0, row_off - halo_px)
                exp_end_col = min(tile.width, col_off + core_w + halo_px)
                exp_end_row = min(tile.height, row_off + core_h + halo_px)
                block_id = f"{tile.tile_id}_{row:04d}_{col:04d}"
                blocks.append(
                    ProcessingBlock(
                        block_id=block_id,
                        scene_id=tile.scene_id,
                        core_window=(int(col_off), int(row_off), int(core_w), int(core_h)),
                        expanded_window=(int(exp_col), int(exp_row), int(exp_end_col - exp_col), int(exp_end_row - exp_row)),
                        core_bbox=window_bounds(transform, col_off, row_off, core_w, core_h),
                        expanded_bbox=window_bounds(transform, exp_col, exp_row, exp_end_col - exp_col, exp_end_row - exp_row),
                        crs=tile.crs,
                        intersecting_tile_ids=[tile.tile_id],
                    )
                )
    return blocks


def build_vectorization_plan(
    tiles: list[PredictionTileInfo],
    *,
    core_size_px: int,
    halo_px: int,
    workers_requested: int,
    workers_effective: int,
    memory_guard: dict[str, Any],
) -> VectorizationPlan:
    blocks = build_processing_blocks(tiles, core_size_px=core_size_px, halo_px=halo_px)
    crs_values = sorted({tile.crs for tile in tiles if tile.crs})
    bounds = _union_bounds([tile.bounds for tile in tiles])
    return VectorizationPlan(
        tiles_total=len(tiles),
        blocks_total=len(blocks),
        crs=crs_values[0] if len(crs_values) == 1 else None,
        bounds=bounds,
        core_size_px=int(core_size_px),
        halo_px=int(halo_px),
        workers_requested=int(workers_requested),
        workers_effective=int(workers_effective),
        memory_guard=memory_guard,
        blocks=blocks,
    )


def write_processing_blocks_geojson(path: str | Path, blocks: list[ProcessingBlock]) -> None:
    features = []
    for block in blocks:
        x0, y0, x1, y1 = block.core_bbox
        features.append(
            {
                "type": "Feature",
                "properties": {"block_id": block.block_id, "scene_id": block.scene_id},
                "geometry": {"type": "Polygon", "coordinates": [[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]]},
            }
        )
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")


def write_block_tile_intersections(path: str | Path, blocks: list[ProcessingBlock]) -> None:
    write_json(path, {block.block_id: block.intersecting_tile_ids for block in blocks})


def _union_bounds(bounds: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not bounds:
        return None
    return (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )
