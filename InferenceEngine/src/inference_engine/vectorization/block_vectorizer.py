from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from affine import Affine
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from ..contracts import ProbabilityMap
from ..postprocessing.vectorization import METRIC_CRS, vectorize_probability_map
from ..storage.local_io import write_json
from .contracts import BlockVectorizationJob, BlockVectorizationResult, PredictionTileInfo
from .tile_index import load_tile_probability


def vectorize_block(job: BlockVectorizationJob) -> BlockVectorizationResult:
    started = time.time()
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / f"{job.block.block_id}.geojson"
    summary_path = output_dir / f"{job.block.block_id}.summary.json"
    warnings: list[str] = []
    try:
        tile = _single_tile_for_block(job)
        if not tile.crs:
            warnings.append("tile CRS is missing; geometry is treated as already metric/projected")
        prob, block_transform, coverage_fraction = _expanded_probability(tile, job.block.expanded_window)
        probability_map = ProbabilityMap(
            scene_id=tile.scene_id,
            prob=prob,
            weight_sum=np.ones(prob.shape, dtype=np.float32),
            coverage_mask=np.ones(prob.shape, dtype=bool),
            coverage_fraction=coverage_fraction,
            transform=block_transform,
            crs=tile.crs,
            metadata={"block_id": job.block.block_id, "scene_name": tile.scene_name},
        )
        result = vectorize_probability_map(
            probability_map,
            scene_name=tile.scene_name,
            threshold=float(job.threshold),
            min_area_m2=float(job.local_min_area or 0.0),
        )
        core_box = _metric_core_box(job.block.core_bbox, tile.crs or job.block.crs)
        features, boundary_candidates = _clip_features_to_core(result.features_raw, core_box, job)
        payload = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": result.crs}},
            "features": features,
        }
        vector_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        block_result = BlockVectorizationResult(
            block_id=job.block.block_id,
            status="success",
            output_vector_path=str(vector_path),
            summary_path=str(summary_path),
            polygons_before_clip=int(len(result.features_raw)),
            polygons_after_clip=int(len(features)),
            polygons_after_local_filter=int(len(features)),
            boundary_candidates_count=int(boundary_candidates),
            duration_sec=round(time.time() - started, 3),
            warnings=warnings,
        )
        write_json(summary_path, block_result.to_dict())
        return block_result
    except Exception as exc:  # noqa: BLE001 - worker errors must be serialized.
        block_result = BlockVectorizationResult(
            block_id=job.block.block_id,
            status="failed",
            output_vector_path=str(vector_path),
            summary_path=str(summary_path),
            duration_sec=round(time.time() - started, 3),
            warnings=warnings,
            errors=[f"{type(exc).__name__}: {exc}"],
        )
        write_json(summary_path, block_result.to_dict())
        return block_result


def _single_tile_for_block(job: BlockVectorizationJob) -> PredictionTileInfo:
    tile_ids = set(job.block.intersecting_tile_ids)
    for tile in job.tiles:
        if tile.tile_id in tile_ids:
            return tile
    if job.tiles:
        return job.tiles[0]
    raise ValueError(f"No prediction tiles for block {job.block.block_id}")


def _expanded_probability(tile: PredictionTileInfo, window: tuple[int, int, int, int]) -> tuple[np.ndarray, Affine, float]:
    col_off, row_off, width, height = window
    prob = load_tile_probability(tile)
    subset = prob[row_off : row_off + height, col_off : col_off + width].astype(np.float32, copy=False)
    coverage_fraction = float(np.count_nonzero(np.isfinite(subset)) / subset.size) if subset.size else 0.0
    transform = Affine(*tile.transform) * Affine.translation(col_off, row_off)
    return subset, transform, coverage_fraction


def _metric_core_box(core_bbox: tuple[float, float, float, float], source_crs: str | None) -> object:
    geom = mapping(box(*core_bbox))
    if not source_crs or "3857" in str(source_crs):
        return shape(geom)
    transformed = transform_geom(str(source_crs), METRIC_CRS, geom, precision=-1)
    return shape(transformed)


def _clip_features_to_core(features: list[dict], core_box: object, job: BlockVectorizationJob) -> tuple[list[dict], int]:
    clipped_features: list[dict] = []
    boundary_candidates = 0
    for feature in features:
        geom = shape(feature.get("geometry"))
        if geom.is_empty:
            continue
        clipped = geom.intersection(core_box)
        if clipped.is_empty:
            continue
        try:
            clipped = unary_union([clipped])
        except Exception:
            clipped = clipped.buffer(0)
        if clipped.is_empty:
            continue
        if float(job.local_min_area or 0.0) > 0 and clipped.area < float(job.local_min_area):
            continue
        props = dict(feature.get("properties") or {})
        props.update(
            {
                "block_id": job.block.block_id,
                "scene_id": job.block.scene_id,
                "polygon_touches_core_boundary": bool(clipped.touches(core_box.boundary)),
                "polygon_was_clipped_by_core": bool(not clipped.equals(geom)),
            }
        )
        if props["polygon_touches_core_boundary"] or props["polygon_was_clipped_by_core"]:
            boundary_candidates += 1
        clipped_features.append({"type": "Feature", "properties": props, "geometry": mapping(clipped)})
    return clipped_features, boundary_candidates
