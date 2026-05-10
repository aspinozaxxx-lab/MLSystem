from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from affine import Affine
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from ..api.schemas import PseudolabelConfig
from ..contracts import ProbabilityMap
from ..planning.planner import BlockDescriptor, ScenePlan, TileDescriptor
from ..postprocessing.vectorization import vectorize_probability_map
from ..storage.artifacts import checksum_matches, write_checksum
from ..storage.local_io import write_json


def materialize_expanded_block(block: BlockDescriptor, plan: ScenePlan, tiles_by_id: dict[str, TileDescriptor]) -> dict[str, Any]:
    out_path = Path(block.artifact_path)
    checksum_path = out_path.with_suffix(out_path.suffix + ".sha256")
    if checksum_matches(out_path, checksum_path):
        return {"block_id": block.block_id, "artifact_path": str(out_path), "idempotent_hit": True}

    exp_x, exp_y, exp_w, exp_h = block.expanded_window
    canvas = np.zeros((exp_h, exp_w), dtype=np.float32)
    weight = np.zeros((exp_h, exp_w), dtype=np.float32)
    for tile_id in block.dependency_tile_ids:
        tile = tiles_by_id[tile_id]
        with np.load(tile.artifact_path) as payload:
            tile_prob = payload["prob_uint8"].astype(np.float32) / 255.0 if "prob_uint8" in payload.files else payload["prob"].astype(np.float32)
        tx0, ty0, tw, th = tile.x, tile.y, tile.width, tile.height
        ix0 = max(exp_x, tx0)
        iy0 = max(exp_y, ty0)
        ix1 = min(exp_x + exp_w, tx0 + tw)
        iy1 = min(exp_y + exp_h, ty0 + th)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        src = tile_prob[iy0 - ty0 : iy1 - ty0, ix0 - tx0 : ix1 - tx0]
        dst_y = iy0 - exp_y
        dst_x = ix0 - exp_x
        canvas[dst_y : dst_y + src.shape[0], dst_x : dst_x + src.shape[1]] += src
        weight[dst_y : dst_y + src.shape[0], dst_x : dst_x + src.shape[1]] += 1.0
    valid = weight > 0
    if np.any(valid):
        canvas[valid] = canvas[valid] / np.maximum(weight[valid], 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, prob=canvas.astype(np.float32, copy=False), weight_sum=weight.astype(np.float32, copy=False))
    checksum = write_checksum(out_path, checksum_path)
    return {"block_id": block.block_id, "artifact_path": str(out_path), "checksum": checksum}


def vectorize_expanded_block(block: BlockDescriptor, plan: ScenePlan, vector_cfg: PseudolabelConfig) -> dict[str, Any]:
    started = time.time()
    vector_path = Path(block.vector_path)
    summary_path = Path(block.summary_path)
    if vector_path.exists() and summary_path.exists():
        return {"block_id": block.block_id, "status": "success", "vector_path": str(vector_path), "summary_path": str(summary_path), "idempotent_hit": True}

    with np.load(block.artifact_path) as payload:
        prob = payload["prob"].astype(np.float32)
        weight_sum = payload["weight_sum"].astype(np.float32) if "weight_sum" in payload.files else np.ones(prob.shape, dtype=np.float32)
    exp_x, exp_y, _exp_w, _exp_h = block.expanded_window
    transform = Affine(*plan.transform) * Affine.translation(exp_x, exp_y)
    coverage_mask = weight_sum > 0
    probability_map = ProbabilityMap(
        scene_id=plan.scene_id,
        prob=prob,
        weight_sum=weight_sum,
        coverage_mask=coverage_mask,
        coverage_fraction=float(np.count_nonzero(coverage_mask) / max(1, coverage_mask.size)),
        transform=transform,
        crs=plan.crs,
        metadata={"block_id": block.block_id, "scene_name": plan.scene_name},
    )
    vectorized = vectorize_probability_map(
        probability_map,
        scene_name=plan.scene_name,
        threshold=float(vector_cfg.threshold),
        min_area_m2=float(vector_cfg.local_min_area or 0.0),
    )
    core_box = box(*block.core_bbox)
    features, boundary_candidates = _clip_features_to_core(vectorized.features_raw, core_box, block)
    payload = {"type": "FeatureCollection", "features": features}
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    vector_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    summary = {
        "block_id": block.block_id,
        "status": "success",
        "output_vector_path": str(vector_path),
        "summary_path": str(summary_path),
        "polygons_before_clip": int(len(vectorized.features_raw)),
        "polygons_after_clip": int(len(features)),
        "polygons_after_local_filter": int(len(features)),
        "boundary_candidates_count": int(boundary_candidates),
        "duration_sec": round(time.time() - started, 3),
    }
    write_json(summary_path, summary)
    return summary


def _clip_features_to_core(features: list[dict[str, Any]], core_box: Any, block: BlockDescriptor) -> tuple[list[dict[str, Any]], int]:
    clipped_features: list[dict[str, Any]] = []
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
        props = dict(feature.get("properties") or {})
        props.update(
            {
                "block_id": block.block_id,
                "scene_id": block.scene_id,
                "polygon_touches_core_boundary": bool(clipped.touches(core_box.boundary)),
                "polygon_was_clipped_by_core": bool(not clipped.equals(geom)),
            }
        )
        if props["polygon_touches_core_boundary"] or props["polygon_was_clipped_by_core"]:
            boundary_candidates += 1
        clipped_features.append({"type": "Feature", "properties": props, "geometry": mapping(clipped)})
    return clipped_features, boundary_candidates
