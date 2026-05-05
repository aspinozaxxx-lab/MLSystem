from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from ..storage.local_io import write_json


def merge_block_vectors(
    block_vector_paths: list[str],
    *,
    output_geojson: str | Path,
    final_min_area: float = 0.0,
    merge_epsilon: float = 0.0,
) -> dict[str, Any]:
    started = time.time()
    geometries = []
    for path in block_vector_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for feature in payload.get("features") or []:
            geom = shape(feature.get("geometry"))
            if not geom.is_empty:
                geometries.append(geom)
    before_merge = len(geometries)
    area_before_merge = float(sum(geom.area for geom in geometries))
    merged = unary_union(geometries) if geometries else []
    if merge_epsilon and not getattr(merged, "is_empty", True):
        merged = merged.buffer(float(merge_epsilon)).buffer(-float(merge_epsilon))
    area_after_dissolve = float(getattr(merged, "area", 0.0) or 0.0)
    merged_geoms = _flatten_geometries(merged)
    filtered = [geom for geom in merged_geoms if not geom.is_empty and geom.area >= float(final_min_area or 0.0)]
    area_after_final_filter = float(sum(geom.area for geom in filtered))
    features = [
        {
            "type": "Feature",
            "properties": {"object_id": idx, "area_m2": float(geom.area)},
            "geometry": mapping(geom),
        }
        for idx, geom in enumerate(filtered, start=1)
    ]
    output_path = Path(output_geojson)
    output_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")
    summary = {
        "total_polygons_before_merge": before_merge,
        "total_polygons_after_dissolve": len(merged_geoms),
        "total_polygons_after_final_filter": len(features),
        "final_objects": len(features),
        "area_before_merge_m2": area_before_merge,
        "area_after_dissolve_m2": area_after_dissolve,
        "area_after_final_filter_m2": area_after_final_filter,
        "area_ratio_after_merge_to_before_merge": (area_after_dissolve / area_before_merge if area_before_merge else None),
        "area_ratio_final_to_before_merge": (area_after_final_filter / area_before_merge if area_before_merge else None),
        "final_geojson_size_mb": round(output_path.stat().st_size / (1024 * 1024), 6),
        "merge_duration_sec": round(time.time() - started, 3),
        "accepted_geojson": str(output_path),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def _flatten_geometries(geom: Any) -> list[Any]:
    if isinstance(geom, list):
        return geom
    if getattr(geom, "is_empty", True):
        return []
    geom_type = getattr(geom, "geom_type", "")
    if geom_type == "GeometryCollection" or geom_type.startswith("Multi"):
        return [item for item in geom.geoms if not item.is_empty]
    return [geom]
