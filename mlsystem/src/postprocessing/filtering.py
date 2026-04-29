from __future__ import annotations

from typing import Any

from shapely.geometry import mapping, shape

from ..pipeline.contracts import PostprocessResult, VectorizationResult
from .simplification import simplify_features


def filter_features_by_area(features: list[dict[str, Any]], min_area_m2: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for feature in features:
        geom = shape(feature["geometry"])
        if float(geom.area) < float(min_area_m2):
            continue
        properties = dict(feature.get("properties") or {})
        properties["area_m2"] = float(geom.area)
        out.append({"type": "Feature", "properties": properties, "geometry": mapping(geom)})
    return out


def limit_top_features(features: list[dict[str, Any]], max_objects: int) -> list[dict[str, Any]]:
    rows = sorted(features, key=lambda item: float((item.get("properties") or {}).get("area_m2") or 0), reverse=True)
    return rows[: max(0, int(max_objects))]


def postprocess_vectorization_result(
    vectorization: VectorizationResult,
    *,
    min_area_m2: float,
    simplify_tolerance_m: float,
    max_objects: int,
) -> PostprocessResult:
    area_filtered = filter_features_by_area(vectorization.features_raw, min_area_m2)
    simplified = simplify_features(area_filtered, simplify_tolerance_m)
    top = limit_top_features(simplified, max_objects)
    return PostprocessResult(
        features=top,
        objects_before_filter=vectorization.raw_count,
        objects_after_filter=len(simplified),
        objects_after_top=len(top),
        params={
            "threshold": vectorization.threshold,
            "min_object_area_m2": float(min_area_m2),
            "simplify_tolerance_m": float(simplify_tolerance_m),
            "max_objects": int(max_objects),
        },
        warnings=["top_limit_applied"] if len(simplified) > int(max_objects) else [],
    )
