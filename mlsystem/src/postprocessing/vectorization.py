from __future__ import annotations

from typing import Any

from rasterio.features import shapes as raster_shapes
from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from ..pipeline.contracts import ProbabilityMap, VectorizationResult
from .thresholding import threshold_probability_map


def vertex_count(geom_mapping: dict[str, Any]) -> int:
    coords = geom_mapping.get("coordinates") or []
    if geom_mapping.get("type") == "Polygon":
        return sum(len(ring) for ring in coords)
    if geom_mapping.get("type") == "MultiPolygon":
        return sum(len(ring) for poly in coords for ring in poly)
    return 0


def vectorize_mask(mask: Any, transform: Any, *, scene_name: str, threshold: float) -> tuple[list[dict[str, Any]], int]:
    features: list[dict[str, Any]] = []
    vertices = 0
    for geom, value in raster_shapes(mask, mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue
        poly = shape(geom)
        try:
            poly = make_valid(poly)
        except Exception:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        mapped = mapping(poly)
        vertices += vertex_count(mapped)
        features.append(
            {
                "type": "Feature",
                "properties": {"scene": scene_name, "threshold": float(threshold), "area_m2": float(poly.area)},
                "geometry": mapped,
            }
        )
    return features, vertices


def vectorize_probability_map(probability_map: ProbabilityMap, *, scene_name: str, threshold: float) -> VectorizationResult:
    mask = threshold_probability_map(probability_map.prob, threshold)
    features, vertices = vectorize_mask(mask, probability_map.transform, scene_name=scene_name, threshold=threshold)
    return VectorizationResult(
        features_raw=features,
        raw_count=len(features),
        vertices_before=vertices,
        threshold=float(threshold),
        crs=probability_map.crs,
        metadata={"coverage_fraction": probability_map.coverage_fraction},
    )
