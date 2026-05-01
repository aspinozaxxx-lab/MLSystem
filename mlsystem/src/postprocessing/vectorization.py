from __future__ import annotations

from typing import Any

from rasterio.features import shapes as raster_shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from ..pipeline.contracts import ProbabilityMap, VectorizationResult
from .thresholding import threshold_probability_map


METRIC_CRS = "EPSG:3857"


def vertex_count(geom_mapping: dict[str, Any]) -> int:
    coords = geom_mapping.get("coordinates") or []
    if geom_mapping.get("type") == "Polygon":
        return sum(len(ring) for ring in coords)
    if geom_mapping.get("type") == "MultiPolygon":
        return sum(len(ring) for poly in coords for ring in poly)
    return 0


def _normalize_crs(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).upper()
    if "3857" in text:
        return METRIC_CRS
    if "4326" in text:
        return "EPSG:4326"
    return str(value)


def _to_metric_geometry(geom: dict[str, Any], source_crs: Any) -> dict[str, Any]:
    normalized = _normalize_crs(source_crs)
    if not normalized or normalized == METRIC_CRS:
        return geom
    return transform_geom(normalized, METRIC_CRS, geom, precision=-1)


def vectorize_mask(
    mask: Any,
    transform: Any,
    *,
    scene_name: str,
    threshold: float,
    source_crs: Any = None,
    min_area_m2: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    features: list[dict[str, Any]] = []
    vertices = 0
    min_area = float(min_area_m2 or 0)
    for geom, value in raster_shapes(mask, mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue
        metric_geom = _to_metric_geometry(geom, source_crs)
        poly = shape(metric_geom)
        try:
            poly = make_valid(poly)
        except Exception:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        area_m2 = float(poly.area)
        if min_area > 0 and area_m2 < min_area:
            continue
        mapped = mapping(poly)
        vertices += vertex_count(mapped)
        features.append(
            {
                "type": "Feature",
                "properties": {"scene": scene_name, "threshold": float(threshold), "area_m2": area_m2},
                "geometry": mapped,
            }
        )
    return features, vertices


def vectorize_probability_map(
    probability_map: ProbabilityMap,
    *,
    scene_name: str,
    threshold: float,
    min_area_m2: float | None = None,
) -> VectorizationResult:
    mask = threshold_probability_map(probability_map.prob, threshold)
    features, vertices = vectorize_mask(
        mask,
        probability_map.transform,
        scene_name=scene_name,
        threshold=threshold,
        source_crs=probability_map.crs,
        min_area_m2=min_area_m2,
    )
    return VectorizationResult(
        features_raw=features,
        raw_count=len(features),
        vertices_before=vertices,
        threshold=float(threshold),
        crs=METRIC_CRS,
        metadata={
            "coverage_fraction": probability_map.coverage_fraction,
            "source_crs": probability_map.crs,
            "min_area_m2_prefilter": float(min_area_m2 or 0),
        },
    )
