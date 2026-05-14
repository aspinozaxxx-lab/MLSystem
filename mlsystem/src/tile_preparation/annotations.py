from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyproj import Transformer
from rasterio.crs import CRS
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform

from .config import AnnotationInput


@dataclass
class AnnotationGeometrySet:
    geometries: list[Any]
    source_crs: CRS | None
    source_crs_text: str | None
    crs_source: str
    raster_crs: CRS | None
    transformed_to_raster_crs: bool
    feature_count: int
    valid_geometry_count: int
    invalid_geometry_count: int
    bounds_before_transform: list[float] | None
    bounds_after_transform: list[float] | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self, geojson_path: str | Path) -> dict[str, Any]:
        return {
            "geojson_path": str(geojson_path),
            "annotation_crs": self.source_crs_text,
            "annotation_crs_source": self.crs_source,
            "raster_crs": str(self.raster_crs) if self.raster_crs else None,
            "transformed_to_raster_crs": bool(self.transformed_to_raster_crs),
            "feature_count": int(self.feature_count),
            "valid_geometry_count": int(self.valid_geometry_count),
            "invalid_skipped_geometry_count": int(self.invalid_geometry_count),
            "annotation_bounds_before_transform": self.bounds_before_transform,
            "annotation_bounds_after_transform": self.bounds_after_transform,
            "warnings": list(self.warnings),
        }


def load_annotation_geometries(annotation: AnnotationInput, *, raster_crs: CRS | str | None) -> AnnotationGeometrySet:
    path = Path(annotation.geojson_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_geometries, feature_count, invalid_count, warnings = _extract_geometries(payload)
    raster_crs_obj = CRS.from_user_input(raster_crs) if raster_crs else None
    source_crs, source_text, crs_source, crs_warnings = _resolve_annotation_crs(
        payload,
        annotation,
        raw_geometries,
        raster_crs_obj,
    )
    warnings.extend(crs_warnings)
    transformed = False
    geometries = raw_geometries
    if source_crs and raster_crs_obj and source_crs != raster_crs_obj:
        transformer = Transformer.from_crs(source_crs, raster_crs_obj, always_xy=True)
        geometries = [shapely_transform(transformer.transform, geom) for geom in raw_geometries]
        transformed = True
    elif source_crs is None and raster_crs_obj is not None:
        warnings.append("annotation CRS is unknown; geometries are treated as raster CRS")
        source_text = str(raster_crs_obj)
        crs_source = "same_as_raster"
        source_crs = raster_crs_obj

    return AnnotationGeometrySet(
        geometries=[geom for geom in geometries if geom.is_valid and not geom.is_empty],
        source_crs=source_crs,
        source_crs_text=source_text,
        crs_source=crs_source,
        raster_crs=raster_crs_obj,
        transformed_to_raster_crs=transformed,
        feature_count=feature_count,
        valid_geometry_count=len([geom for geom in geometries if geom.is_valid and not geom.is_empty]),
        invalid_geometry_count=invalid_count + len([geom for geom in geometries if not geom.is_valid or geom.is_empty]),
        bounds_before_transform=_combined_bounds(raw_geometries),
        bounds_after_transform=_combined_bounds(geometries),
        warnings=warnings,
    )


def _extract_geometries(payload: dict[str, Any]) -> tuple[list[Any], int, int, list[str]]:
    if payload.get("type") == "FeatureCollection":
        features = payload.get("features") or []
    elif payload.get("type") == "Feature":
        features = [payload]
    else:
        features = [{"type": "Feature", "geometry": payload, "properties": {}}]
    geometries: list[Any] = []
    invalid_count = 0
    warnings: list[str] = []
    for index, feature in enumerate(features):
        geometry_payload = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry_payload:
            invalid_count += 1
            warnings.append(f"feature {index} has no geometry and was skipped")
            continue
        try:
            geom = shape(geometry_payload)
            if geom.is_empty:
                invalid_count += 1
                continue
            if not geom.is_valid:
                fixed = geom.buffer(0)
                if fixed.is_empty or not fixed.is_valid:
                    invalid_count += 1
                    warnings.append(f"feature {index} is invalid and could not be repaired")
                    continue
                geom = fixed
                warnings.append(f"feature {index} was repaired with buffer(0)")
            geometries.append(geom)
        except Exception as exc:  # noqa: BLE001
            invalid_count += 1
            warnings.append(f"feature {index} could not be parsed: {type(exc).__name__}: {exc}")
    return geometries, len(features), invalid_count, warnings


def _resolve_annotation_crs(
    payload: dict[str, Any],
    annotation: AnnotationInput,
    geometries: list[Any],
    raster_crs: CRS | None,
) -> tuple[CRS | None, str | None, str, list[str]]:
    warnings: list[str] = []
    requested = annotation.annotation_crs
    if requested and str(requested).lower() not in {"auto", "none", "null"}:
        crs = CRS.from_user_input(requested)
        return crs, str(crs), "explicit", warnings
    geojson_crs = _geojson_crs(payload)
    if geojson_crs:
        crs = CRS.from_user_input(geojson_crs)
        return crs, str(crs), "geojson", warnings
    if requested and str(requested).lower() in {"none", "null"}:
        if not annotation.allow_inferred_annotation_crs:
            raise ValueError("annotation CRS is not set and CRS inference is disabled")
    if annotation.allow_inferred_annotation_crs:
        inferred, source = _infer_crs_from_bounds(geometries, raster_crs)
        if inferred:
            warnings.append(f"annotation CRS was inferred as {inferred}; pass --annotation-crs to avoid inference")
            return inferred, str(inferred), source, warnings
    raise ValueError("annotation CRS is missing; pass --annotation-crs or enable --allow-inferred-annotation-crs")


def _geojson_crs(payload: dict[str, Any]) -> str | None:
    crs_payload = payload.get("crs")
    if not isinstance(crs_payload, dict):
        return None
    properties = crs_payload.get("properties") or {}
    name = properties.get("name") or properties.get("href")
    return str(name) if name else None


def _infer_crs_from_bounds(geometries: list[Any], raster_crs: CRS | None) -> tuple[CRS | None, str]:
    bounds = _combined_bounds(geometries)
    if not bounds:
        return raster_crs, "same_as_raster"
    minx, miny, maxx, maxy = bounds
    if -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0:
        return CRS.from_epsg(4326), "inferred"
    if max(abs(minx), abs(maxx), abs(miny), abs(maxy)) > 1000:
        return CRS.from_epsg(3857), "inferred"
    return raster_crs, "same_as_raster"


def _combined_bounds(geometries: list[Any]) -> list[float] | None:
    valid = [geom for geom in geometries if geom is not None and not geom.is_empty]
    if not valid:
        return None
    minx = min(float(geom.bounds[0]) for geom in valid)
    miny = min(float(geom.bounds[1]) for geom in valid)
    maxx = max(float(geom.bounds[2]) for geom in valid)
    maxy = max(float(geom.bounds[3]) for geom in valid)
    return [minx, miny, maxx, maxy]
