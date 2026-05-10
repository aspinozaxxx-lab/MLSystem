from __future__ import annotations

from typing import Any

from shapely.geometry import mapping, shape


def simplify_features(features: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    if float(tolerance) <= 0:
        return features
    out: list[dict[str, Any]] = []
    for feature in features:
        geom = shape(feature["geometry"]).simplify(float(tolerance), preserve_topology=True)
        if geom.is_empty:
            continue
        properties = dict(feature.get("properties") or {})
        properties["area_m2"] = float(geom.area)
        out.append({"type": "Feature", "properties": properties, "geometry": mapping(geom)})
    return out
