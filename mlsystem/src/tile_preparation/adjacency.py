from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from shapely.ops import transform as shapely_transform

from .footprint import SceneFootprint, footprint_affine


Relation = Literal["touch", "overlap", "near", "disjoint"]
Side = Literal["left", "right", "top", "bottom", "overlap", "unknown"]


@dataclass(frozen=True)
class SceneAdjacency:
    anchor_scene_id: str
    neighbor_scene_id: str
    relation: Relation
    side: Side
    intersection_area: float
    distance: float
    anchor_boundary_length: float
    neighbor_coverage_estimated: float
    anchor_crs: str | None
    neighbor_crs: str | None


@dataclass
class SceneAdjacencyIndex:
    by_anchor: dict[str, list[SceneAdjacency]] = field(default_factory=dict)

    def neighbors_for(self, scene_id: str) -> list[SceneAdjacency]:
        return list(self.by_anchor.get(str(scene_id), []))


def build_scene_adjacency_index(
    footprints: dict[str, SceneFootprint],
    *,
    tolerance_px: float = 2.0,
    near_distance_px: float = 0.0,
) -> SceneAdjacencyIndex:
    by_anchor: dict[str, list[SceneAdjacency]] = {}
    ordered = sorted((str(scene_id), footprint) for scene_id, footprint in footprints.items())
    for anchor_id, anchor in ordered:
        anchor_polygon = anchor.polygon_raster_crs
        if anchor_polygon is None or getattr(anchor_polygon, "is_empty", True):
            continue
        tolerance = _pixel_distance_in_raster_units(anchor, tolerance_px)
        near_distance = _pixel_distance_in_raster_units(anchor, near_distance_px)
        max_distance = max(float(tolerance), float(near_distance))
        entries: list[SceneAdjacency] = []
        for neighbor_id, neighbor in ordered:
            if neighbor_id == anchor_id:
                continue
            neighbor_polygon = footprint_polygon_in_anchor_crs(anchor, neighbor)
            if neighbor_polygon is None or getattr(neighbor_polygon, "is_empty", True):
                continue
            try:
                intersection_area = float(anchor_polygon.intersection(neighbor_polygon).area)
                distance = float(anchor_polygon.distance(neighbor_polygon))
            except Exception:  # noqa: BLE001
                continue
            relation: Relation
            if intersection_area > 0.0:
                relation = "overlap"
            elif distance <= tolerance:
                relation = "touch"
            elif distance <= max_distance:
                relation = "near"
            else:
                continue
            anchor_area = max(1e-9, float(anchor_polygon.area))
            entries.append(
                SceneAdjacency(
                    anchor_scene_id=anchor_id,
                    neighbor_scene_id=neighbor_id,
                    relation=relation,
                    side=_side(anchor_polygon, neighbor_polygon, relation),
                    intersection_area=intersection_area,
                    distance=distance,
                    anchor_boundary_length=float(anchor_polygon.boundary.length),
                    neighbor_coverage_estimated=max(0.0, min(1.0, intersection_area / anchor_area)),
                    anchor_crs=anchor.raster_crs,
                    neighbor_crs=neighbor.raster_crs,
                )
            )
        entries.sort(key=lambda item: ({"overlap": 0, "touch": 1, "near": 2, "disjoint": 3}[item.relation], item.distance, item.neighbor_scene_id))
        by_anchor[anchor_id] = entries
    return SceneAdjacencyIndex(by_anchor=by_anchor)


def footprint_polygon_in_anchor_crs(anchor: SceneFootprint, neighbor: SceneFootprint):
    polygon = neighbor.polygon_raster_crs
    if polygon is None:
        return None
    anchor_crs = str(anchor.raster_crs or "")
    neighbor_crs = str(neighbor.raster_crs or "")
    if not anchor_crs or not neighbor_crs or anchor_crs == neighbor_crs:
        return polygon
    try:
        from pyproj import Transformer

        transformer = Transformer.from_crs(neighbor_crs, anchor_crs, always_xy=True)
        return shapely_transform(transformer.transform, polygon)
    except Exception:  # noqa: BLE001
        return None


def _pixel_distance_in_raster_units(footprint: SceneFootprint, pixels: float) -> float:
    pixels = max(0.0, float(pixels))
    affine = footprint_affine(footprint)
    if affine is None:
        return pixels
    x_size = math.hypot(float(affine.a), float(affine.d))
    y_size = math.hypot(float(affine.b), float(affine.e))
    scale = max(x_size, y_size, 1e-9)
    return pixels * scale


def _side(anchor_polygon, neighbor_polygon, relation: Relation) -> Side:
    if relation == "overlap":
        return "overlap"
    ax0, ay0, ax1, ay1 = anchor_polygon.bounds
    nx0, ny0, nx1, ny1 = neighbor_polygon.bounds
    acx = (ax0 + ax1) * 0.5
    acy = (ay0 + ay1) * 0.5
    ncx = (nx0 + nx1) * 0.5
    ncy = (ny0 + ny1) * 0.5
    dx = ncx - acx
    dy = ncy - acy
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "top" if dy >= 0 else "bottom"
