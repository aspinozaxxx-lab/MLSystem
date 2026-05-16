from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.ops import unary_union

from .adjacency import SceneAdjacencyIndex, footprint_polygon_in_anchor_crs
from .footprint import SceneFootprint, pixel_geometry_to_raster_crs, window_polygon_pixel, window_polygon_raster_crs


@dataclass(frozen=True)
class TileMosaicPlan:
    needed: bool
    reason: str
    side: str
    candidate_scene_ids: list[str] = field(default_factory=list)
    estimated_gap_share: float = 0.0
    estimated_neighbor_cover_share: float = 0.0
    overlap_record: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mosaic_needed": bool(self.needed),
            "mosaic_reason": self.reason,
            "mosaic_side": self.side,
            "mosaic_candidate_scene_ids": list(self.candidate_scene_ids),
            "mosaic_estimated_gap_share": float(self.estimated_gap_share),
            "mosaic_estimated_neighbor_cover_share": float(self.estimated_neighbor_cover_share),
            "mosaic_overlap_record": bool(self.overlap_record),
        }


def build_tile_mosaic_plan(
    record: Any,
    anchor_footprint: SceneFootprint,
    footprints: dict[str, SceneFootprint],
    adjacency_index: SceneAdjacencyIndex,
) -> TileMosaicPlan:
    if bool(getattr(record, "metadata", {}).get("footprint_fully_inside", False)):
        return TileMosaicPlan(False, "fully_inside_footprint", "unknown")

    window_pixel = window_polygon_pixel(record)
    try:
        gap_pixel = window_pixel.difference(anchor_footprint.polygon_pixel)
    except Exception:  # noqa: BLE001
        gap_pixel = None
    window_area = max(1.0, float(int(record.width) * int(record.height)))
    gap_share = float(getattr(gap_pixel, "area", 0.0) or 0.0) / window_area if gap_pixel is not None else 0.0
    if gap_pixel is None or getattr(gap_pixel, "is_empty", True) or gap_share <= 1e-6:
        return TileMosaicPlan(False, "no_gap", "unknown", estimated_gap_share=max(0.0, gap_share), overlap_record=_overlap_record(record, anchor_footprint, footprints, adjacency_index))

    gap_raster = pixel_geometry_to_raster_crs(anchor_footprint, gap_pixel)
    tile_raster = window_polygon_raster_crs(anchor_footprint, record)
    if gap_raster is None or getattr(gap_raster, "is_empty", True):
        return TileMosaicPlan(False, "no_raster_transform", "unknown", estimated_gap_share=max(0.0, gap_share), overlap_record=False)

    candidates: list[tuple[str, str, Any]] = []
    overlap_record = False
    for adjacency in adjacency_index.neighbors_for(str(record.scene_id)):
        neighbor = footprints.get(adjacency.neighbor_scene_id)
        if neighbor is None:
            continue
        neighbor_polygon = footprint_polygon_in_anchor_crs(anchor_footprint, neighbor)
        if neighbor_polygon is None or getattr(neighbor_polygon, "is_empty", True):
            continue
        try:
            if tile_raster is not None and adjacency.relation == "overlap" and neighbor_polygon.intersects(tile_raster):
                overlap_record = True
            if neighbor_polygon.intersects(gap_raster) and float(neighbor_polygon.intersection(gap_raster).area) > 0.0:
                candidates.append((adjacency.neighbor_scene_id, adjacency.side, neighbor_polygon.intersection(gap_raster)))
        except Exception:  # noqa: BLE001
            continue

    if not candidates:
        return TileMosaicPlan(False, "no_neighbor_for_gap", "unknown", estimated_gap_share=max(0.0, gap_share), overlap_record=overlap_record)

    cover = _coverage_share([item[2] for item in candidates], float(gap_raster.area))
    if cover <= 0.0:
        return TileMosaicPlan(False, "no_neighbor_for_gap", "unknown", estimated_gap_share=max(0.0, gap_share), overlap_record=overlap_record)
    side = candidates[0][1] if candidates else "unknown"
    return TileMosaicPlan(
        True,
        "neighbor_covers_gap",
        side,
        candidate_scene_ids=[item[0] for item in candidates],
        estimated_gap_share=max(0.0, min(1.0, gap_share)),
        estimated_neighbor_cover_share=cover,
        overlap_record=overlap_record,
    )


def mosaic_plan_counters(records: list[Any]) -> dict[str, int]:
    counters = {
        "mosaic_attempted": 0,
        "mosaic_skipped_fully_inside": 0,
        "mosaic_skipped_no_gap": 0,
        "mosaic_skipped_no_candidates": 0,
        "mosaic_candidate_neighbors_total": 0,
        "mosaic_intersecting_neighbors_total": 0,
        "mosaic_warped_vrt_calls": 0,
        "mosaic_used_neighbors": 0,
        "mosaic_filled_pixels": 0,
        "mosaic_zero_fill_attempts": 0,
        "mosaic_boundary_records": 0,
        "mosaic_overlap_records": 0,
    }
    for record in records:
        metadata = getattr(record, "metadata", {}) or {}
        reason = str(metadata.get("mosaic_reason") or "")
        needed = bool(metadata.get("mosaic_needed", False))
        candidates = metadata.get("mosaic_candidate_scene_ids") or []
        counters["mosaic_attempted"] += int(needed)
        counters["mosaic_skipped_fully_inside"] += int(reason == "fully_inside_footprint")
        counters["mosaic_skipped_no_gap"] += int(reason == "no_gap")
        counters["mosaic_skipped_no_candidates"] += int(reason in {"no_neighbor_for_gap", "no_raster_transform"})
        counters["mosaic_candidate_neighbors_total"] += int(len(candidates))
        counters["mosaic_intersecting_neighbors_total"] += int(len(candidates))
        counters["mosaic_boundary_records"] += int(bool(metadata.get("footprint_boundary", False)))
        counters["mosaic_overlap_records"] += int(bool(metadata.get("mosaic_overlap_record", False)))
        counters["mosaic_warped_vrt_calls"] += int(metadata.get("mosaic_warped_vrt_calls") or 0)
        counters["mosaic_used_neighbors"] += int(metadata.get("mosaic_used_neighbors") or 0)
        counters["mosaic_filled_pixels"] += int(metadata.get("mosaic_filled_pixel_count") or 0)
        counters["mosaic_zero_fill_attempts"] += int(metadata.get("mosaic_zero_fill_attempts") or 0)
    return counters


def apply_mosaic_plans(
    records: list[Any],
    footprints: dict[str, SceneFootprint],
    adjacency_index: SceneAdjacencyIndex,
) -> dict[str, int]:
    for record in records:
        anchor = footprints.get(str(record.scene_id))
        if anchor is None:
            plan = TileMosaicPlan(False, "missing_anchor_footprint", "unknown")
        else:
            plan = build_tile_mosaic_plan(record, anchor, footprints, adjacency_index)
        record.metadata.update(plan.to_metadata())
    return mosaic_plan_counters(records)


def _coverage_share(geometries: list[Any], denominator_area: float) -> float:
    if denominator_area <= 0.0 or not geometries:
        return 0.0
    try:
        area = float(unary_union(geometries).area)
    except Exception:  # noqa: BLE001
        area = sum(float(getattr(item, "area", 0.0) or 0.0) for item in geometries)
    return max(0.0, min(1.0, area / denominator_area))


def _overlap_record(record: Any, anchor_footprint: SceneFootprint, footprints: dict[str, SceneFootprint], adjacency_index: SceneAdjacencyIndex) -> bool:
    tile_raster = window_polygon_raster_crs(anchor_footprint, record)
    if tile_raster is None:
        return False
    for adjacency in adjacency_index.neighbors_for(str(record.scene_id)):
        if adjacency.relation != "overlap":
            continue
        neighbor = footprints.get(adjacency.neighbor_scene_id)
        if neighbor is None:
            continue
        polygon = footprint_polygon_in_anchor_crs(anchor_footprint, neighbor)
        if polygon is None:
            continue
        try:
            if polygon.intersects(tile_raster):
                return True
        except Exception:  # noqa: BLE001
            continue
    return False
