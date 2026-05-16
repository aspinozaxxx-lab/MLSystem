from __future__ import annotations

from collections import Counter
from typing import Any

from .records import TileSampleRecord


def summarize_tile_records(records: list[TileSampleRecord], *, prefix: str = "") -> dict[str, Any]:
    counts = Counter(record.kind for record in records)
    total = len(records)
    empty = int(counts.get("negative", 0) + counts.get("hard_negative", 0))
    stem = f"{prefix}_" if prefix else ""
    raw_pixels = int(sum(record.mask_pixels_before_valid_clip for record in records))
    clipped_pixels = int(sum(record.mask_pixels_after_valid_clip for record in records))
    valid_share_mean = float(sum(record.valid_pixel_share for record in records) / total) if total else 0.0
    mosaic_filled = int(sum(record.mosaic_filled_pixel_count for record in records))
    mosaic_unfilled = int(sum(record.mosaic_unfilled_pixel_count for record in records))
    mosaic_counters = _mosaic_counters(records)
    return {
        f"{stem}tile_count": int(total),
        f"{stem}positive_tiles": int(counts.get("positive", 0)),
        f"{stem}partial_positive_tiles": int(counts.get("partial_positive", 0)),
        f"{stem}hard_negative_tiles": int(counts.get("hard_negative", 0)),
        f"{stem}negative_tiles": int(counts.get("negative", 0)),
        f"{stem}empty_tiles": int(empty),
        f"{stem}empty_tile_share": float(empty) / float(total) if total else 0.0,
        f"{stem}valid_pixel_share_mean": valid_share_mean,
        f"{stem}raw_mask_positive_pixels": raw_pixels,
        f"{stem}clipped_mask_positive_pixels": clipped_pixels,
        f"{stem}mask_pixels_removed_by_valid_clip": max(0, raw_pixels - clipped_pixels),
        f"{stem}mosaic_filled_pixels": mosaic_filled,
        f"{stem}mosaic_unfilled_pixels": mosaic_unfilled,
        **{f"{stem}{key}": value for key, value in mosaic_counters.items() if key != "mosaic_filled_pixels"},
    }


def _mosaic_counters(records: list[TileSampleRecord]) -> dict[str, int]:
    counters = {
        "mosaic_attempted": 0,
        "mosaic_skipped_fully_inside": 0,
        "mosaic_skipped_no_gap": 0,
        "mosaic_skipped_no_candidates": 0,
        "mosaic_candidate_neighbors_total": 0,
        "mosaic_intersecting_neighbors_total": 0,
        "mosaic_warped_vrt_calls": 0,
        "mosaic_used_neighbors": 0,
        "mosaic_zero_fill_attempts": 0,
        "mosaic_boundary_records": 0,
        "mosaic_overlap_records": 0,
    }
    for record in records:
        metadata = record.metadata or {}
        reason = str(metadata.get("mosaic_reason") or "")
        candidates = metadata.get("mosaic_candidate_scene_ids") or []
        counters["mosaic_attempted"] += int(bool(metadata.get("mosaic_needed", False)))
        counters["mosaic_skipped_fully_inside"] += int(reason == "fully_inside_footprint")
        counters["mosaic_skipped_no_gap"] += int(reason == "no_gap")
        counters["mosaic_skipped_no_candidates"] += int(reason in {"no_neighbor_for_gap", "no_raster_transform", "missing_anchor_footprint"})
        counters["mosaic_candidate_neighbors_total"] += int(len(candidates))
        counters["mosaic_intersecting_neighbors_total"] += int(len(candidates))
        counters["mosaic_warped_vrt_calls"] += int(metadata.get("mosaic_warped_vrt_calls") or 0)
        counters["mosaic_used_neighbors"] += int(metadata.get("mosaic_used_neighbors") or 0)
        counters["mosaic_zero_fill_attempts"] += int(metadata.get("mosaic_zero_fill_attempts") or 0)
        counters["mosaic_boundary_records"] += int(bool(metadata.get("footprint_boundary", False)))
        counters["mosaic_overlap_records"] += int(bool(metadata.get("mosaic_overlap_record", False)))
    return counters
