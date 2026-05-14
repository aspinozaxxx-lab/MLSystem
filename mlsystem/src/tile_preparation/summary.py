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
    }
