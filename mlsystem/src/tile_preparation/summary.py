from __future__ import annotations

from collections import Counter
from typing import Any

from .records import TileSampleRecord


def summarize_tile_records(records: list[TileSampleRecord], *, prefix: str = "") -> dict[str, Any]:
    counts = Counter(record.kind for record in records)
    total = len(records)
    empty = int(counts.get("negative", 0) + counts.get("hard_negative", 0))
    stem = f"{prefix}_" if prefix else ""
    return {
        f"{stem}tile_count": int(total),
        f"{stem}positive_tiles": int(counts.get("positive", 0)),
        f"{stem}partial_positive_tiles": int(counts.get("partial_positive", 0)),
        f"{stem}hard_negative_tiles": int(counts.get("hard_negative", 0)),
        f"{stem}negative_tiles": int(counts.get("negative", 0)),
        f"{stem}empty_tiles": int(empty),
        f"{stem}empty_tile_share": float(empty) / float(total) if total else 0.0,
    }
