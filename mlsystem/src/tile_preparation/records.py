from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


TileKind = Literal["positive", "partial_positive", "hard_negative", "negative"]


@dataclass(frozen=True)
class TileWindow:
    scene_id: str
    x: int
    y: int
    width: int
    height: int
    tile_size: int
    stride: int
    index: int = 0


@dataclass
class TileSampleRecord:
    scene_id: str
    image_path: str | None
    x: int
    y: int
    width: int
    height: int
    tile_size: int
    stride: int
    kind: TileKind
    positive_pixels: int
    source: str
    base_record_id: str | None = None
    repeat_index: int | None = None
    geometries_intersecting: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def record_id(self) -> str:
        existing = self.metadata.get("record_id")
        if existing:
            return str(existing)
        return f"{self.scene_id}:x{self.x}:y{self.y}:w{self.width}:h{self.height}:{self.kind}"

    def to_dict(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata.setdefault("record_id", self.record_id)
        return {
            "sample_id": metadata.get("sample_id") or self.record_id,
            "scene_id": self.scene_id,
            "scene": self.scene_id,
            "tile_id": f"x{self.x}_y{self.y}_w{self.width}_h{self.height}",
            "source_image_path": self.image_path,
            "window": {"x": int(self.x), "y": int(self.y), "width": int(self.width), "height": int(self.height)},
            "tile_size": int(self.tile_size),
            "stride": int(self.stride),
            "kind": self.kind,
            "positive_tile": self.kind in {"positive", "partial_positive"},
            "positive_pixels": int(self.positive_pixels),
            "gt_positive_pixels": int(self.positive_pixels),
            "geometries_intersecting": int(self.geometries_intersecting),
            "source": self.source,
            "base_record_id": self.base_record_id,
            "repeat_index": self.repeat_index,
            "metadata": metadata,
        }


@dataclass
class ReadyTileSample:
    image: np.ndarray
    mask: np.ndarray
    record: TileSampleRecord
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TileRecordBuildResult:
    records: list[TileSampleRecord]
    base_records: list[TileSampleRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scene_reports: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
