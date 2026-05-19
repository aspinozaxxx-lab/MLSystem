from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneInputContract:
    path: Path | str
    scene_id: str | None = None


@dataclass(frozen=True)
class AnnotationInputContract:
    geojson_path: Path | str
    annotation_crs: str | None = "auto"
    allow_inferred_annotation_crs: bool = True


@dataclass
class TileDatasetRequest:
    train_scenes: list[SceneInputContract]
    val_scenes: list[SceneInputContract]
    annotation_path: Path | str
    tile_size: int
    stride: int
    augmentation_level: int = 2
    mosaic_enabled: bool | None = None
    normalization_mode: str = "uint8_255"
    max_empty_tile_share: float | None = None
    max_tiles_per_scene: int | None = None
    max_train_tiles: int | None = None
    max_val_tiles: int | None = None
    augmentations: dict[str, Any] | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
