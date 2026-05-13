from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneInput:
    image_path: Path
    scene_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_scene_id(self) -> str:
        return self.scene_id or Path(self.image_path).stem


@dataclass(frozen=True)
class AnnotationInput:
    geojson_path: Path
    annotation_crs: str | None = "auto"
    allow_inferred_annotation_crs: bool = False


@dataclass
class TilePreparationConfig:
    tile_size: int = 768
    stride: int = 512
    positive_stride_factor: float = 1.0
    hard_negative_stride_factor: float = 1.0
    negative_stride_factor: float = 1.0
    min_positive_pixels: int = 1
    include_partial_positive: bool = True
    partial_positive_fraction: float = 0.0
    max_empty_tile_share: float | None = None
    hard_negative_context_px: int | None = None
    virtual_epoch_multiplier: int = 1
    positive_repeat_factor: int = 1
    hard_negative_repeat_factor: int = 1
    negative_repeat_factor: int = 1
    all_touched: bool = False
    output_format: str = "chw_float32"
    normalize: bool = True
    input_bands: list[int] | None = None
    augmentations: dict[str, Any] = field(default_factory=dict)
    apply_random_augmentations: bool = False
    max_records: int | None = None
    max_records_per_scene: int | None = None
    shuffle: bool = False
    seed: int = 42
    augmentation_mode: str = "all"
    augmentation_seed: int = 42

    def __post_init__(self) -> None:
        self.tile_size = max(1, int(self.tile_size))
        self.stride = max(1, int(self.stride))
        self.positive_stride_factor = max(0.000001, float(self.positive_stride_factor))
        self.hard_negative_stride_factor = max(0.000001, float(self.hard_negative_stride_factor))
        self.negative_stride_factor = max(0.000001, float(self.negative_stride_factor))
        self.min_positive_pixels = max(1, int(self.min_positive_pixels))
        self.partial_positive_fraction = max(0.0, float(self.partial_positive_fraction))
        if self.max_empty_tile_share is not None:
            self.max_empty_tile_share = max(0.0, min(1.0, float(self.max_empty_tile_share)))
        if self.hard_negative_context_px is not None:
            self.hard_negative_context_px = max(0, int(self.hard_negative_context_px))
        self.virtual_epoch_multiplier = max(1, int(self.virtual_epoch_multiplier))
        self.positive_repeat_factor = max(1, int(self.positive_repeat_factor))
        self.hard_negative_repeat_factor = max(1, int(self.hard_negative_repeat_factor))
        self.negative_repeat_factor = max(1, int(self.negative_repeat_factor))
        if self.max_records is not None:
            self.max_records = max(0, int(self.max_records))
        if self.max_records_per_scene is not None:
            self.max_records_per_scene = max(0, int(self.max_records_per_scene))

    @property
    def positive_stride(self) -> int:
        return effective_stride(self.stride, self.positive_stride_factor)

    @property
    def hard_negative_stride(self) -> int:
        return effective_stride(self.stride, self.hard_negative_stride_factor)

    @property
    def negative_stride(self) -> int:
        return effective_stride(self.stride, self.negative_stride_factor)


def effective_stride(base_stride: int, factor: float) -> int:
    return max(1, int(max(1, int(base_stride)) * max(0.000001, float(factor))))
