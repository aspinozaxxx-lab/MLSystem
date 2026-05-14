from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal


@dataclass(frozen=True)
class SceneInput:
    image_path: Path | str
    scene_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def resolved_scene_id(self) -> str:
        return self.scene_id or PurePosixPath(str(self.image_path).replace("\\", "/")).stem


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
    batch_positive_fraction: float | None = None
    batch_hard_negative_fraction: float | None = None
    batch_negative_fraction: float | None = None
    all_touched: bool = False
    output_format: str = "chw_float32"
    normalize: bool = True
    input_bands: list[int] | None = None
    valid_pixel_mode: str = "auto"
    min_valid_pixel_share: float = 0.0
    clip_mask_to_valid_data: bool = True
    exclude_empty_valid_tiles: bool = False
    drop_fully_invalid_tiles: bool = True
    mosaic_enabled: bool = False
    mosaic_fill_nodata: bool = True
    mosaic_require_same_crs: bool = False
    mosaic_resampling: str = "bilinear"
    augmentations: dict[str, Any] = field(default_factory=dict)
    apply_random_augmentations: bool = False
    cutout_mask_mode: Literal["erase", "preserve", "ignore"] = "erase"
    max_records: int | None = None
    max_records_per_scene: int | None = None
    shuffle: bool = False
    seed: int = 42
    augmentation_mode: str = "all"
    augmentation_seed: int = 42
    augmentation_level: int = 0

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
        self.valid_pixel_mode = str(self.valid_pixel_mode or "auto")
        self.min_valid_pixel_share = max(0.0, min(1.0, float(self.min_valid_pixel_share)))
        self.mosaic_resampling = str(self.mosaic_resampling or "bilinear")
        self.cutout_mask_mode = str(self.cutout_mask_mode or "erase").lower()
        if self.cutout_mask_mode not in {"erase", "preserve", "ignore"}:
            raise ValueError("cutout_mask_mode must be one of: erase, preserve, ignore")
        self.augmentation_level = max(0, min(3, int(self.augmentation_level)))

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


def train_sampling_enabled(preprocess: dict[str, Any] | None) -> bool:
    raw = ((preprocess or {}).get("train_sampling") or {}) if isinstance(preprocess, dict) else {}
    return bool(raw.get("enabled", False))


def resolve_tile_preparation_config(
    preprocess: dict[str, Any] | None,
    train: dict[str, Any] | None,
    *,
    tile_size: int,
    stride: int,
    input_bands: list[int] | None,
    seed: int,
    train_mode: bool,
    max_records: int | None = None,
    max_records_per_scene: int | None = None,
) -> TilePreparationConfig:
    preprocess = preprocess or {}
    train = train or {}
    raw = dict(preprocess.get("train_sampling") or {})
    enabled = bool(raw.get("enabled", False))
    random_jitter = raw.get("random_jitter") or {}
    max_empty_tile_share = raw.get("max_empty_tile_share")
    if max_empty_tile_share is None and "max_empty_tile_share" in preprocess:
        max_empty_tile_share = preprocess.get("max_empty_tile_share")
        if not train_mode:
            return TilePreparationConfig(
            tile_size=tile_size,
            stride=stride,
            min_positive_pixels=max(1, int(raw.get("min_positive_pixels", 1) or 1)),
            include_partial_positive=bool(raw.get("include_partial_positive", True)),
                partial_positive_fraction=max(0.0, float(raw.get("partial_positive_fraction", 0.0) or 0.0)),
                valid_pixel_mode=str(raw.get("valid_pixel_mode", preprocess.get("valid_pixel_mode", "auto")) or "auto"),
                min_valid_pixel_share=max(0.0, min(1.0, float(raw.get("min_valid_pixel_share", preprocess.get("min_valid_pixel_share", 0.0)) or 0.0))),
                clip_mask_to_valid_data=bool(raw.get("clip_mask_to_valid_data", preprocess.get("clip_mask_to_valid_data", True))),
                exclude_empty_valid_tiles=bool(raw.get("exclude_empty_valid_tiles", preprocess.get("exclude_empty_valid_tiles", False))),
                drop_fully_invalid_tiles=bool(raw.get("drop_fully_invalid_tiles", preprocess.get("drop_fully_invalid_tiles", True))),
                mosaic_enabled=bool(raw.get("mosaic_enabled", preprocess.get("mosaic_enabled", False))),
                mosaic_fill_nodata=bool(raw.get("mosaic_fill_nodata", preprocess.get("mosaic_fill_nodata", True))),
                mosaic_require_same_crs=bool(raw.get("mosaic_require_same_crs", preprocess.get("mosaic_require_same_crs", False))),
                mosaic_resampling=str(raw.get("mosaic_resampling", preprocess.get("mosaic_resampling", "bilinear")) or "bilinear"),
                cutout_mask_mode=str(raw.get("cutout_mask_mode", preprocess.get("cutout_mask_mode", "erase")) or "erase"),
                input_bands=input_bands,
            apply_random_augmentations=False,
            max_records=max_records,
            max_records_per_scene=max_records_per_scene,
            seed=seed,
        )
    augmentations = dict(train.get("augmentations") or {})
    return TilePreparationConfig(
        tile_size=tile_size,
        stride=stride,
        positive_stride_factor=max(0.000001, float(raw.get("positive_stride_factor", 1.0 if enabled else 1.0) or 1.0)),
        hard_negative_stride_factor=max(0.000001, float(raw.get("hard_negative_stride_factor", 1.0 if enabled else 1.0) or 1.0)),
        negative_stride_factor=max(0.000001, float(raw.get("negative_stride_factor", 1.0) or 1.0)),
        min_positive_pixels=max(1, int(raw.get("min_positive_pixels", 1) or 1)),
        include_partial_positive=bool(raw.get("include_partial_positive", True)),
        partial_positive_fraction=max(0.0, float(raw.get("partial_positive_fraction", 0.0) or 0.0)),
        valid_pixel_mode=str(raw.get("valid_pixel_mode", preprocess.get("valid_pixel_mode", "auto")) or "auto"),
        min_valid_pixel_share=max(0.0, min(1.0, float(raw.get("min_valid_pixel_share", preprocess.get("min_valid_pixel_share", 0.0)) or 0.0))),
        clip_mask_to_valid_data=bool(raw.get("clip_mask_to_valid_data", preprocess.get("clip_mask_to_valid_data", True))),
        exclude_empty_valid_tiles=bool(raw.get("exclude_empty_valid_tiles", preprocess.get("exclude_empty_valid_tiles", False))),
        drop_fully_invalid_tiles=bool(raw.get("drop_fully_invalid_tiles", preprocess.get("drop_fully_invalid_tiles", True))),
        mosaic_enabled=bool(raw.get("mosaic_enabled", preprocess.get("mosaic_enabled", False))),
        mosaic_fill_nodata=bool(raw.get("mosaic_fill_nodata", preprocess.get("mosaic_fill_nodata", True))),
        mosaic_require_same_crs=bool(raw.get("mosaic_require_same_crs", preprocess.get("mosaic_require_same_crs", False))),
        mosaic_resampling=str(raw.get("mosaic_resampling", preprocess.get("mosaic_resampling", "bilinear")) or "bilinear"),
        cutout_mask_mode=str(raw.get("cutout_mask_mode", preprocess.get("cutout_mask_mode", "erase")) or "erase"),
        max_empty_tile_share=None if max_empty_tile_share is None else max(0.0, min(1.0, float(max_empty_tile_share))),
        hard_negative_context_px=None if raw.get("hard_negative_context_px") is None else max(0, int(raw.get("hard_negative_context_px") or 0)),
        virtual_epoch_multiplier=max(1, int(raw.get("virtual_epoch_multiplier", 1) or 1)) if enabled else 1,
        positive_repeat_factor=max(1, int(raw.get("positive_repeat_factor", 1) or 1)) if enabled else 1,
        hard_negative_repeat_factor=max(1, int(raw.get("hard_negative_repeat_factor", 1) or 1)) if enabled else 1,
        negative_repeat_factor=max(1, int(raw.get("negative_repeat_factor", 1) or 1)) if enabled else 1,
        batch_positive_fraction=_optional_fraction(raw.get("batch_positive_fraction")),
        batch_hard_negative_fraction=_optional_fraction(raw.get("batch_hard_negative_fraction")),
        batch_negative_fraction=_optional_fraction(raw.get("batch_negative_fraction")),
        input_bands=input_bands,
        augmentations=augmentations,
        apply_random_augmentations=bool(augmentations),
        max_records=max_records,
        max_records_per_scene=max_records_per_scene,
        shuffle=bool(train.get("shuffle_tiles", False)),
        seed=seed,
    )


def _optional_fraction(value: Any) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))
