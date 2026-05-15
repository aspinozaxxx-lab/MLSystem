from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from .config import AnnotationInput, SceneInput, TilePreparationConfig
from .dataloader import make_tile_dataloader
from .dataset import TrainingTileDataset, iter_dataset_batches
from .report import generate_annotated_tile_report
from .summary import summarize_tile_records


DEFAULT_SEED = 42
DEFAULT_MOSAIC_MODE = "auto"
DEFAULT_ANNOTATION_CRS = "auto"
DEFAULT_ALLOW_INFERRED_ANNOTATION_CRS = True
DEFAULT_INPUT_BANDS = None


@dataclass
class TilePreparationBundle:
    config: TilePreparationConfig
    train_dataset: TrainingTileDataset
    val_dataset: TrainingTileDataset
    train_summary: dict
    val_summary: dict
    warnings: list[str]


def config_from_augmentation_level(
    level: int,
    *,
    tile_size: int,
    stride: int,
) -> TilePreparationConfig:
    level = max(0, min(3, int(level)))
    tile_size = max(1, int(tile_size))
    stride = max(1, int(stride))

    presets = {
        0: {
            "augmentations": {},
            "positive_repeat_factor": 1,
            "hard_negative_repeat_factor": 1,
            "negative_repeat_factor": 1,
            "positive_stride_factor": 1.0,
            "hard_negative_stride_factor": 1.0,
            "negative_stride_factor": 1.0,
            "max_empty_tile_share": None,
        },
        1: {
            "augmentations": {"flips": True, "rot90": True},
            "positive_repeat_factor": 2,
            "hard_negative_repeat_factor": 1,
            "negative_repeat_factor": 1,
            "positive_stride_factor": 1.0,
            "hard_negative_stride_factor": 1.0,
            "negative_stride_factor": 1.0,
            "max_empty_tile_share": 0.5,
        },
        2: {
            "augmentations": {"flips": True, "rot90": True, "brightness_contrast": True, "gamma": True, "noise": True, "blur": True},
            "positive_repeat_factor": 4,
            "hard_negative_repeat_factor": 2,
            "negative_repeat_factor": 1,
            "positive_stride_factor": 0.5,
            "hard_negative_stride_factor": 0.5,
            "negative_stride_factor": 1.0,
            "max_empty_tile_share": 0.35,
        },
        3: {
            "augmentations": {
                "flips": True,
                "rot90": True,
                "brightness_contrast": True,
                "color_jitter": True,
                "gamma": True,
                "noise": True,
                "blur": True,
                "cutout": True,
                "coarse_dropout": True,
            },
            "positive_repeat_factor": 6,
            "hard_negative_repeat_factor": 3,
            "negative_repeat_factor": 1,
            "positive_stride_factor": 0.25,
            "hard_negative_stride_factor": 0.5,
            "negative_stride_factor": 1.0,
            "max_empty_tile_share": 0.25,
        },
    }
    preset = dict(presets[level])
    return TilePreparationConfig(
        tile_size=tile_size,
        stride=stride,
        input_bands=DEFAULT_INPUT_BANDS,
        seed=DEFAULT_SEED,
        augmentation_level=level,
        apply_random_augmentations=bool(preset["augmentations"]),
        cutout_mask_mode="erase",
        **preset,
    )


class TilePreparationFacade:
    @staticmethod
    def default_config(
        *,
        tile_size: int,
        stride: int,
        augmentation_level: int = 2,
    ) -> TilePreparationConfig:
        return config_from_augmentation_level(augmentation_level, tile_size=tile_size, stride=stride)

    @staticmethod
    def build_datasets(
        *,
        train_scenes: list[SceneInput],
        val_scenes: list[SceneInput],
        annotation_path: Path,
        tile_size: int,
        stride: int,
        augmentation_level: int = 2,
        mosaic_enabled: bool | None = None,
        normalization_mode: str = "uint8_255",
    ) -> TilePreparationBundle:
        annotation = AnnotationInput(
            geojson_path=Path(annotation_path),
            annotation_crs=DEFAULT_ANNOTATION_CRS,
            allow_inferred_annotation_crs=DEFAULT_ALLOW_INFERRED_ANNOTATION_CRS,
        )
        train_config = TilePreparationFacade.default_config(
            tile_size=tile_size,
            stride=stride,
            augmentation_level=augmentation_level,
        )
        normalization_mode = str(normalization_mode or "uint8_255").lower()
        if normalization_mode not in {"uint8_255", "tile_percentile", "scene_percentile"}:
            raise ValueError("normalization_mode must be one of: uint8_255, tile_percentile, scene_percentile")
        train_config.normalization_mode = normalization_mode
        mosaic_active = DEFAULT_MOSAIC_MODE == "auto" if mosaic_enabled is None else bool(mosaic_enabled)
        train_config.mosaic_enabled = mosaic_active and len(train_scenes) > 1

        val_config = replace(
            train_config,
            positive_stride_factor=1.0,
            hard_negative_stride_factor=1.0,
            negative_stride_factor=1.0,
            virtual_epoch_multiplier=1,
            positive_repeat_factor=1,
            hard_negative_repeat_factor=1,
            negative_repeat_factor=1,
            batch_positive_fraction=None,
            batch_hard_negative_fraction=None,
            batch_negative_fraction=None,
            max_empty_tile_share=None,
            augmentations={},
            apply_random_augmentations=False,
            shuffle=False,
            augmentation_level=0,
        )
        val_config.mosaic_enabled = mosaic_active and len(val_scenes) > 1

        train_dataset = TrainingTileDataset(train_scenes, annotation, train_config, train=True)
        val_dataset = TrainingTileDataset(val_scenes, annotation, val_config, train=False)
        warnings = list(train_dataset.warnings) + list(val_dataset.warnings)
        train_summary = _dataset_summary(train_dataset, train_config, augmentation_level=augmentation_level)
        val_summary = _dataset_summary(val_dataset, val_config, augmentation_level=0)
        return TilePreparationBundle(
            config=train_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_summary=train_summary,
            val_summary=val_summary,
            warnings=warnings,
        )

    @staticmethod
    def train_dataloader(
        bundle: TilePreparationBundle,
        batch_size: int,
        workers: int | None = None,
        prefetch_factor: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        *,
        seed: int | None = None,
    ):
        return make_tile_dataloader(
            bundle.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=bundle.config.seed if seed is None else int(seed),
            workers=workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    @staticmethod
    def val_dataloader(
        bundle: TilePreparationBundle,
        batch_size: int,
        workers: int | None = None,
        prefetch_factor: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        *,
        seed: int | None = None,
    ):
        return make_tile_dataloader(
            bundle.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=bundle.config.seed if seed is None else int(seed),
            workers=workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    @staticmethod
    def train_batch_iterator(bundle: TilePreparationBundle, batch_size: int):
        return iter_dataset_batches(bundle.train_dataset, batch_size, shuffle=True, seed=bundle.config.seed)

    @staticmethod
    def val_batch_iterator(bundle: TilePreparationBundle, batch_size: int):
        return iter_dataset_batches(bundle.val_dataset, batch_size, shuffle=False, seed=bundle.config.seed)

    @staticmethod
    def write_debug_report(
        *,
        input_dir: Path,
        output_dir: Path | None = None,
        config: TilePreparationConfig,
        annotation_crs: str | None = DEFAULT_ANNOTATION_CRS,
        allow_inferred_annotation_crs: bool = DEFAULT_ALLOW_INFERRED_ANNOTATION_CRS,
        max_tile_examples: int = 32,
        max_augmentation_tiles: int = 8,
    ) -> dict:
        return generate_annotated_tile_report(
            input_dir,
            output_dir=output_dir,
            config=config,
            annotation_crs=annotation_crs,
            allow_inferred_annotation_crs=allow_inferred_annotation_crs,
            max_tile_examples=max_tile_examples,
            max_augmentation_tiles=max_augmentation_tiles,
            augmentation_mode=config.augmentation_mode,
            augmentation_seed=config.augmentation_seed,
        )


def _dataset_summary(dataset: TrainingTileDataset, config: TilePreparationConfig, *, augmentation_level: int) -> dict:
    return {
        **summarize_tile_records(dataset.base_records),
        "records": len(dataset.records),
        "base_records": len(dataset.base_records),
        "skipped_fully_invalid_tiles": int(dataset.metadata.get("skipped_fully_invalid_tiles", 0)),
        "skipped_low_valid_share_tiles": int(dataset.metadata.get("skipped_low_valid_share_tiles", 0)),
        "min_valid_pixel_share": config.min_valid_pixel_share,
        "drop_fully_invalid_tiles": config.drop_fully_invalid_tiles,
        "augmentation_level": augmentation_level,
        "mosaic_enabled": config.mosaic_enabled,
        "cutout_mask_mode": config.cutout_mask_mode,
        "normalization_mode": config.normalization_mode,
    }


def bundle_to_jsonable(bundle: TilePreparationBundle) -> dict:
    return {
        "train_summary": bundle.train_summary,
        "val_summary": bundle.val_summary,
        "warnings": bundle.warnings,
        "config": {
            "tile_size": bundle.config.tile_size,
            "stride": bundle.config.stride,
            "positive_stride_factor": bundle.config.positive_stride_factor,
            "hard_negative_stride_factor": bundle.config.hard_negative_stride_factor,
            "negative_stride_factor": bundle.config.negative_stride_factor,
            "positive_repeat_factor": bundle.config.positive_repeat_factor,
            "hard_negative_repeat_factor": bundle.config.hard_negative_repeat_factor,
            "negative_repeat_factor": bundle.config.negative_repeat_factor,
            "max_empty_tile_share": bundle.config.max_empty_tile_share,
            "mosaic_enabled": bundle.config.mosaic_enabled,
            "cutout_mask_mode": bundle.config.cutout_mask_mode,
            "drop_fully_invalid_tiles": bundle.config.drop_fully_invalid_tiles,
            "min_valid_pixel_share": bundle.config.min_valid_pixel_share,
            "augmentation_level": bundle.config.augmentation_level,
            "augmentations": bundle.config.augmentations,
            "normalization_mode": bundle.config.normalization_mode,
        },
    }


def write_bundle_summary(bundle: TilePreparationBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle_to_jsonable(bundle), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
