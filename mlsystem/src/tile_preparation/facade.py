from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .config import AnnotationInput, SceneInput, TilePreparationConfig
from .dataset import TrainingTileDataset, iter_dataset_batches
from .report import generate_annotated_tile_report
from .summary import summarize_tile_records


@dataclass(frozen=True)
class TilePreparationSimpleParams:
    tile_size: int = 768
    stride: int | None = None
    stride_ratio: float = 0.67
    augmentation_level: int = 2
    mosaic_mode: Literal["off", "auto", "force"] = "auto"
    max_empty_tile_share: float | None = None
    seed: int = 42
    input_bands: list[int] | None = None


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
    tile_size: int = 768,
    stride: int | None = None,
    stride_ratio: float = 0.67,
    max_empty_tile_share: float | None = None,
    seed: int = 42,
    input_bands: list[int] | None = None,
) -> TilePreparationConfig:
    level = max(0, min(3, int(level)))
    tile_size = max(1, int(tile_size))
    effective_stride = max(1, int(stride if stride is not None else round(tile_size * float(stride_ratio))))

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
    if max_empty_tile_share is not None:
        preset["max_empty_tile_share"] = max_empty_tile_share
    return TilePreparationConfig(
        tile_size=tile_size,
        stride=effective_stride,
        input_bands=input_bands,
        seed=seed,
        augmentation_level=level,
        apply_random_augmentations=bool(preset["augmentations"]),
        cutout_mask_mode="erase",
        **preset,
    )


class TilePreparationFacade:
    @staticmethod
    def default_config(
        *,
        tile_size: int = 768,
        stride: int | None = None,
        stride_ratio: float = 0.67,
        augmentation_level: int = 2,
        mosaic_mode: str = "auto",
        max_empty_tile_share: float | None = None,
        seed: int = 42,
        input_bands: list[int] | None = None,
    ) -> TilePreparationConfig:
        config = config_from_augmentation_level(
            augmentation_level,
            tile_size=tile_size,
            stride=stride,
            stride_ratio=stride_ratio,
            max_empty_tile_share=max_empty_tile_share,
            seed=seed,
            input_bands=input_bands,
        )
        if mosaic_mode == "force":
            config.mosaic_enabled = True
        elif mosaic_mode == "off":
            config.mosaic_enabled = False
        return config

    @staticmethod
    def from_scene_list(
        *,
        images_root: Path,
        scene_list_path: Path,
        annotation_path: Path,
        tile_size: int = 768,
        stride: int | None = None,
        stride_ratio: float = 0.67,
        augmentation_level: int = 2,
        mosaic_mode: str = "auto",
        train_fraction: float = 0.8,
        seed: int = 42,
        input_bands: list[int] | None = None,
    ) -> TilePreparationBundle:
        scenes = _read_scene_list(images_root, scene_list_path)
        train_scenes, val_scenes, warnings = _split_scenes(scenes, train_fraction=train_fraction, seed=seed)
        annotation = AnnotationInput(Path(annotation_path), annotation_crs="auto", allow_inferred_annotation_crs=True)
        params = TilePreparationSimpleParams(
            tile_size=tile_size,
            stride=stride,
            stride_ratio=stride_ratio,
            augmentation_level=augmentation_level,
            mosaic_mode=mosaic_mode,  # type: ignore[arg-type]
            seed=seed,
            input_bands=input_bands,
        )
        bundle = TilePreparationFacade.build_datasets(train_scenes=train_scenes, val_scenes=val_scenes, annotation=annotation, params=params)
        bundle.warnings.extend(warnings)
        return bundle

    @staticmethod
    def build_datasets(
        *,
        train_scenes: list[SceneInput],
        val_scenes: list[SceneInput],
        annotation: AnnotationInput,
        params: TilePreparationSimpleParams,
    ) -> TilePreparationBundle:
        train_config = TilePreparationFacade.default_config(
            tile_size=params.tile_size,
            stride=params.stride,
            stride_ratio=params.stride_ratio,
            augmentation_level=params.augmentation_level,
            mosaic_mode=params.mosaic_mode,
            max_empty_tile_share=params.max_empty_tile_share,
            seed=params.seed,
            input_bands=params.input_bands,
        )
        train_config.mosaic_enabled = params.mosaic_mode == "force" or (params.mosaic_mode == "auto" and len(train_scenes) > 1)

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
        )
        val_config.mosaic_enabled = params.mosaic_mode == "force" or (params.mosaic_mode == "auto" and len(val_scenes) > 1)

        train_dataset = TrainingTileDataset(train_scenes, annotation, train_config, train=True)
        val_dataset = TrainingTileDataset(val_scenes, annotation, val_config, train=False)
        warnings = list(train_dataset.warnings) + list(val_dataset.warnings)
        train_summary = {
            **summarize_tile_records(train_dataset.base_records),
            "records": len(train_dataset.records),
            "base_records": len(train_dataset.base_records),
            "skipped_fully_invalid_tiles": int(train_dataset.metadata.get("skipped_fully_invalid_tiles", 0)),
            "skipped_low_valid_share_tiles": int(train_dataset.metadata.get("skipped_low_valid_share_tiles", 0)),
            "min_valid_pixel_share": train_config.min_valid_pixel_share,
            "drop_fully_invalid_tiles": train_config.drop_fully_invalid_tiles,
            "augmentation_level": params.augmentation_level,
            "mosaic_enabled": train_config.mosaic_enabled,
            "cutout_mask_mode": train_config.cutout_mask_mode,
        }
        val_summary = {
            **summarize_tile_records(val_dataset.base_records),
            "records": len(val_dataset.records),
            "base_records": len(val_dataset.base_records),
            "skipped_fully_invalid_tiles": int(val_dataset.metadata.get("skipped_fully_invalid_tiles", 0)),
            "skipped_low_valid_share_tiles": int(val_dataset.metadata.get("skipped_low_valid_share_tiles", 0)),
            "min_valid_pixel_share": val_config.min_valid_pixel_share,
            "drop_fully_invalid_tiles": val_config.drop_fully_invalid_tiles,
            "augmentation_level": 0,
            "mosaic_enabled": val_config.mosaic_enabled,
            "cutout_mask_mode": val_config.cutout_mask_mode,
        }
        return TilePreparationBundle(
            config=train_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_summary=train_summary,
            val_summary=val_summary,
            warnings=warnings,
        )

    @staticmethod
    def train_dataloader(bundle: TilePreparationBundle, batch_size: int, workers: int = 0):
        _ = workers
        return iter_dataset_batches(bundle.train_dataset, batch_size, shuffle=True, seed=bundle.config.seed)

    @staticmethod
    def val_dataloader(bundle: TilePreparationBundle, batch_size: int, workers: int = 0):
        _ = workers
        return iter_dataset_batches(bundle.val_dataset, batch_size, shuffle=False, seed=bundle.config.seed)

    @staticmethod
    def write_debug_report(
        *,
        input_dir: Path,
        output_dir: Path | None = None,
        config: TilePreparationConfig,
        annotation_crs: str | None = "auto",
        allow_inferred_annotation_crs: bool = True,
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


def _read_scene_list(images_root: Path, scene_list_path: Path) -> list[SceneInput]:
    root = Path(images_root)
    rows = Path(scene_list_path).read_text(encoding="utf-8").splitlines()
    scenes: list[SceneInput] = []
    for line in rows:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value)
        image_path = path if path.is_absolute() else root / path
        if not image_path.is_file() or image_path.suffix.lower() not in {".tif", ".tiff"}:
            matches = sorted(root.glob(f"{value}.tif")) + sorted(root.glob(f"{value}.tiff"))
            if matches:
                image_path = matches[0]
        if not image_path.is_file() or image_path.suffix.lower() not in {".tif", ".tiff"}:
            raise FileNotFoundError(f"Scene list entry does not resolve to an image: {value}")
        scenes.append(SceneInput(image_path=image_path, scene_id=image_path.stem))
    if not scenes:
        raise ValueError(f"Scene list is empty: {scene_list_path}")
    return scenes


def _split_scenes(scenes: list[SceneInput], *, train_fraction: float, seed: int) -> tuple[list[SceneInput], list[SceneInput], list[str]]:
    warnings: list[str] = []
    shuffled = list(scenes)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        warnings.append("scene list contains one scene; train-smoke reuses it for validation")
        return shuffled, shuffled, warnings
    train_count = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * float(train_fraction)))))
    return shuffled[:train_count], shuffled[train_count:], warnings


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
        },
    }


def write_bundle_summary(bundle: TilePreparationBundle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle_to_jsonable(bundle), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
