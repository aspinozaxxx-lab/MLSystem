from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .annotations import load_annotation_geometries
from .augmentations import apply_training_augmentation
from .config import AnnotationInput, SceneInput, TilePreparationConfig
from .iterator import build_tile_records, build_validation_tile_records
from .mask_rasterizer import rasterize_mask_for_window
from .mosaic import read_mosaic_window
from .raster_reader import format_training_image, read_training_image
from .records import ReadyTileSample, TileRecordBuildResult, TileSampleRecord
from .validity import read_valid_data_mask_with_source


class TrainingTileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scenes: list[SceneInput],
        annotation: AnnotationInput,
        config: TilePreparationConfig,
        records: list[TileSampleRecord] | None = None,
        *,
        train: bool = True,
        build_result: TileRecordBuildResult | None = None,
    ) -> None:
        self.scenes = scenes
        self.annotation = annotation
        self.config = config
        self.train = bool(train)
        if build_result is None:
            build_result = build_tile_records(scenes, annotation, config) if train else build_validation_tile_records(scenes, annotation, config)
        self.build_result = build_result
        self.records = list(records) if records is not None else list(build_result.records)
        self.base_records = list(build_result.base_records or self.records)
        self.scene_reports = list(build_result.scene_reports)
        self.warnings = list(build_result.warnings)
        self.metadata = dict(build_result.metadata)
        self._datasets: dict[str, Any] = {}
        self._annotation_cache: dict[str, Any] = {}
        self._scene_id_by_path = {str(scene.image_path): scene.resolved_scene_id() for scene in scenes}
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def __getitem__(self, index: int) -> ReadyTileSample:
        record = self.records[int(index)]
        if not record.image_path:
            raise ValueError(f"record {record.record_id} has no image_path")
        ds = self._dataset(record.image_path)
        annotation_geoms = self._annotation_geometries(record.image_path, ds)
        valid = read_valid_data_mask_with_source(ds, record, mode=self.config.valid_pixel_mode)
        mosaic_metadata: dict[str, Any] = {}
        if self.config.mosaic_enabled:
            mosaic = read_mosaic_window(ds, self._neighbor_datasets(record.image_path), record, self.config)
            valid_mask = mosaic.valid_mask
            image = format_training_image(mosaic.image, self.config)
            mosaic_metadata = {
                "mosaic_sources": mosaic.source_scenes,
                "mosaic_filled_pixel_count": mosaic.filled_pixel_count,
                "mosaic_unfilled_pixel_count": mosaic.unfilled_pixel_count,
                "valid_pixel_share_before_mosaic": mosaic.anchor_valid_pixel_share,
                "valid_pixel_share_after_mosaic": mosaic.final_valid_pixel_share,
                "mosaic_warnings": mosaic.warnings,
            }
            valid_source = mosaic.valid_data_source
        else:
            valid_mask = valid.mask
            image = read_training_image(ds, record, self.config)
            valid_source = valid.source
        mask_result = rasterize_mask_for_window(
            ds,
            annotation_geoms.geometries,
            record,
            all_touched=self.config.all_touched,
            valid_mask=valid_mask if self.config.clip_mask_to_valid_data else None,
        )
        mask = mask_result.mask
        metadata: dict[str, Any] = {
            "geometries_intersecting": mask_result.geom_count,
            "valid_pixel_share": mask_result.valid_pixel_share,
            "invalid_pixel_share": 1.0 - mask_result.valid_pixel_share,
            "mask_pixels_before_valid_clip": mask_result.raw_positive_pixels,
            "mask_pixels_after_valid_clip": mask_result.clipped_positive_pixels,
            "valid_data_source": valid_source,
            **mosaic_metadata,
        }
        if self.train and self.config.apply_random_augmentations and any(bool(value) for value in (self.config.augmentations or {}).values()):
            image, mask, augmentation_metadata = apply_training_augmentation(
                image,
                mask,
                self.config.augmentations,
                seed=int(self.config.seed) + self._epoch * 1_000_003 + int(index),
                cutout_mask_mode=self.config.cutout_mask_mode,
            )
            metadata["augmentation"] = augmentation_metadata
        if image.ndim == 3 and image.shape[0] <= 16:
            out_image = image.astype("float32")
        elif image.ndim == 3:
            out_image = np.transpose(image, (2, 0, 1)).astype("float32")
        else:
            raise ValueError(f"unexpected image shape for {record.record_id}: {image.shape}")
        out_mask = mask.astype("float32")[None, :, :]
        return ReadyTileSample(image=out_image, mask=out_mask, record=record, metadata=metadata)

    def sample_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(self.records):
            row = record.to_dict()
            row["sample_index"] = index
            row["virtual_record"] = record.repeat_index is not None or record.source == "virtual_repeat"
            row["positive_tile"] = record.kind in {"positive", "partial_positive"}
            row["gt_positive_pixels"] = int(record.positive_pixels)
            rows.append(row)
        return rows

    def close(self) -> None:
        for ds in self._datasets.values():
            ds.close()
        self._datasets.clear()

    def _dataset(self, image_path: str) -> Any:
        ds = self._datasets.get(image_path)
        if ds is None:
            import rasterio

            ds = rasterio.open(str(image_path))
            self._datasets[image_path] = ds
        return ds

    def _neighbor_datasets(self, image_path: str) -> list[tuple[str, Any]]:
        if not self.config.mosaic_enabled:
            return []
        neighbors: list[tuple[str, Any]] = []
        for scene in self.scenes:
            path = str(scene.image_path)
            if path == image_path:
                continue
            neighbors.append((scene.resolved_scene_id(), self._dataset(path)))
        return neighbors

    def _annotation_geometries(self, image_path: str, ds: Any) -> Any:
        cached = self._annotation_cache.get(image_path)
        if cached is None:
            cached = load_annotation_geometries(self.annotation, raster_crs=ds.crs)
            self._annotation_cache[image_path] = cached
        return cached


def iter_dataset_batches(
    dataset: TrainingTileDataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[list[int], torch.Tensor, torch.Tensor]]:
    indices = list(range(len(dataset)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    batch_size = max(1, int(batch_size))
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        samples = [dataset[index] for index in batch_indices]
        x = torch.from_numpy(np.stack([sample.image for sample in samples]))
        y = torch.from_numpy(np.stack([sample.mask for sample in samples]))
        yield batch_indices, x, y
