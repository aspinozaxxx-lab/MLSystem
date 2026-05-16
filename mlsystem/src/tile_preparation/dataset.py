from __future__ import annotations

import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .annotations import load_annotation_geometries
from .augmentations import apply_training_augmentation
from .config import AnnotationInput, SceneInput, TilePreparationConfig
from .footprint import SceneFootprint, rasterize_footprint_for_window
from .geometry_index import GeometryWindowIndex
from .iterator import build_tile_records, build_validation_tile_records
from .mask_rasterizer import rasterize_mask_for_window
from .mosaic import read_mosaic_window
from .profiling import TilePrepProfiler
from .raster_reader import compute_scene_percentile_stats, format_training_image, read_band_window
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
        self._geometry_index_cache: dict[str, GeometryWindowIndex] = {}
        self._scene_stats_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._footprint_cache: dict[str, SceneFootprint] = {}
        self._profile = TilePrepProfiler()
        self._scene_id_by_path = {str(scene.image_path): scene.resolved_scene_id() for scene in scenes}
        self._epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def __getitem__(self, index: int) -> ReadyTileSample:
        item_started = time.perf_counter()
        sample_profile: dict[str, float] = {}
        index = int(index)
        record = self.records[index]
        if not record.image_path:
            raise ValueError(f"record {record.record_id} has no image_path")
        ds = self._dataset(record.image_path)
        annotation_geoms = self._annotation_geometries(record.image_path, ds)
        mosaic_metadata: dict[str, Any] = {}
        use_mosaic = self.config.mosaic_enabled and bool(record.metadata.get("mosaic_needed", False))
        if use_mosaic:
            candidate_scene_ids = [str(item) for item in record.metadata.get("mosaic_candidate_scene_ids") or []]
            with self._profile.time_sample("mosaic_sec", sample_profile):
                mosaic = read_mosaic_window(
                    ds,
                    self._neighbor_datasets(record.image_path, candidate_scene_ids=candidate_scene_ids),
                    record,
                    self.config,
                    anchor_footprint=self._footprint(record.image_path),
                    neighbor_footprints=self._neighbor_footprints(record.image_path, candidate_scene_ids=candidate_scene_ids),
                )
            valid_mask = mosaic.valid_mask
            with self._profile.time_sample("normalize_sec", sample_profile):
                image = format_training_image(mosaic.image, self.config, scene_stats=self._scene_stats(record.image_path, ds))
            mosaic_metadata = {
                "mosaic_sources": mosaic.source_scenes,
                "mosaic_filled_pixel_count": mosaic.filled_pixel_count,
                "mosaic_unfilled_pixel_count": mosaic.unfilled_pixel_count,
                "valid_pixel_share_before_mosaic": mosaic.anchor_valid_pixel_share,
                "valid_pixel_share_after_mosaic": mosaic.final_valid_pixel_share,
                "mosaic_candidate_neighbors": mosaic.candidate_neighbors,
                "mosaic_intersecting_neighbors": mosaic.intersecting_neighbors,
                "mosaic_actually_used_neighbors": mosaic.actually_used_neighbors,
                "mosaic_skipped_non_intersecting_neighbors": mosaic.skipped_non_intersecting_neighbors,
                "mosaic_attempted": mosaic.attempted,
                "mosaic_skipped_no_candidates": mosaic.skipped_no_candidates,
                "mosaic_skipped_no_gap": mosaic.skipped_no_gap,
                "mosaic_warped_vrt_calls": mosaic.warped_vrt_calls,
                "mosaic_used_neighbors": len(mosaic.actually_used_neighbors),
                "mosaic_zero_fill_attempts": mosaic.zero_fill_attempts,
                "mosaic_neighbor_valid_from_footprint": mosaic.neighbor_valid_from_footprint,
                "mosaic_neighbor_valid_from_pixels_fallback": mosaic.neighbor_valid_from_pixels_fallback,
                "mosaic_warnings": mosaic.warnings,
            }
            valid_source = mosaic.valid_data_source
        else:
            valid_mask, valid_source = self._valid_mask_for_record(ds, record, sample_profile)
            with self._profile.time_sample("read_image_sec", sample_profile):
                arr = read_band_window(ds, record, bands=self.config.input_bands)
            with self._profile.time_sample("normalize_sec", sample_profile):
                image = format_training_image(arr, self.config, scene_stats=self._scene_stats(record.image_path, ds))
            mosaic_metadata = self._mosaic_skip_metadata(record)
        with self._profile.time_sample("rasterize_sec", sample_profile):
            mask_result = rasterize_mask_for_window(
                ds,
                annotation_geoms.geometries,
                record,
                all_touched=self.config.all_touched,
                valid_mask=valid_mask if self.config.clip_mask_to_valid_data else None,
                geometry_index=self._geometry_index(record.image_path, annotation_geoms.geometries),
                profile=sample_profile if self._profile.enabled else None,
            )
        mask = mask_result.mask
        metadata: dict[str, Any] = {
            "geometries_intersecting": mask_result.geom_count,
            "valid_pixel_share": mask_result.valid_pixel_share,
            "invalid_pixel_share": 1.0 - mask_result.valid_pixel_share,
            "mask_pixels_before_valid_clip": mask_result.raw_positive_pixels,
            "mask_pixels_after_valid_clip": mask_result.clipped_positive_pixels,
            "valid_data_source": valid_source,
            "mosaic_needed": bool(record.metadata.get("mosaic_needed", False)),
            "mosaic_reason": record.metadata.get("mosaic_reason"),
            "mosaic_side": record.metadata.get("mosaic_side"),
            "mosaic_candidate_scene_ids": list(record.metadata.get("mosaic_candidate_scene_ids") or []),
            "mosaic_estimated_gap_share": float(record.metadata.get("mosaic_estimated_gap_share", 0.0) or 0.0),
            "mosaic_estimated_neighbor_cover_share": float(record.metadata.get("mosaic_estimated_neighbor_cover_share", 0.0) or 0.0),
            "mosaic_overlap_record": bool(record.metadata.get("mosaic_overlap_record", False)),
            **mosaic_metadata,
        }
        if self.train and self.config.apply_random_augmentations and any(bool(value) for value in (self.config.augmentations or {}).values()):
            with self._profile.time_sample("augmentation_sec", sample_profile):
                image, mask, augmentation_metadata = apply_training_augmentation(
                    image,
                    mask,
                    self.config.augmentations,
                    seed=int(self.config.seed) + self._epoch * 1_000_003 + int(index),
                    cutout_mask_mode=self.config.cutout_mask_mode,
                )
            metadata["augmentation"] = augmentation_metadata
        metadata["sample_index"] = index
        if image.ndim == 3 and image.shape[0] <= 16:
            out_image = image.astype("float32")
        elif image.ndim == 3:
            out_image = np.transpose(image, (2, 0, 1)).astype("float32")
        else:
            raise ValueError(f"unexpected image shape for {record.record_id}: {image.shape}")
        out_mask = mask.astype("float32")[None, :, :]
        total_getitem_sec = time.perf_counter() - item_started
        self._profile.add("total_getitem_sec", total_getitem_sec)
        if self._profile.enabled:
            sample_profile["total_getitem_sec"] = float(total_getitem_sec)
            metadata["tile_prep_profile"] = dict(sample_profile)
        return ReadyTileSample(image=out_image, mask=out_mask, record=record, metadata=metadata)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_datasets"] = {}
        state["_annotation_cache"] = {}
        state["_geometry_index_cache"] = {}
        state["_scene_stats_cache"] = {}
        state["_footprint_cache"] = {}
        state["_profile"] = TilePrepProfiler()
        return state

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
        self._annotation_cache.clear()
        self._geometry_index_cache.clear()
        self._scene_stats_cache.clear()
        self._footprint_cache.clear()

    def profile_summary(self, *, reset: bool = False) -> dict[str, float]:
        return self._profile.summary(reset=reset)

    def _dataset(self, image_path: str) -> Any:
        ds = self._datasets.get(image_path)
        if ds is None:
            import rasterio

            ds = rasterio.open(str(image_path))
            self._datasets[image_path] = ds
        return ds

    def _neighbor_datasets(self, image_path: str, *, candidate_scene_ids: list[str] | None = None) -> list[tuple[str, Any]]:
        if not self.config.mosaic_enabled:
            return []
        candidate_set = {str(item) for item in candidate_scene_ids or []}
        neighbors: list[tuple[str, Any]] = []
        for scene in self.scenes:
            path = str(scene.image_path)
            if path == image_path:
                continue
            scene_id = scene.resolved_scene_id()
            if candidate_set and scene_id not in candidate_set:
                continue
            neighbors.append((scene_id, self._dataset(path)))
        return neighbors

    def _neighbor_footprints(self, image_path: str, *, candidate_scene_ids: list[str] | None = None) -> dict[str, SceneFootprint]:
        if not self.config.mosaic_enabled:
            return {}
        candidate_set = {str(item) for item in candidate_scene_ids or []}
        footprints: dict[str, SceneFootprint] = {}
        for scene in self.scenes:
            path = str(scene.image_path)
            if path == image_path:
                continue
            scene_id = scene.resolved_scene_id()
            if candidate_set and scene_id not in candidate_set:
                continue
            footprint = self._footprint(path)
            if footprint is not None:
                footprints[scene_id] = footprint
        return footprints

    def _annotation_geometries(self, image_path: str, ds: Any) -> Any:
        cached = self._annotation_cache.get(image_path)
        if cached is None:
            cached = load_annotation_geometries(self.annotation, raster_crs=ds.crs)
            self._annotation_cache[image_path] = cached
        return cached

    def _geometry_index(self, image_path: str, geometries: list[Any]) -> GeometryWindowIndex:
        cached = self._geometry_index_cache.get(image_path)
        if cached is None:
            cached = GeometryWindowIndex.build(geometries)
            self._geometry_index_cache[image_path] = cached
        return cached

    def _scene_stats(self, image_path: str, ds: Any) -> tuple[np.ndarray, np.ndarray] | None:
        if str(getattr(self.config, "normalization_mode", "uint8_255") or "uint8_255").lower() != "scene_percentile":
            return None
        cached = self._scene_stats_cache.get(image_path)
        if cached is None:
            cached = compute_scene_percentile_stats(ds, bands=self.config.input_bands)
            self._scene_stats_cache[image_path] = cached
        return cached

    def _valid_mask_for_record(self, ds: Any, record: TileSampleRecord, sample_profile: dict[str, float]) -> tuple[np.ndarray | None, str]:
        mode = str(record.metadata.get("footprint_valid_mask_mode") or "")
        if mode == "full":
            return None, str(record.metadata.get("footprint_source") or "footprint")
        if mode == "footprint_boundary":
            footprint = self._footprint(record.image_path or "")
            if footprint is not None:
                return rasterize_footprint_for_window(footprint, record), str(record.metadata.get("footprint_source") or footprint.source)
        with self._profile.time_sample("read_valid_mask_sec", sample_profile):
            valid = read_valid_data_mask_with_source(ds, record, mode=self.config.valid_pixel_mode)
        return valid.mask, f"{valid.source}:fallback"

    def _footprint(self, image_path: str) -> SceneFootprint | None:
        cached = self._footprint_cache.get(image_path)
        if cached is not None:
            return cached
        payload = (self.metadata.get("footprints_by_image_path") or {}).get(image_path)
        if not isinstance(payload, dict):
            return None
        footprint = SceneFootprint.from_metadata(payload)
        self._footprint_cache[image_path] = footprint
        return footprint

    def _mosaic_skip_metadata(self, record: TileSampleRecord) -> dict[str, Any]:
        reason = str(record.metadata.get("mosaic_reason") or ("mosaic_disabled" if not self.config.mosaic_enabled else "not_needed"))
        candidates = list(record.metadata.get("mosaic_candidate_scene_ids") or [])
        return {
            "mosaic_sources": [record.scene_id],
            "mosaic_filled_pixel_count": 0,
            "mosaic_unfilled_pixel_count": int(record.width * record.height * max(0.0, float(record.invalid_pixel_share))),
            "valid_pixel_share_before_mosaic": float(record.metadata.get("valid_pixel_share_before_mosaic", record.valid_pixel_share) or record.valid_pixel_share),
            "valid_pixel_share_after_mosaic": float(record.valid_pixel_share),
            "mosaic_candidate_neighbors": int(len(candidates)),
            "mosaic_intersecting_neighbors": 0,
            "mosaic_actually_used_neighbors": [],
            "mosaic_skipped_non_intersecting_neighbors": int(len(candidates)),
            "mosaic_attempted": False,
            "mosaic_skipped_fully_inside": reason == "fully_inside_footprint",
            "mosaic_skipped_no_gap": reason == "no_gap",
            "mosaic_skipped_no_candidates": reason in {"no_neighbor_for_gap", "no_raster_transform", "missing_anchor_footprint"},
            "mosaic_warped_vrt_calls": 0,
            "mosaic_used_neighbors": 0,
            "mosaic_zero_fill_attempts": 0,
            "mosaic_neighbor_valid_from_footprint": 0,
            "mosaic_neighbor_valid_from_pixels_fallback": 0,
            "mosaic_warnings": [],
        }


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
