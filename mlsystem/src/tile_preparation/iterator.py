from __future__ import annotations

import itertools
import math
import os
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from .annotations import AnnotationGeometrySet, load_annotation_geometries
from .augmentations import apply_training_augmentation
from .config import AnnotationInput, SceneInput, TilePreparationConfig
from .mask_rasterizer import positive_pixel_bbox, rasterize_mask_for_window
from .mosaic import read_mosaic_window
from .raster_reader import format_training_image, read_training_image
from .records import ReadyTileSample, TileKind, TileRecordBuildResult, TileSampleRecord, TileWindow
from .summary import summarize_tile_records
from .validity import read_valid_data_mask_with_source
from .windows import generate_window_grid_for_scene


def build_tile_records(
    scenes: list[SceneInput],
    annotation: AnnotationInput,
    config: TilePreparationConfig,
) -> TileRecordBuildResult:
    warnings: list[str] = []
    base_records: list[TileSampleRecord] = []
    scene_reports: list[dict[str, Any]] = []
    annotation_summary: dict[str, Any] | None = None
    annotation_by_scene: dict[str, dict[str, Any]] = {}
    skip_totals = {"skipped_fully_invalid_tiles": 0, "skipped_low_valid_share_tiles": 0}
    rng = random.Random(config.seed)
    open_datasets: dict[str, Any] = {}
    scene_ids_by_path: dict[str, str] = {}
    try:
        for scene in scenes:
            image_path = _rasterio_path(scene.image_path)
            open_datasets[image_path] = rasterio.open(image_path)
            scene_ids_by_path[image_path] = scene.resolved_scene_id()

        total_scenes = len(scenes)
        for scene_index, scene in enumerate(scenes, start=1):
            if bool(scene.metadata.get("mosaic_neighbor_only")):
                continue
            image_path = _rasterio_path(scene.image_path)
            scene_id = scene.resolved_scene_id()
            _log_progress(f"build_train_records scene={scene_id} index={scene_index}/{total_scenes}")
            ds = open_datasets[image_path]
            annotation_geoms = load_annotation_geometries(annotation, raster_crs=ds.crs)
            annotation_summary = annotation_geoms.to_dict(annotation.geojson_path)
            annotation_by_scene[scene_id] = dict(annotation_summary)
            warnings.extend(annotation_geoms.warnings)
            neighbor_datasets = _neighbor_datasets(image_path, open_datasets, scene_ids_by_path, config)
            scene_records, scene_skip_counts = _build_scene_records(ds, scene_id, image_path, annotation_geoms, config, neighbor_datasets=neighbor_datasets)
            _log_progress(f"built_train_records scene={scene_id} records={len(scene_records)}")
            for key in skip_totals:
                skip_totals[key] += int(scene_skip_counts.get(key, 0))
            before_limit = list(scene_records)
            scene_records, limit_warnings = limit_empty_tile_share(scene_records, config.max_empty_tile_share, seed=config.seed)
            warnings.extend(limit_warnings)
            if config.max_records_per_scene is not None and len(scene_records) > config.max_records_per_scene:
                scene_records = _limit_records(scene_records, config.max_records_per_scene, rng)
            base_records.extend(scene_records)
            scene_summary = summarize_tile_records(scene_records)
            scene_reports.append(
                {
                    "scene": scene_id,
                    "image_path": str(image_path),
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "bands": int(ds.count),
                    "crs": str(ds.crs) if ds.crs else None,
                    "raster_crs": str(ds.crs) if ds.crs else None,
                    "annotation_crs": annotation_summary.get("annotation_crs"),
                    "annotation_crs_source": annotation_summary.get("annotation_crs_source"),
                    "transformed_to_raster_crs": annotation_summary.get("transformed_to_raster_crs"),
                    "records_before_empty_limit": len(before_limit),
                    "records_after_empty_limit": len(scene_records),
                    **scene_skip_counts,
                    **scene_summary,
                }
            )
    finally:
        for ds in open_datasets.values():
            ds.close()

    if config.max_records is not None and len(base_records) > config.max_records:
        base_records = _limit_records(base_records, config.max_records, rng)
    if config.shuffle:
        rng.shuffle(base_records)
    virtual_records = apply_virtual_repeats(base_records, config)
    if config.virtual_epoch_multiplier > 1:
        virtual_records = virtual_records * config.virtual_epoch_multiplier
    virtual_records, balance_warnings = build_balanced_epoch_records(virtual_records, config, seed=config.seed + 17)
    warnings.extend(balance_warnings)
    return TileRecordBuildResult(
        records=virtual_records,
        base_records=base_records,
        warnings=warnings,
        scene_reports=scene_reports,
        metadata={
            "annotation": annotation_summary,
            "annotation_by_scene": annotation_by_scene,
            "positive_stride": config.positive_stride,
            "hard_negative_stride": config.hard_negative_stride,
            "negative_stride": config.negative_stride,
            "base_record_count": len(base_records),
            "virtual_record_count": len(virtual_records),
            "effective_samples_per_epoch": len(virtual_records),
            "min_valid_pixel_share": config.min_valid_pixel_share,
            "drop_fully_invalid_tiles": config.drop_fully_invalid_tiles,
            **skip_totals,
        },
    )


def build_balanced_epoch_records(
    records: list[TileSampleRecord],
    config: TilePreparationConfig,
    *,
    seed: int = 0,
) -> tuple[list[TileSampleRecord], list[str]]:
    requested = {
        "positive": config.batch_positive_fraction,
        "hard_negative": config.batch_hard_negative_fraction,
        "negative": config.batch_negative_fraction,
    }
    if not records or not any(value is not None for value in requested.values()):
        return list(records), []
    rng = random.Random(seed)
    total = len(records)
    groups = {
        "positive": [record for record in records if record.kind in {"positive", "partial_positive"}],
        "hard_negative": [record for record in records if record.kind == "hard_negative"],
        "negative": [record for record in records if record.kind == "negative"],
    }
    warnings: list[str] = []
    explicit_sum = sum(float(value) for value in requested.values() if value is not None)
    remaining_groups = [name for name, value in requested.items() if value is None]
    remaining_fraction = max(0.0, 1.0 - explicit_sum)
    fractions = {
        name: (float(value) if value is not None else remaining_fraction / max(1, len(remaining_groups)))
        for name, value in requested.items()
    }
    if sum(fractions.values()) <= 0:
        return list(records), []
    normalized_total = sum(fractions.values())
    fractions = {name: value / normalized_total for name, value in fractions.items()}
    balanced: list[TileSampleRecord] = []
    for name, fraction in fractions.items():
        count = int(round(total * fraction))
        if count <= 0:
            continue
        group = groups.get(name) or []
        if not group:
            warnings.append(f"batch balancing requested {name} records, but the group is empty")
            continue
        balanced.extend(group[rng.randrange(len(group))] for _ in range(count))
    while len(balanced) < total:
        balanced.append(records[rng.randrange(len(records))])
    rng.shuffle(balanced)
    return balanced[:total], warnings


def build_validation_tile_records(
    scenes: list[SceneInput],
    annotation: AnnotationInput,
    config: TilePreparationConfig,
) -> TileRecordBuildResult:
    warnings: list[str] = []
    records: list[TileSampleRecord] = []
    scene_reports: list[dict[str, Any]] = []
    annotation_summary: dict[str, Any] | None = None
    annotation_by_scene: dict[str, dict[str, Any]] = {}
    skip_totals = {"skipped_fully_invalid_tiles": 0, "skipped_low_valid_share_tiles": 0}
    open_datasets: dict[str, Any] = {}
    scene_ids_by_path: dict[str, str] = {}
    try:
        for scene in scenes:
            image_path = _rasterio_path(scene.image_path)
            open_datasets[image_path] = rasterio.open(image_path)
            scene_ids_by_path[image_path] = scene.resolved_scene_id()
        total_scenes = len(scenes)
        for scene_index, scene in enumerate(scenes, start=1):
            if bool(scene.metadata.get("mosaic_neighbor_only")):
                continue
            image_path = _rasterio_path(scene.image_path)
            scene_id = scene.resolved_scene_id()
            _log_progress(f"build_val_records scene={scene_id} index={scene_index}/{total_scenes}")
            ds = open_datasets[image_path]
            annotation_geoms = load_annotation_geometries(annotation, raster_crs=ds.crs)
            annotation_summary = annotation_geoms.to_dict(annotation.geojson_path)
            annotation_by_scene[scene_id] = dict(annotation_summary)
            warnings.extend(annotation_geoms.warnings)
            neighbor_datasets = _neighbor_datasets(image_path, open_datasets, scene_ids_by_path, config)
            scene_records: list[TileSampleRecord] = []
            scene_skip_counts = {"skipped_fully_invalid_tiles": 0, "skipped_low_valid_share_tiles": 0}
            for window in generate_window_grid_for_scene(ds.width, ds.height, config.tile_size, config.stride, scene_id=scene_id):
                valid = read_valid_data_mask_with_source(ds, window, mode=config.valid_pixel_mode)
                valid_mask = valid.mask
                mosaic_sources: list[str] = []
                mosaic_filled = 0
                mosaic_unfilled = int(valid_mask.size - np.count_nonzero(valid_mask))
                mosaic_candidate_neighbors = 0
                mosaic_intersecting_neighbors = 0
                mosaic_actually_used_neighbors: list[str] = []
                mosaic_skipped_non_intersecting_neighbors = 0
                if config.mosaic_enabled and neighbor_datasets:
                    mosaic = read_mosaic_window(ds, neighbor_datasets, window, config)
                    valid_mask = mosaic.valid_mask
                    mosaic_sources = mosaic.source_scenes
                    mosaic_filled = mosaic.filled_pixel_count
                    mosaic_unfilled = mosaic.unfilled_pixel_count
                    mosaic_candidate_neighbors = mosaic.candidate_neighbors
                    mosaic_intersecting_neighbors = mosaic.intersecting_neighbors
                    mosaic_actually_used_neighbors = mosaic.actually_used_neighbors
                    mosaic_skipped_non_intersecting_neighbors = mosaic.skipped_non_intersecting_neighbors
                final_valid_share = float(np.count_nonzero(valid_mask)) / float(valid_mask.size) if valid_mask.size else 0.0
                if config.drop_fully_invalid_tiles and final_valid_share <= 0.0:
                    scene_skip_counts["skipped_fully_invalid_tiles"] += 1
                    continue
                mask_result = rasterize_mask_for_window(
                    ds,
                    annotation_geoms.geometries,
                    window,
                    all_touched=config.all_touched,
                    valid_mask=valid_mask if config.clip_mask_to_valid_data else None,
                )
                kind, positive_pixels = classify_mask(mask_result.mask, config)
                if kind == "hard_negative":
                    kind = "negative"
                if config.exclude_empty_valid_tiles and final_valid_share < config.min_valid_pixel_share:
                    scene_skip_counts["skipped_low_valid_share_tiles"] += 1
                    continue
                scene_records.append(
                    TileSampleRecord(
                        scene_id=scene_id,
                        image_path=str(image_path),
                        x=int(window.x),
                        y=int(window.y),
                        width=int(window.width),
                        height=int(window.height),
                        tile_size=int(window.tile_size),
                        stride=int(window.stride),
                        kind=kind if kind in {"positive", "partial_positive"} else "negative",
                        positive_pixels=int(positive_pixels),
                        source="validation_grid",
                        geometries_intersecting=int(mask_result.geom_count),
                        valid_pixel_share=mask_result.valid_pixel_share,
                        invalid_pixel_share=1.0 - mask_result.valid_pixel_share,
                        mask_pixels_before_valid_clip=mask_result.raw_positive_pixels,
                        mask_pixels_after_valid_clip=mask_result.clipped_positive_pixels,
                        valid_data_source=valid.source,
                        mosaic_sources=mosaic_sources,
                        mosaic_filled_pixel_count=mosaic_filled,
                        mosaic_unfilled_pixel_count=mosaic_unfilled,
                        metadata={
                            "mosaic_candidate_neighbors": mosaic_candidate_neighbors,
                            "mosaic_intersecting_neighbors": mosaic_intersecting_neighbors,
                            "mosaic_actually_used_neighbors": mosaic_actually_used_neighbors,
                            "mosaic_skipped_non_intersecting_neighbors": mosaic_skipped_non_intersecting_neighbors,
                        },
                    )
                )
            for key in skip_totals:
                skip_totals[key] += int(scene_skip_counts.get(key, 0))
            scene_records.sort(key=lambda item: (item.y, item.x))
            if config.max_records_per_scene is not None and len(scene_records) > config.max_records_per_scene:
                scene_records = scene_records[: config.max_records_per_scene]
            records.extend(scene_records)
            _log_progress(f"built_val_records scene={scene_id} records={len(scene_records)}")
            summary = summarize_tile_records(scene_records)
            scene_reports.append(
                {
                    "scene": scene_id,
                    "image_path": str(image_path),
                    "width": int(ds.width),
                    "height": int(ds.height),
                    "bands": int(ds.count),
                    "crs": str(ds.crs) if ds.crs else None,
                    "raster_crs": str(ds.crs) if ds.crs else None,
                    "annotation_crs": annotation_summary.get("annotation_crs"),
                    "annotation_crs_source": annotation_summary.get("annotation_crs_source"),
                    "transformed_to_raster_crs": annotation_summary.get("transformed_to_raster_crs"),
                    "positive_scene": bool(summary["positive_tiles"] or summary["partial_positive_tiles"]),
                    "samples": len(scene_records),
                    **scene_skip_counts,
                    **summary,
                }
            )
            if config.max_records is not None and len(records) >= config.max_records:
                records = records[: config.max_records]
                break
    finally:
        for ds in open_datasets.values():
            ds.close()
    return TileRecordBuildResult(
        records=records,
        base_records=records,
        warnings=warnings,
        scene_reports=scene_reports,
        metadata={
            "annotation": annotation_summary,
            "annotation_by_scene": annotation_by_scene,
            "positive_stride": config.stride,
            "hard_negative_stride": config.stride,
            "negative_stride": config.stride,
            "base_record_count": len(records),
            "virtual_record_count": len(records),
            "effective_samples_per_epoch": len(records),
            "min_valid_pixel_share": config.min_valid_pixel_share,
            "drop_fully_invalid_tiles": config.drop_fully_invalid_tiles,
            **skip_totals,
        },
    )


def iter_training_tiles(
    scenes: list[SceneInput],
    annotation: AnnotationInput,
    config: TilePreparationConfig,
) -> Iterator[ReadyTileSample]:
    result = build_tile_records(scenes, annotation, config)
    records = list(result.records)
    if config.shuffle:
        random.Random(config.seed + 101).shuffle(records)
    open_datasets: dict[str, Any] = {}
    annotation_cache: dict[str, AnnotationGeometrySet] = {}
    scene_ids_by_path = {_rasterio_path(scene.image_path): scene.resolved_scene_id() for scene in scenes}
    try:
        if config.mosaic_enabled:
            for scene in scenes:
                path = _rasterio_path(scene.image_path)
                open_datasets[path] = rasterio.open(path)
        for index, record in enumerate(records):
            if config.max_records is not None and index >= config.max_records:
                break
            if not record.image_path:
                continue
            ds = open_datasets.get(record.image_path)
            if ds is None:
                ds = rasterio.open(_rasterio_path(record.image_path))
                open_datasets[record.image_path] = ds
            annotation_geoms = annotation_cache.get(record.image_path)
            if annotation_geoms is None:
                annotation_geoms = load_annotation_geometries(annotation, raster_crs=ds.crs)
                annotation_cache[record.image_path] = annotation_geoms
            valid = read_valid_data_mask_with_source(ds, record, mode=config.valid_pixel_mode)
            mosaic_metadata: dict[str, Any] = {}
            if config.mosaic_enabled:
                mosaic = read_mosaic_window(ds, _neighbor_datasets(record.image_path, open_datasets, scene_ids_by_path, config), record, config)
                valid_mask = mosaic.valid_mask
                image = format_training_image(mosaic.image, config)
                valid_source = mosaic.valid_data_source
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
                    "mosaic_warnings": mosaic.warnings,
                }
            else:
                valid_mask = valid.mask
                image = read_training_image(ds, record, config)
                valid_source = valid.source
            mask_result = rasterize_mask_for_window(
                ds,
                annotation_geoms.geometries,
                record,
                all_touched=config.all_touched,
                valid_mask=valid_mask if config.clip_mask_to_valid_data else None,
            )
            mask = mask_result.mask
            metadata: dict[str, Any] = {
                "geometries_intersecting": mask_result.geom_count,
                "valid_pixel_share": mask_result.valid_pixel_share,
                "mask_pixels_before_valid_clip": mask_result.raw_positive_pixels,
                "mask_pixels_after_valid_clip": mask_result.clipped_positive_pixels,
                "valid_data_source": valid_source,
                **mosaic_metadata,
            }
            if config.apply_random_augmentations and any(bool(value) for value in (config.augmentations or {}).values()):
                image, mask, aug_metadata = apply_training_augmentation(
                    image,
                    mask,
                    config.augmentations,
                    seed=config.seed + index,
                    cutout_mask_mode=config.cutout_mask_mode,
                )
                metadata["augmentation"] = aug_metadata
            out_mask = mask.astype("float32")[None, :, :] if config.output_format.startswith("chw") else mask.astype("uint8")
            yield ReadyTileSample(image=image, mask=out_mask, record=record, metadata=metadata)
    finally:
        for ds in open_datasets.values():
            ds.close()


def iter_batches(samples: Iterator[ReadyTileSample], batch_size: int) -> Iterator[list[ReadyTileSample]]:
    batch_size = max(1, int(batch_size))
    iterator = iter(samples)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def apply_virtual_repeats(records: list[TileSampleRecord], config: TilePreparationConfig) -> list[TileSampleRecord]:
    repeated: list[TileSampleRecord] = []
    for record in records:
        if record.kind in {"positive", "partial_positive"}:
            factor = config.positive_repeat_factor
        elif record.kind == "hard_negative":
            factor = config.hard_negative_repeat_factor
        else:
            factor = config.negative_repeat_factor
        base_id = record.base_record_id or record.record_id
        for repeat_index in range(max(1, int(factor))):
            metadata = dict(record.metadata)
            metadata.setdefault("base_record_id", base_id)
            metadata["record_id"] = f"{base_id}:repeat{repeat_index}" if factor > 1 else base_id
            repeated.append(
                TileSampleRecord(
                    scene_id=record.scene_id,
                    image_path=record.image_path,
                    x=record.x,
                    y=record.y,
                    width=record.width,
                    height=record.height,
                    tile_size=record.tile_size,
                    stride=record.stride,
                    kind=record.kind,
                    positive_pixels=record.positive_pixels,
                    source=record.source if repeat_index == 0 else "virtual_repeat",
                    base_record_id=base_id,
                    repeat_index=repeat_index if factor > 1 else None,
                    geometries_intersecting=record.geometries_intersecting,
                    valid_pixel_share=record.valid_pixel_share,
                    invalid_pixel_share=record.invalid_pixel_share,
                    mask_pixels_before_valid_clip=record.mask_pixels_before_valid_clip,
                    mask_pixels_after_valid_clip=record.mask_pixels_after_valid_clip,
                    valid_data_source=record.valid_data_source,
                    mosaic_sources=list(record.mosaic_sources),
                    mosaic_filled_pixel_count=record.mosaic_filled_pixel_count,
                    mosaic_unfilled_pixel_count=record.mosaic_unfilled_pixel_count,
                    metadata=metadata,
                )
            )
    return repeated


def limit_empty_tile_share(
    records: list[TileSampleRecord],
    max_empty_tile_share: float | None,
    *,
    seed: int = 0,
) -> tuple[list[TileSampleRecord], list[str]]:
    if max_empty_tile_share is None:
        return list(records), []
    share = max(0.0, min(1.0, float(max_empty_tile_share)))
    if share >= 1.0 or not records:
        return list(records), []
    empty_indices = [idx for idx, record in enumerate(records) if record.kind in {"negative", "hard_negative"}]
    filled_indices = [idx for idx, record in enumerate(records) if record.kind not in {"negative", "hard_negative"}]
    if not empty_indices:
        return list(records), []
    if not filled_indices:
        return list(records), ["max_empty_tile_share cannot be enforced because there are no positive records"]
    max_empty = int(math.floor((share * len(filled_indices)) / max(1e-12, 1.0 - share)))
    if len(empty_indices) <= max_empty:
        return list(records), []
    rng = random.Random(seed)
    selected_empty = set(rng.sample(empty_indices, max(0, max_empty)))
    selected = set(filled_indices) | selected_empty
    return [record for idx, record in enumerate(records) if idx in selected], [
        f"max_empty_tile_share limited empty records from {len(empty_indices)} to {len(selected_empty)}"
    ]


def _build_scene_records(
    ds: Any,
    scene_id: str,
    image_path: str,
    annotation_geoms: AnnotationGeometrySet,
    config: TilePreparationConfig,
    *,
    neighbor_datasets: list[tuple[str, Any]] | None = None,
) -> tuple[list[TileSampleRecord], dict[str, int]]:
    skip_counts = {"skipped_fully_invalid_tiles": 0, "skipped_low_valid_share_tiles": 0}
    candidates: dict[tuple[int, int, int, int], tuple[TileWindow, set[str]]] = {}
    for source, stride in (
        ("positive_dense", config.positive_stride),
        ("hard_negative_dense", config.hard_negative_stride),
        ("negative_grid", config.negative_stride),
    ):
        for window in generate_window_grid_for_scene(ds.width, ds.height, config.tile_size, stride, scene_id=scene_id):
            key = (window.x, window.y, window.width, window.height)
            existing = candidates.get(key)
            if existing is None:
                candidates[key] = (window, {source})
            else:
                existing[1].add(source)

    classified: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    positive_bboxes: list[tuple[int, int, int, int]] = []
    for key, (window, sources) in candidates.items():
        valid = read_valid_data_mask_with_source(ds, window, mode=config.valid_pixel_mode)
        valid_mask = valid.mask
        mosaic_sources: list[str] = []
        filled_pixel_count = 0
        unfilled_pixel_count = int(valid_mask.size - np.count_nonzero(valid_mask))
        valid_before_mosaic = valid.valid_pixel_share
        mosaic_candidate_neighbors = 0
        mosaic_intersecting_neighbors = 0
        mosaic_actually_used_neighbors: list[str] = []
        mosaic_skipped_non_intersecting_neighbors = 0
        if config.mosaic_enabled and neighbor_datasets:
            mosaic = read_mosaic_window(ds, neighbor_datasets, window, config)
            valid_mask = mosaic.valid_mask
            mosaic_sources = mosaic.source_scenes
            filled_pixel_count = mosaic.filled_pixel_count
            unfilled_pixel_count = mosaic.unfilled_pixel_count
            valid_before_mosaic = mosaic.anchor_valid_pixel_share
            mosaic_candidate_neighbors = mosaic.candidate_neighbors
            mosaic_intersecting_neighbors = mosaic.intersecting_neighbors
            mosaic_actually_used_neighbors = mosaic.actually_used_neighbors
            mosaic_skipped_non_intersecting_neighbors = mosaic.skipped_non_intersecting_neighbors
        final_valid_share = float(np.count_nonzero(valid_mask)) / float(valid_mask.size) if valid_mask.size else 0.0
        if config.drop_fully_invalid_tiles and final_valid_share <= 0.0:
            skip_counts["skipped_fully_invalid_tiles"] += 1
            continue
        mask_result = rasterize_mask_for_window(
            ds,
            annotation_geoms.geometries,
            window,
            all_touched=config.all_touched,
            valid_mask=valid_mask if config.clip_mask_to_valid_data else None,
        )
        if config.exclude_empty_valid_tiles and final_valid_share < config.min_valid_pixel_share:
            skip_counts["skipped_low_valid_share_tiles"] += 1
            continue
        kind, positive_pixels = classify_mask(mask_result.mask, config)
        bbox = positive_pixel_bbox(mask_result.mask, x_offset=window.x, y_offset=window.y)
        if bbox and kind in {"positive", "partial_positive"}:
            positive_bboxes.append(bbox)
        classified[key] = {
            "window": window,
            "sources": sources,
            "kind": kind,
            "positive_pixels": positive_pixels,
            "geometries_intersecting": mask_result.geom_count,
            "valid_pixel_share": mask_result.valid_pixel_share,
            "invalid_pixel_share": 1.0 - mask_result.valid_pixel_share,
            "valid_pixel_share_before_mosaic": valid_before_mosaic,
            "mask_pixels_before_valid_clip": mask_result.raw_positive_pixels,
            "mask_pixels_after_valid_clip": mask_result.clipped_positive_pixels,
            "valid_data_source": valid.source,
            "mosaic_sources": mosaic_sources,
            "mosaic_filled_pixel_count": filled_pixel_count,
            "mosaic_unfilled_pixel_count": unfilled_pixel_count,
            "mosaic_candidate_neighbors": mosaic_candidate_neighbors,
            "mosaic_intersecting_neighbors": mosaic_intersecting_neighbors,
            "mosaic_actually_used_neighbors": mosaic_actually_used_neighbors,
            "mosaic_skipped_non_intersecting_neighbors": mosaic_skipped_non_intersecting_neighbors,
        }

    records: list[TileSampleRecord] = []
    context_px = config.hard_negative_context_px if config.hard_negative_context_px is not None else max(1, config.tile_size // 2)
    for key, item in classified.items():
        window = item["window"]
        sources = item["sources"]
        kind: TileKind = item["kind"]
        source = "negative_grid"
        if kind in {"positive", "partial_positive"}:
            if "positive_dense" not in sources and "negative_grid" not in sources:
                continue
            source = "positive_dense" if "positive_dense" in sources else "negative_grid"
        else:
            if _is_hard_negative((window.x, window.y, window.x + window.width, window.y + window.height), positive_bboxes, context_px):
                kind = "hard_negative"
                if "hard_negative_dense" not in sources and "negative_grid" not in sources:
                    continue
                source = "hard_negative_dense" if "hard_negative_dense" in sources else "negative_grid"
            elif "negative_grid" not in sources:
                continue
        records.append(
            TileSampleRecord(
                scene_id=scene_id,
                image_path=str(image_path),
                x=int(window.x),
                y=int(window.y),
                width=int(window.width),
                height=int(window.height),
                tile_size=int(window.tile_size),
                stride=int(window.stride),
                kind=kind,
                positive_pixels=int(item["positive_pixels"]),
                source=source,
                geometries_intersecting=int(item["geometries_intersecting"]),
                valid_pixel_share=float(item["valid_pixel_share"]),
                invalid_pixel_share=float(item["invalid_pixel_share"]),
                mask_pixels_before_valid_clip=int(item["mask_pixels_before_valid_clip"]),
                mask_pixels_after_valid_clip=int(item["mask_pixels_after_valid_clip"]),
                valid_data_source=str(item["valid_data_source"]),
                mosaic_sources=list(item["mosaic_sources"]),
                mosaic_filled_pixel_count=int(item["mosaic_filled_pixel_count"]),
                mosaic_unfilled_pixel_count=int(item["mosaic_unfilled_pixel_count"]),
                metadata={
                    "sources": sorted(sources),
                    "valid_pixel_share_before_mosaic": item["valid_pixel_share_before_mosaic"],
                    "mosaic_candidate_neighbors": item["mosaic_candidate_neighbors"],
                    "mosaic_intersecting_neighbors": item["mosaic_intersecting_neighbors"],
                    "mosaic_actually_used_neighbors": item["mosaic_actually_used_neighbors"],
                    "mosaic_skipped_non_intersecting_neighbors": item["mosaic_skipped_non_intersecting_neighbors"],
                },
            )
        )
    records.sort(key=lambda item: (_kind_priority(item.kind), item.y, item.x, item.stride))
    return records, skip_counts


def classify_mask(mask: np.ndarray, config: TilePreparationConfig) -> tuple[TileKind, int]:
    positive_pixels = int(np.count_nonzero(mask))
    if positive_pixels >= config.min_positive_pixels:
        return "positive", positive_pixels
    if positive_pixels > 0 and config.include_partial_positive:
        if config.partial_positive_fraction <= 0:
            return "partial_positive", positive_pixels
        area = max(1, int(mask.shape[-1]) * int(mask.shape[-2]))
        if positive_pixels / area >= config.partial_positive_fraction:
            return "partial_positive", positive_pixels
    return "negative", positive_pixels


def _is_hard_negative(
    window_bbox: tuple[int, int, int, int],
    positive_bboxes: list[tuple[int, int, int, int]],
    context_px: int | None,
) -> bool:
    if context_px is None or context_px <= 0 or not positive_bboxes:
        return False
    current = box(*window_bbox)
    for x0, y0, x1, y1 in positive_bboxes:
        if current.intersects(box(x0 - context_px, y0 - context_px, x1 + context_px, y1 + context_px)):
            return True
    return False


def _format_image(rgb: np.ndarray, config: TilePreparationConfig) -> np.ndarray:
    if config.output_format == "hwc_uint8":
        return rgb
    data = rgb.astype("float32") / 255.0 if config.normalize else rgb.astype("float32")
    if config.output_format == "chw_float32":
        return np.transpose(data, (2, 0, 1)).astype("float32")
    if config.output_format == "hwc_float32":
        return data.astype("float32")
    raise ValueError(f"Unsupported output_format: {config.output_format}")


def _limit_records(records: list[TileSampleRecord], limit: int, rng: random.Random) -> list[TileSampleRecord]:
    if len(records) <= limit:
        return records
    positives = [record for record in records if record.kind in {"positive", "partial_positive"}]
    empties = [record for record in records if record.kind in {"negative", "hard_negative"}]
    rng.shuffle(positives)
    rng.shuffle(empties)
    selected = positives[:limit]
    if len(selected) < limit:
        selected.extend(empties[: limit - len(selected)])
    selected.sort(key=lambda item: (_kind_priority(item.kind), item.y, item.x, item.stride))
    return selected


def _kind_priority(kind: str) -> int:
    return {"positive": 0, "partial_positive": 1, "hard_negative": 2, "negative": 3}.get(kind, 9)


def _log_progress(message: str) -> None:
    value = os.getenv("MLSYSTEM_TILE_PREPARATION_PROGRESS", "1").strip().lower()
    if value in {"0", "false", "no", "off"}:
        return
    print(f"[tile_preparation] {message}", flush=True)


def _neighbor_datasets(
    image_path: str,
    datasets: dict[str, Any],
    scene_ids_by_path: dict[str, str],
    config: TilePreparationConfig,
) -> list[tuple[str, Any]]:
    if not config.mosaic_enabled:
        return []
    return [
        (scene_ids_by_path.get(path, path), ds)
        for path, ds in datasets.items()
        if path != image_path
    ]


def _rasterio_path(value: str | Path | None) -> str:
    text = str(value) if value is not None else ""
    if text.startswith("\\vsis3\\"):
        return "/" + text.lstrip("\\").replace("\\", "/")
    return text
