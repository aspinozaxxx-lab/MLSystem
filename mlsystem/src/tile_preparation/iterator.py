from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from .annotations import AnnotationGeometrySet, load_annotation_geometries
from .augmentations import apply_random_training_augmentation
from .config import AnnotationInput, SceneInput, TilePreparationConfig
from .mask_rasterizer import positive_pixel_bbox, rasterize_mask_for_window
from .raster_reader import read_rgb_window
from .records import ReadyTileSample, TileKind, TileRecordBuildResult, TileSampleRecord, TileWindow
from .summary import summarize_tile_records
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
    rng = random.Random(config.seed)

    for scene in scenes:
        image_path = Path(scene.image_path)
        scene_id = scene.resolved_scene_id()
        with rasterio.open(image_path) as ds:
            annotation_geoms = load_annotation_geometries(annotation, raster_crs=ds.crs)
            annotation_summary = annotation_geoms.to_dict(annotation.geojson_path)
            warnings.extend(annotation_geoms.warnings)
            scene_records = _build_scene_records(ds, scene_id, image_path, annotation_geoms, config)
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
                    "records_before_empty_limit": len(before_limit),
                    "records_after_empty_limit": len(scene_records),
                    **scene_summary,
                }
            )

    if config.max_records is not None and len(base_records) > config.max_records:
        base_records = _limit_records(base_records, config.max_records, rng)
    if config.shuffle:
        rng.shuffle(base_records)
    virtual_records = apply_virtual_repeats(base_records, config)
    if config.virtual_epoch_multiplier > 1:
        virtual_records = virtual_records * config.virtual_epoch_multiplier
    return TileRecordBuildResult(
        records=virtual_records,
        base_records=base_records,
        warnings=warnings,
        scene_reports=scene_reports,
        metadata={
            "annotation": annotation_summary,
            "positive_stride": config.positive_stride,
            "hard_negative_stride": config.hard_negative_stride,
            "negative_stride": config.negative_stride,
            "base_record_count": len(base_records),
            "virtual_record_count": len(virtual_records),
            "effective_samples_per_epoch": len(virtual_records),
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
    try:
        for index, record in enumerate(records):
            if config.max_records is not None and index >= config.max_records:
                break
            if not record.image_path:
                continue
            ds = open_datasets.get(record.image_path)
            if ds is None:
                ds = rasterio.open(record.image_path)
                open_datasets[record.image_path] = ds
            annotation_geoms = annotation_cache.get(record.image_path)
            if annotation_geoms is None:
                annotation_geoms = load_annotation_geometries(annotation, raster_crs=ds.crs)
                annotation_cache[record.image_path] = annotation_geoms
            rgb = read_rgb_window(ds, record, bands=config.input_bands)
            mask, geom_count = rasterize_mask_for_window(ds, annotation_geoms.geometries, record, all_touched=config.all_touched)
            metadata: dict[str, Any] = {"geometries_intersecting": geom_count}
            if config.apply_random_augmentations and any(bool(value) for value in (config.augmentations or {}).values()):
                rgb, mask, aug_metadata = apply_random_training_augmentation(
                    rgb,
                    mask,
                    config.augmentations,
                    seed=config.seed + index,
                )
                metadata["augmentation"] = aug_metadata
            image = _format_image(rgb, config)
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
    image_path: Path,
    annotation_geoms: AnnotationGeometrySet,
    config: TilePreparationConfig,
) -> list[TileSampleRecord]:
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
        mask, geom_count = rasterize_mask_for_window(ds, annotation_geoms.geometries, window, all_touched=config.all_touched)
        kind, positive_pixels = classify_mask(mask, config)
        bbox = positive_pixel_bbox(mask, x_offset=window.x, y_offset=window.y)
        if bbox and kind in {"positive", "partial_positive"}:
            positive_bboxes.append(bbox)
        classified[key] = {
            "window": window,
            "sources": sources,
            "kind": kind,
            "positive_pixels": positive_pixels,
            "geometries_intersecting": geom_count,
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
                metadata={"sources": sorted(sources)},
            )
        )
    records.sort(key=lambda item: (_kind_priority(item.kind), item.y, item.x, item.stride))
    return records


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
