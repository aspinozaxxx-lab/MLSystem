from __future__ import annotations

import html
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.features import rasterize
from rasterio.transform import Affine

from .annotations import load_annotation_geometries
from .augmentations import (
    DEFAULT_AUGMENTATION_OPERATIONS,
    TRAINING_AUGMENTATION_KEYS,
    apply_debug_augmentation_with_mask,
    augmentation_catalog,
    resolve_augmentation_operations,
)
from .config import AnnotationInput, SceneInput, TilePreparationConfig
from .iterator import build_tile_records
from .mosaic import read_mosaic_window
from .mask_rasterizer import rasterize_mask_for_window
from .raster_reader import read_band_window, read_rgb_window, to_rgb_uint8
from .records import TileSampleRecord
from .summary import summarize_tile_records
from .validity import read_valid_data_mask_with_source
from .windows import build_tiling_check


RASTER_SUFFIXES = {".tif", ".tiff"}
MASK_VISUALIZATION = {"mode": "dashed_contour", "color": "red", "dash": 8, "gap": 5, "width": 2}


def preview_annotated_tile_report(
    input_dir: str | Path,
    *,
    tile_size: int,
    stride: int,
    positive_stride_factor: float = 1.0,
    hard_negative_stride_factor: float = 1.0,
    negative_stride_factor: float = 1.0,
    min_positive_pixels: int = 1,
    max_scenes: int | None = 1,
    max_records_preview: int = 20,
    include_annotation_summary: bool = True,
    include_augmentation_catalog: bool = False,
    annotation_crs: str | None = "auto",
    allow_inferred_annotation_crs: bool = True,
    anchor_scene: str | None = None,
    annotation_name: str | None = None,
    include_neighbors: bool = False,
    mosaic_enabled: bool = False,
    augmentation_level: int | None = None,
    cutout_mask_mode: str = "erase",
) -> dict[str, Any]:
    scenes, annotation = find_annotated_scenes(
        input_dir,
        anchor_scene=anchor_scene,
        annotation_name=annotation_name,
        include_neighbors=include_neighbors,
        annotation_crs=annotation_crs,
        allow_inferred_annotation_crs=allow_inferred_annotation_crs,
    )
    scene = scenes[0]
    config = TilePreparationConfig(
        tile_size=tile_size,
        stride=stride,
        positive_stride_factor=positive_stride_factor,
        hard_negative_stride_factor=hard_negative_stride_factor,
        negative_stride_factor=negative_stride_factor,
        min_positive_pixels=min_positive_pixels,
        hard_negative_context_px=max(1, int(tile_size) // 2),
        mosaic_enabled=mosaic_enabled,
        cutout_mask_mode=cutout_mask_mode,
        augmentation_level=augmentation_level,
    )
    result = build_tile_records(scenes, annotation, config)
    with rasterio.open(scene.image_path) as ds:
        raster = _raster_metadata(ds, scene.image_path)
    base_summary = summarize_tile_records(result.base_records)
    response: dict[str, Any] = {
        "status": "ok",
        "facade_available": True,
        "recommended_entrypoint": "TilePreparationFacade",
        "augmentation_level": augmentation_level,
        "cutout_mask_mode": config.cutout_mask_mode,
        "mask_visualization": dict(MASK_VISUALIZATION),
        "valid_data_clipping": {
            "enabled": bool(config.clip_mask_to_valid_data),
            "valid_pixel_mode": config.valid_pixel_mode,
        },
        "mosaic": {
            "enabled": bool(config.mosaic_enabled),
            "scene_count": len(scenes),
            "scenes": [item.resolved_scene_id() for item in scenes],
        },
        "raster": raster,
        "tiling_summary": _tiling_checks(raster["width"], raster["height"], config),
        "classification_summary": {
            **base_summary,
            "virtual_records_after_repeats": len(result.records),
            "effective_samples_per_epoch": len(result.records),
        },
        "preview_records": [record.to_dict() for record in result.base_records[: max(0, int(max_records_preview))]],
        "warnings": result.warnings,
    }
    if include_annotation_summary:
        response["annotation"] = result.metadata.get("annotation")
    if include_augmentation_catalog:
        response["augmentation_operations_supported"] = list(DEFAULT_AUGMENTATION_OPERATIONS)
        response["training_augmentation_keys_supported"] = list(TRAINING_AUGMENTATION_KEYS)
        response["augmentation_catalog"] = augmentation_catalog()
    if max_scenes is not None:
        response["scene_count"] = min(1, max(0, int(max_scenes)))
    return response


def generate_annotated_tile_report(
    input_dir: str | Path,
    *,
    config: TilePreparationConfig,
    scenes: list[SceneInput] | None = None,
    annotation: AnnotationInput | None = None,
    output_dir: str | Path | None = None,
    annotation_crs: str | None = "auto",
    allow_inferred_annotation_crs: bool = True,
    max_overview_size: int = 1600,
    max_tile_examples: int = 32,
    max_augmentation_tiles: int = 8,
    augmentation_mode: str = "all",
    augmentation_seed: int = 42,
) -> dict[str, Any]:
    root = Path(output_dir) if output_dir is not None else Path(input_dir)
    input_root = Path(input_dir)
    if scenes is None or annotation is None:
        scene, annotation = find_single_annotated_scene(
            input_root,
            annotation_crs=annotation_crs,
            allow_inferred_annotation_crs=allow_inferred_annotation_crs,
        )
        scenes = [scene]
    else:
        scene = next((item for item in scenes if not bool(item.metadata.get("mosaic_neighbor_only"))), scenes[0])
    scene_dir = root / scene.resolved_scene_id()
    scene_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = scene_dir / "annotated_tiles"
    augmentations_dir = scene_dir / "annotated_augmentations"
    previews_dir.mkdir(parents=True, exist_ok=True)
    augmentations_dir.mkdir(parents=True, exist_ok=True)
    _clear_pngs(previews_dir)
    _clear_pngs(augmentations_dir)

    result = build_tile_records(scenes, annotation, config)
    with rasterio.open(scene.image_path) as ds:
        annotation_geoms = load_annotation_geometries(annotation, raster_crs=ds.crs)
        raster = _raster_metadata(ds, scene.image_path)
        overviews = _write_overviews(ds, annotation_geoms.geometries, result.base_records, scene_dir, config, max_overview_size)
        examples = _select_examples(result.base_records, max_tile_examples)
        tile_examples = _write_tile_examples(
            ds,
            annotation_geoms.geometries,
            examples,
            previews_dir,
            augmentations_dir,
            config,
            scenes=scenes,
            max_augmentation_tiles=max_augmentation_tiles,
            augmentation_mode=augmentation_mode,
            augmentation_seed=augmentation_seed,
        )

    classification_summary = {
        **summarize_tile_records(result.base_records),
        "total_base_records": len(result.base_records),
        "virtual_records_after_repeats": len(result.records),
        "effective_samples_per_epoch": len(result.records),
    }
    augmentation_report = _build_augmentation_report(tile_examples, augmentation_mode)
    summary = {
        "scene": scene.resolved_scene_id(),
        "image_path": str(scene.image_path),
        "annotation_path": str(annotation.geojson_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "facade": {
            "available": True,
            "recommended_entrypoint": "TilePreparationFacade",
            "augmentation_level": config.augmentation_level,
        },
        "mask_visualization": dict(MASK_VISUALIZATION),
        "valid_data_clipping": {
            "enabled": bool(config.clip_mask_to_valid_data),
            "valid_pixel_mode": config.valid_pixel_mode,
            "min_valid_pixel_share": config.min_valid_pixel_share,
            "exclude_empty_valid_tiles": config.exclude_empty_valid_tiles,
        },
        "mosaic": {
            "enabled": bool(config.mosaic_enabled),
            "fill_nodata": bool(config.mosaic_fill_nodata),
            "resampling": config.mosaic_resampling,
            "scene_count": len(scenes),
            "scenes": [item.resolved_scene_id() for item in scenes],
        },
        "raster": raster,
        "annotation": result.metadata.get("annotation"),
        "config": _config_to_dict(config, augmentation_mode, augmentation_seed, max_tile_examples, max_augmentation_tiles),
        "tiling_checks": _tiling_checks(raster["width"], raster["height"], config),
        "classification_summary": classification_summary,
        "virtual_sampling_summary": {
            "positive_repeat_factor": config.positive_repeat_factor,
            "hard_negative_repeat_factor": config.hard_negative_repeat_factor,
            "negative_repeat_factor": config.negative_repeat_factor,
            "virtual_epoch_multiplier": config.virtual_epoch_multiplier,
            "base_records": len(result.base_records),
            "virtual_records": len(result.records),
            "effective_samples_per_epoch": len(result.records),
        },
        "overview_images": overviews,
        "tile_examples": tile_examples,
        "augmentation_report": augmentation_report,
        "warnings": result.warnings,
    }
    summary_path = scene_dir / "annotated_scene_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    report_path = scene_dir / "annotated_tile_sampling_report.html"
    report_path.write_text(_render_html(summary), encoding="utf-8")
    index = {
        "status": "ok",
        "input_dir": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_count": 1,
        "scenes": [
            {
                "scene": summary["scene"],
                "image_path": summary["image_path"],
                "annotation_path": summary["annotation_path"],
                "report_path": str(report_path),
                "summary_path": str(summary_path),
                "width": raster["width"],
                "height": raster["height"],
                "bands": raster["bands"],
                "crs": raster["crs"],
                **classification_summary,
                "augmentation_checks": augmentation_report["checks_summary"],
                "valid_data_clipping": summary["valid_data_clipping"],
                "mosaic": summary["mosaic"],
            }
        ],
        "warnings": result.warnings,
    }
    (root / "annotated_tile_sampling_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (root / "annotated_tile_sampling_index.html").write_text(_render_index(index), encoding="utf-8")
    return index


def find_single_annotated_scene(
    input_dir: str | Path,
    *,
    annotation_crs: str | None = "auto",
    allow_inferred_annotation_crs: bool = True,
) -> tuple[SceneInput, AnnotationInput]:
    root = Path(input_dir)
    rasters = sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES])
    if len(rasters) != 1:
        raise ValueError(f"expected exactly one GeoTIFF in {root}, found {len(rasters)}")
    geojsons = sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".geojson"])
    if not geojsons:
        raise ValueError(f"expected one GeoJSON annotation next to {rasters[0].name}, found none")
    same_stem = [path for path in geojsons if path.stem == rasters[0].stem]
    if len(geojsons) == 1:
        annotation_path = geojsons[0]
    elif len(same_stem) == 1:
        annotation_path = same_stem[0]
    else:
        raise ValueError(f"multiple GeoJSON files found in {root}; keep one or use the same stem as the raster")
    return (
        SceneInput(image_path=rasters[0], scene_id=rasters[0].stem),
        AnnotationInput(
            geojson_path=annotation_path,
            annotation_crs=annotation_crs,
            allow_inferred_annotation_crs=allow_inferred_annotation_crs,
        ),
    )


def find_annotated_scenes(
    input_dir: str | Path,
    *,
    anchor_scene: str | None = None,
    annotation_name: str | None = None,
    include_neighbors: bool = False,
    annotation_crs: str | None = "auto",
    allow_inferred_annotation_crs: bool = True,
) -> tuple[list[SceneInput], AnnotationInput]:
    root = Path(input_dir)
    rasters = sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES])
    if not rasters:
        raise ValueError(f"expected at least one GeoTIFF in {root}")
    if anchor_scene:
        anchor_candidates = [path for path in rasters if path.name == anchor_scene or path.stem == Path(anchor_scene).stem]
        if len(anchor_candidates) != 1:
            raise ValueError(f"expected exactly one anchor scene matching {anchor_scene!r}, found {len(anchor_candidates)}")
        anchor = anchor_candidates[0]
    elif len(rasters) == 1:
        anchor = rasters[0]
    else:
        raise ValueError(f"multiple GeoTIFF files found in {root}; pass --anchor-scene")

    geojsons = sorted([path for path in root.iterdir() if path.is_file() and path.suffix.lower() == ".geojson"])
    if annotation_name:
        matches = [path for path in geojsons if path.name == annotation_name or path.stem == Path(annotation_name).stem]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one annotation matching {annotation_name!r}, found {len(matches)}")
        annotation_path = matches[0]
    elif len(geojsons) == 1:
        annotation_path = geojsons[0]
    else:
        raise ValueError(f"expected one GeoJSON annotation in {root}, found {len(geojsons)}")

    scenes = [SceneInput(image_path=anchor, scene_id=anchor.stem)]
    if include_neighbors:
        for path in rasters:
            if path == anchor:
                continue
            scenes.append(SceneInput(image_path=path, scene_id=path.stem, metadata={"mosaic_neighbor_only": True}))
    return (
        scenes,
        AnnotationInput(
            geojson_path=annotation_path,
            annotation_crs=annotation_crs,
            allow_inferred_annotation_crs=allow_inferred_annotation_crs,
        ),
    )


def _write_overviews(
    ds: Any,
    geometries: list[Any],
    records: list[TileSampleRecord],
    scene_dir: Path,
    config: TilePreparationConfig,
    max_size: int,
) -> dict[str, str]:
    overview = _read_overview(ds, max_size)
    raw_mask = _overview_mask(ds, geometries, overview.shape[1], overview.shape[0], config.all_touched)
    valid_mask = _overview_valid_mask(ds, overview.shape[1], overview.shape[0], config)
    clipped_mask = (raw_mask & valid_mask).astype("uint8") if config.clip_mask_to_valid_data else raw_mask
    overlay = overlay_mask_contour(
        overlay_mask_contour(overview, raw_mask, color=(255, 230, 0), width=max(2, int(round(max(overview.shape[:2]) / 900))), shadow=True),
        clipped_mask,
        width=max(2, int(round(max(overview.shape[:2]) / 900))),
    )
    base = Image.fromarray(overlay, mode="RGB").convert("RGBA")
    scale_x = overview.shape[1] / float(ds.width)
    scale_y = overview.shape[0] / float(ds.height)
    draw = ImageDraw.Draw(base)
    for record in records:
        color = _kind_color(record.kind)
        draw.rectangle(
            (
                int(record.x * scale_x),
                int(record.y * scale_y),
                int((record.x + record.width) * scale_x),
                int((record.y + record.height) * scale_y),
            ),
            outline=color,
            width=1,
        )
    outputs = {
        "overview_raster_grid_mask.png": "overview_raster_grid_mask.png",
        "overview_mask_only.png": "overview_mask_only.png",
        "overview_valid_mask.png": "overview_valid_mask.png",
        "overview_raw_annotation_mask.png": "overview_raw_annotation_mask.png",
        "overview_grid_positive_negative.png": "overview_grid_positive_negative.png",
    }
    base.convert("RGB").save(scene_dir / "overview_raster_grid_mask.png")
    Image.fromarray(_mask_only_contour(clipped_mask), mode="RGB").save(scene_dir / "overview_mask_only.png")
    Image.fromarray((valid_mask * 255).astype("uint8"), mode="L").save(scene_dir / "overview_valid_mask.png")
    Image.fromarray(_mask_only_contour(raw_mask), mode="RGB").save(scene_dir / "overview_raw_annotation_mask.png")
    Image.fromarray(overview, mode="RGB").convert("RGBA").save(scene_dir / "overview_grid_positive_negative.png")
    grid = Image.open(scene_dir / "overview_grid_positive_negative.png").convert("RGBA")
    draw = ImageDraw.Draw(grid)
    for record in records:
        color = _kind_color(record.kind)
        draw.rectangle(
            (
                int(record.x * scale_x),
                int(record.y * scale_y),
                int((record.x + record.width) * scale_x),
                int((record.y + record.height) * scale_y),
            ),
            outline=color,
            width=1,
        )
    grid.convert("RGB").save(scene_dir / "overview_grid_positive_negative.png")
    return outputs


def _write_tile_examples(
    ds: Any,
    geometries: list[Any],
    records: list[TileSampleRecord],
    previews_dir: Path,
    augmentations_dir: Path,
    config: TilePreparationConfig,
    *,
    scenes: list[SceneInput],
    max_augmentation_tiles: int,
    augmentation_mode: str,
    augmentation_seed: int,
) -> list[dict[str, Any]]:
    operations = resolve_augmentation_operations(augmentation_mode)
    rows: list[dict[str, Any]] = []
    augmented_positive_tiles = 0
    neighbor_handles: list[tuple[str, Any]] = []
    for scene in scenes:
        if scene.resolved_scene_id() == (records[0].scene_id if records else ""):
            continue
        try:
            neighbor_handles.append((scene.resolved_scene_id(), rasterio.open(scene.image_path)))
        except Exception:  # noqa: BLE001
            continue
    for index, record in enumerate(records):
        rgb_before = read_rgb_window(ds, record, bands=config.input_bands)
        valid = read_valid_data_mask_with_source(ds, record, mode=config.valid_pixel_mode)
        valid_mask = valid.mask
        mosaic_info: dict[str, Any] = {}
        source_map = None
        if config.mosaic_enabled:
            mosaic = read_mosaic_window(ds, neighbor_handles, record, config)
            rgb = to_rgb_uint8(mosaic.image)
            valid_mask = mosaic.valid_mask
            source_map = mosaic.source_map
            mosaic_info = {
                "mosaic_sources": mosaic.source_scenes,
                "mosaic_filled_pixel_count": mosaic.filled_pixel_count,
                "mosaic_unfilled_pixel_count": mosaic.unfilled_pixel_count,
                "valid_pixel_share_before_mosaic": mosaic.anchor_valid_pixel_share,
                "valid_pixel_share_after_mosaic": mosaic.final_valid_pixel_share,
                "mosaic_warnings": mosaic.warnings,
            }
        else:
            rgb = rgb_before
            mosaic_info = {
                "mosaic_sources": [],
                "mosaic_filled_pixel_count": 0,
                "mosaic_unfilled_pixel_count": int(valid_mask.size - np.count_nonzero(valid_mask)),
                "valid_pixel_share_before_mosaic": valid.valid_pixel_share,
                "valid_pixel_share_after_mosaic": valid.valid_pixel_share,
                "mosaic_warnings": [],
            }
        mask_result = rasterize_mask_for_window(
            ds,
            geometries,
            record,
            all_touched=config.all_touched,
            valid_mask=valid_mask if config.clip_mask_to_valid_data else None,
        )
        mask = mask_result.mask
        stem = f"tile_{index:03d}_{record.kind}"
        original_name = f"{stem}_rgb.png"
        before_name = f"{stem}_rgb_before_mosaic.png"
        mask_name = f"{stem}_mask.png"
        raw_mask_name = f"{stem}_raw_annotation_mask.png"
        valid_mask_name = f"{stem}_valid_mask.png"
        overlay_name = f"{stem}_overlay.png"
        raw_overlay_name = f"{stem}_raw_yellow_clipped_red_overlay.png"
        Image.fromarray(rgb, mode="RGB").save(previews_dir / original_name)
        Image.fromarray(rgb_before, mode="RGB").save(previews_dir / before_name)
        Image.fromarray((mask * 255).astype("uint8"), mode="L").save(previews_dir / mask_name)
        Image.fromarray((mask_result.raw_mask * 255).astype("uint8"), mode="L").save(previews_dir / raw_mask_name)
        Image.fromarray((valid_mask * 255).astype("uint8"), mode="L").save(previews_dir / valid_mask_name)
        Image.fromarray(overlay_mask_contour(rgb, mask), mode="RGB").save(previews_dir / overlay_name)
        raw_then_clipped = overlay_mask_contour(rgb, mask_result.raw_mask, color=(255, 230, 0), shadow=True)
        raw_then_clipped = overlay_mask_contour(raw_then_clipped, mask, color=(255, 0, 0), shadow=True)
        Image.fromarray(raw_then_clipped, mode="RGB").save(previews_dir / raw_overlay_name)
        source_map_path = None
        if source_map is not None:
            source_map_name = f"{stem}_source_map.png"
            Image.fromarray(_source_map_rgb(source_map), mode="RGB").save(previews_dir / source_map_name)
            source_map_path = f"annotated_tiles/{source_map_name}"
        row = {
            **record.to_dict(),
            "geometries_intersecting": mask_result.geom_count,
            "positive_pixels_runtime": int(mask.sum()),
            "raw_positive_pixels_runtime": int(mask_result.raw_positive_pixels),
            "clipped_positive_pixels_runtime": int(mask_result.clipped_positive_pixels),
            "valid_pixel_share_runtime": mask_result.valid_pixel_share,
            "valid_data_source_runtime": valid.source,
            "preview_rgb": f"annotated_tiles/{original_name}",
            "preview_rgb_before_mosaic": f"annotated_tiles/{before_name}",
            "preview_mask": f"annotated_tiles/{mask_name}",
            "preview_raw_mask": f"annotated_tiles/{raw_mask_name}",
            "preview_valid_mask": f"annotated_tiles/{valid_mask_name}",
            "preview_overlay": f"annotated_tiles/{overlay_name}",
            "preview_raw_yellow_clipped_red_overlay": f"annotated_tiles/{raw_overlay_name}",
            "preview_source_map": source_map_path,
            **mosaic_info,
            "augmentations": [],
        }
        should_write_aug = int(mask.sum()) > 0 and augmented_positive_tiles < max_augmentation_tiles
        if should_write_aug:
            augmented_positive_tiles += 1
            for aug_index, operation in enumerate(operations):
                seed = augmentation_seed + index * 1000 + aug_index
                aug_rgb, aug_mask, metadata = apply_debug_augmentation_with_mask(
                    rgb,
                    operation,
                    seed=seed,
                    mask=mask,
                    cutout_mask_mode=config.cutout_mask_mode,
                )
                aug_stem = f"tile_{index:03d}_{operation}"
                rgb_name = f"{aug_stem}_rgb.png"
                mask_aug_name = f"{aug_stem}_mask.png"
                overlay_aug_name = f"{aug_stem}_overlay.png"
                mask_before_name = f"{aug_stem}_mask_before.png"
                overlay_before_name = f"{aug_stem}_overlay_before.png"
                Image.fromarray(aug_rgb, mode="RGB").save(augmentations_dir / rgb_name)
                after_mask = aug_mask if aug_mask is not None else mask
                Image.fromarray((mask * 255).astype("uint8"), mode="L").save(augmentations_dir / mask_before_name)
                Image.fromarray((after_mask * 255).astype("uint8"), mode="L").save(augmentations_dir / mask_aug_name)
                Image.fromarray(overlay_mask_contour(rgb, mask), mode="RGB").save(augmentations_dir / overlay_before_name)
                Image.fromarray(overlay_mask_contour(aug_rgb, after_mask), mode="RGB").save(augmentations_dir / overlay_aug_name)
                row["augmentations"].append(
                    {
                        "rgb_path": f"annotated_augmentations/{rgb_name}",
                        "mask_path": f"annotated_augmentations/{mask_aug_name}",
                        "mask_after_path": f"annotated_augmentations/{mask_aug_name}",
                        "overlay_path": f"annotated_augmentations/{overlay_aug_name}",
                        "overlay_after_path": f"annotated_augmentations/{overlay_aug_name}",
                        "mask_before_path": f"annotated_augmentations/{mask_before_name}",
                        "overlay_before_path": f"annotated_augmentations/{overlay_before_name}",
                        **metadata,
                    }
                )
        rows.append(row)
    for _scene_id, handle in neighbor_handles:
        handle.close()
    return rows


def _select_examples(records: list[TileSampleRecord], max_count: int) -> list[TileSampleRecord]:
    selected: list[TileSampleRecord] = []
    seen: set[str] = set()
    for kind in ("positive", "partial_positive", "hard_negative", "negative"):
        for record in records:
            if record.kind == kind and record.record_id not in seen:
                selected.append(record)
                seen.add(record.record_id)
                break
    edge = [record for record in records if record.x == 0 or record.y == 0]
    center = sorted(records, key=lambda item: (item.x - 0.5 * max(1, item.x + item.width)) ** 2 + (item.y - 0.5 * max(1, item.y + item.height)) ** 2)
    for bucket in (edge, center, records):
        for record in bucket:
            if record.record_id in seen:
                continue
            selected.append(record)
            seen.add(record.record_id)
            if len(selected) >= max_count:
                return selected
    return selected[:max_count]


def _build_augmentation_report(tile_examples: list[dict[str, Any]], augmentation_mode: str) -> dict[str, Any]:
    operations = resolve_augmentation_operations(augmentation_mode)
    by_operation: dict[str, list[dict[str, Any]]] = {operation: [] for operation in operations}
    status_counts = {"pass": 0, "warning": 0, "failed": 0}
    for tile in tile_examples:
        for aug in tile.get("augmentations") or []:
            status = str(aug.get("check_status") or "failed")
            status_counts[status] = status_counts.get(status, 0) + 1
            by_operation.setdefault(str(aug.get("operation")), []).append(
                {
                    "tile_id": tile.get("tile_id"),
                    "overlay_path": aug.get("overlay_path"),
                    "seed": aug.get("seed"),
                    "check_status": status,
                    "changed_pixels_fraction": aug.get("changed_pixels_fraction"),
                    "mask_alignment_check": aug.get("mask_alignment_check"),
                    "mask_erased_pixels": aug.get("mask_erased_pixels"),
                    "cutout_intersected_positive": aug.get("cutout_intersected_positive"),
                    "checks": aug.get("checks"),
                    "mask_checks": aug.get("mask_checks"),
                }
            )
    operation_checks: list[dict[str, Any]] = []
    for item in augmentation_catalog(operations):
        examples = by_operation.get(item["operation"]) or []
        statuses = [example.get("check_status") for example in examples]
        status = "failed" if "failed" in statuses else ("warning" if "warning" in statuses else "pass")
        if not examples:
            status = "not_run"
        operation_checks.append(
            {
                **item,
                "status": status,
                "examples": examples[:10],
                "changed_pixels_fraction_mean": round(
                    sum(float(example.get("changed_pixels_fraction") or 0.0) for example in examples) / len(examples),
                    6,
                )
                if examples
                else None,
            }
        )
    return {
        "operations": operations,
        "checks_summary": {
            "passed": int(status_counts.get("pass", 0)),
            "warnings": int(status_counts.get("warning", 0)),
            "failed": int(status_counts.get("failed", 0)),
        },
        "operation_checks": operation_checks,
        "cutout_mask_behavior": "erase",
    }


def _render_html(summary: dict[str, Any]) -> str:
    tiling_rows = "\n".join(
        f"<tr><td>{c['stride_type']}</td><td>{c['effective_stride']}</td><td>{c['expected_total']}</td><td>{c['actual_total']}</td><td>{c['coverage_x']}/{c['coverage_y']}</td><td>{c['out_of_bounds_windows']}</td><td>{'PASS' if c['pass'] else 'FAIL'}</td></tr>"
        for c in summary["tiling_checks"]
    )
    class_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary["classification_summary"].items()
    )
    overview_html = "".join(f'<figure><img src="{path}"><figcaption>{name}</figcaption></figure>' for name, path in summary["overview_images"].items())
    tile_html = "".join(
        "<figure>"
        f"<img src=\"{tile['preview_overlay']}\">"
        f"<figcaption>{html.escape(tile['kind'])}: {html.escape(tile['tile_id'])}, clipped={tile['clipped_positive_pixels_runtime']}, raw={tile['raw_positive_pixels_runtime']}, valid={float(tile['valid_pixel_share_runtime']):.3f}, filled={tile.get('mosaic_filled_pixel_count', 0)}</figcaption>"
        "</figure>"
        for tile in summary["tile_examples"]
    )
    tile_diag_html = "".join(
        "<div class=\"diag\">"
        f"<figure><img src=\"{tile['preview_rgb_before_mosaic']}\"><figcaption>anchor before mosaic</figcaption></figure>"
        f"<figure><img src=\"{tile['preview_rgb']}\"><figcaption>training RGB after mosaic</figcaption></figure>"
        f"<figure><img src=\"{tile['preview_valid_mask']}\"><figcaption>valid data mask</figcaption></figure>"
        f"<figure><img src=\"{tile['preview_raw_mask']}\"><figcaption>raw annotation mask</figcaption></figure>"
        f"<figure><img src=\"{tile['preview_mask']}\"><figcaption>clipped training mask</figcaption></figure>"
        f"<figure><img src=\"{tile['preview_raw_yellow_clipped_red_overlay']}\"><figcaption>yellow raw contour, red clipped contour</figcaption></figure>"
        + (f"<figure><img src=\"{tile['preview_source_map']}\"><figcaption>mosaic source map</figcaption></figure>" if tile.get("preview_source_map") else "")
        + f"<pre>{html.escape(json.dumps({key: tile.get(key) for key in ['tile_id', 'kind', 'raw_positive_pixels_runtime', 'clipped_positive_pixels_runtime', 'valid_pixel_share_runtime', 'mosaic_sources', 'mosaic_filled_pixel_count', 'mosaic_unfilled_pixel_count']}, ensure_ascii=False, indent=2, default=str))}</pre>"
        "</div>"
        for tile in summary["tile_examples"][:8]
    )
    aug_rows = "\n".join(
        f"<tr><td>{item['operation']}</td><td>{item['training_key']}</td><td>{item['group']}</td><td>{html.escape(json.dumps(item.get('parameters') or {}, ensure_ascii=False))}</td><td>{item['status']}</td><td>{item.get('changed_pixels_fraction_mean')}</td></tr>"
        for item in summary["augmentation_report"]["operation_checks"]
    )
    check_rows = "\n".join(
        f"<tr><td>{aug['operation']}</td><td>{aug.get('training_key')}</td><td>{aug.get('checks', {}).get('shape_preserved')}</td><td>{aug.get('mask_checks', {}).get('mask_shape_preserved')}</td><td>{aug.get('mask_checks', {}).get('mask_binary')}</td><td>{aug.get('checks', {}).get('range_0_255')}</td><td>{aug.get('mask_alignment_check')}</td><td>{aug.get('changed_pixels_fraction')}</td><td>{aug.get('mask_erased_pixels')}</td><td>{aug.get('check_status')}</td></tr>"
        for tile in summary["tile_examples"]
        for aug in tile.get("augmentations", [])
    )
    cutout_rows = "\n".join(
        f"<tr><td>{html.escape(tile['tile_id'])}</td><td>{html.escape(aug['operation'])}</td><td>{html.escape(json.dumps(aug.get('cutout_boxes') or aug.get('parameters', {}).get('cutout_boxes') or [], ensure_ascii=False))}</td><td><a href=\"{aug.get('mask_before_path')}\">mask before</a></td><td><a href=\"{aug.get('mask_after_path')}\">mask after</a></td><td>{aug.get('mask_erased_pixels')}</td><td>{aug.get('check_status')}</td></tr>"
        for tile in summary["tile_examples"]
        for aug in tile.get("augmentations", [])
        if aug.get("operation") in {"cutout_small", "cutout_medium", "coarse_dropout"}
        or (str(aug.get("operation", "")).startswith("training_random_all_enabled") and aug.get("cutout_applied"))
    )
    aug_gallery = "".join(
        "<figure>"
        f"<img src=\"{aug['overlay_path']}\">"
        f"<figcaption>{tile['tile_id']} | {aug['operation']} | {aug.get('check_status')} | mask={aug.get('mask_alignment_check')}</figcaption>"
        "</figure>"
        for tile in summary["tile_examples"]
        for aug in tile.get("augmentations", [])
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Annotated tile sampling report - {html.escape(summary['scene'])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d0d7de; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .diag {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; border: 1px solid #d0d7de; padding: 10px; margin: 12px 0; }}
    .warning {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; margin-top: 4px; color: #4b5563; }}
  </style>
</head>
<body>
  <h1>{html.escape(summary['scene'])}</h1>
  <p><b>Image:</b> {html.escape(summary['image_path'])}</p>
  <p><b>GeoJSON:</b> {html.escape(summary['annotation_path'])}</p>
  <p><b>Generated:</b> {html.escape(summary['generated_at'])}</p>
  <p><b>Git commit:</b> {html.escape(str(summary.get('git_commit')))}</p>
  <h2>Raster metadata</h2>
  <pre>{html.escape(json.dumps(summary['raster'], ensure_ascii=False, indent=2, default=str))}</pre>
  <h2>Annotation metadata</h2>
  <pre>{html.escape(json.dumps(summary['annotation'], ensure_ascii=False, indent=2, default=str))}</pre>
  <h2>Parameters</h2>
  <pre>{html.escape(json.dumps(summary['config'], ensure_ascii=False, indent=2, default=str))}</pre>
  <h2>Valid data clipping</h2>
  <p>Red dashed contour is the clipped training mask. Yellow dashed contour is the raw annotation mask before valid-data clipping.</p>
  <pre>{html.escape(json.dumps(summary['valid_data_clipping'], ensure_ascii=False, indent=2, default=str))}</pre>
  <h2>Mosaic fill</h2>
  <p>When enabled, invalid anchor pixels are filled from neighboring scenes before the final training mask is clipped to the union valid mask.</p>
  <pre>{html.escape(json.dumps(summary['mosaic'], ensure_ascii=False, indent=2, default=str))}</pre>
  <h2>Scene-level mask overview</h2>
  <p><b>Legend:</b> red dashed contour = clipped training mask; yellow dashed contour = raw annotation mask before valid clipping; grid colors: red = positive, orange = partial positive, blue = hard negative, gray = negative.</p>
  <section class="gallery">{overview_html}</section>
  <h2>Tiling summary</h2>
  <table><tr><th>stride type</th><th>effective stride</th><th>expected</th><th>actual</th><th>coverage</th><th>out of bounds</th><th>status</th></tr>{tiling_rows}</table>
  <h2>Training tile classification summary</h2>
  <table><tr><th>metric</th><th>value</th></tr>{class_rows}</table>
  <h2>Positive / negative examples</h2>
  <section class="gallery">{tile_html}</section>
  <h2>Valid clipping / mosaic diagnostics</h2>
  {tile_diag_html}
  <h2>Augmentation examples with masks</h2>
  <section class="gallery">{aug_gallery}</section>
  <h2>Cutout / dropout mask behavior</h2>
  <p>Для cutout и coarse dropout режим маски: <b>{html.escape(str(summary['config'].get('cutout_mask_mode', 'erase')))}</b>. Если чёрный вырез попадает внутрь объекта, соответствующие пиксели training mask удаляются, а красный пунктирный контур после аугментации показывает внешнюю границу объекта и внутреннюю границу вырезанной дырки.</p>
  <table><tr><th>tile</th><th>operation</th><th>cutout boxes</th><th>mask before</th><th>mask after</th><th>erased pixels</th><th>status</th></tr>{cutout_rows}</table>
  <h2>Augmentation checks</h2>
  <table><tr><th>operation</th><th>training_key</th><th>image shape</th><th>mask shape</th><th>mask binary</th><th>image range</th><th>mask alignment</th><th>changed pixels</th><th>mask erased pixels</th><th>status</th></tr>{check_rows}</table>
  <h2>Augmentation operation summary</h2>
  <table><tr><th>operation</th><th>training_key</th><th>group</th><th>parameters</th><th>status</th><th>changed mean</th></tr>{aug_rows}</table>
  <h2>Warnings</h2>
  <pre class="warning">{html.escape(json.dumps(summary.get('warnings') or [], ensure_ascii=False, indent=2))}</pre>
  <p><a href="annotated_scene_summary.json">annotated_scene_summary.json</a></p>
</body>
</html>
"""


def _render_index(index: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(scene['scene'])}</td><td>{scene['width']}x{scene['height']}</td><td>{scene['positive_tiles']}</td><td>{scene['hard_negative_tiles']}</td><td>{scene['negative_tiles']}</td><td><a href=\"{html.escape(Path(scene['report_path']).parent.name)}/annotated_tile_sampling_report.html\">report</a></td></tr>"
        for scene in index["scenes"]
    )
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Annotated tile sampling index</title></head>
<body>
<h1>Annotated tile sampling index</h1>
<p>{html.escape(index['input_dir'])}</p>
<table border="1" cellspacing="0" cellpadding="6">
<tr><th>scene</th><th>size</th><th>positive</th><th>hard_negative</th><th>negative</th><th>report</th></tr>
{rows}
</table>
<p><a href="annotated_tile_sampling_index.json">annotated_tile_sampling_index.json</a></p>
</body></html>
"""


def _raster_metadata(ds: Any, path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "filename": path.name,
        "width": int(ds.width),
        "height": int(ds.height),
        "bands": int(ds.count),
        "dtype": ",".join(str(item) for item in ds.dtypes),
        "crs": str(ds.crs) if ds.crs else None,
        "transform": list(ds.transform)[:6],
        "bounds": [float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top)],
        "nodata": ds.nodata,
    }


def _tiling_checks(width: int, height: int, config: TilePreparationConfig) -> list[dict[str, Any]]:
    return [
        build_tiling_check(width, height, config.tile_size, config.stride, label="base", stride_factor=1.0),
        build_tiling_check(width, height, config.tile_size, config.stride, label="positive_dense", stride_factor=config.positive_stride_factor),
        build_tiling_check(width, height, config.tile_size, config.stride, label="hard_negative_dense", stride_factor=config.hard_negative_stride_factor),
        build_tiling_check(width, height, config.tile_size, config.stride, label="negative", stride_factor=config.negative_stride_factor),
    ]


def _read_overview(ds: Any, max_size: int) -> np.ndarray:
    scale = min(float(max_size) / max(ds.width, ds.height), 1.0)
    out_width = max(1, int(ds.width * scale))
    out_height = max(1, int(ds.height * scale))
    arr = ds.read(list(range(1, min(3, ds.count) + 1)), out_shape=(min(3, ds.count), out_height, out_width), boundless=True, fill_value=0)
    return to_rgb_uint8(arr)


def _overview_mask(ds: Any, geometries: list[Any], out_width: int, out_height: int, all_touched: bool) -> np.ndarray:
    if not geometries:
        return np.zeros((out_height, out_width), dtype="uint8")
    transform = ds.transform * Affine.scale(ds.width / out_width, ds.height / out_height)
    return rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=(out_height, out_width),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    ).astype("uint8")


def _overview_valid_mask(ds: Any, out_width: int, out_height: int, config: TilePreparationConfig) -> np.ndarray:
    bands = [int(band) for band in (config.input_bands or list(range(1, int(ds.count) + 1))) if 1 <= int(band) <= int(ds.count)]
    if not bands:
        return np.zeros((out_height, out_width), dtype="uint8")
    arr = ds.read(
        bands,
        out_shape=(len(bands), out_height, out_width),
        boundless=True,
        fill_value=0,
    )
    mode = str(config.valid_pixel_mode or "auto").lower()
    if mode == "nonzero_all":
        return np.all(arr != 0, axis=0).astype("uint8")
    return np.any(arr != 0, axis=0).astype("uint8")


def _source_map_rgb(source_map: np.ndarray) -> np.ndarray:
    colors = np.array(
        [
            [0, 0, 0],
            [80, 180, 80],
            [0, 140, 255],
            [255, 180, 0],
            [180, 80, 255],
        ],
        dtype="uint8",
    )
    indices = np.asarray(source_map, dtype="int64") % len(colors)
    return colors[indices]


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not binary.any():
        return np.zeros_like(binary, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(binary, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]
    return binary & ~eroded


def dilate_binary(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    binary = np.asarray(mask) > 0
    radius = max(0, int(radius))
    if radius == 0 or not binary.any():
        return binary
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    out = np.zeros_like(binary, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]
    return out


def dashed_boundary(boundary: np.ndarray, dash: int = 8, gap: int = 5) -> np.ndarray:
    binary = np.asarray(boundary) > 0
    if not binary.any():
        return binary
    yy, xx = np.indices(binary.shape)
    period = max(1, int(dash) + max(0, int(gap)))
    dash_mask = ((xx + yy) % period) < max(1, int(dash))
    return binary & dash_mask


def overlay_mask_contour(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    dash: int = 8,
    gap: int = 5,
    width: int = 2,
    shadow: bool = True,
) -> np.ndarray:
    out = np.asarray(rgb).copy()
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 image, got shape={out.shape}")
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype("uint8")
    boundary = mask_boundary(mask)
    contour = dashed_boundary(dilate_binary(boundary, radius=max(0, int(width) - 1)), dash=dash, gap=gap)
    if shadow:
        shadow_mask = dilate_binary(contour, radius=1)
        out[shadow_mask] = np.array((255, 255, 255), dtype="uint8")
    out[contour] = np.array(color, dtype="uint8")
    return out


def _mask_only_contour(mask: np.ndarray) -> np.ndarray:
    base = np.zeros((*np.asarray(mask).shape, 3), dtype="uint8")
    base[:] = (20, 20, 20)
    base[np.asarray(mask) > 0] = (55, 55, 55)
    return overlay_mask_contour(base, mask, width=2, shadow=True)


def _kind_color(kind: str) -> tuple[int, int, int, int]:
    return {
        "positive": (255, 40, 40, 230),
        "partial_positive": (255, 170, 0, 230),
        "hard_negative": (0, 130, 255, 220),
        "negative": (180, 180, 180, 160),
    }.get(kind, (255, 255, 255, 180))


def _config_to_dict(config: TilePreparationConfig, augmentation_mode: str, augmentation_seed: int, max_tile_examples: int, max_augmentation_tiles: int) -> dict[str, Any]:
    payload = asdict(config)
    payload.update(
        {
            "positive_stride": config.positive_stride,
            "hard_negative_stride": config.hard_negative_stride,
            "negative_stride": config.negative_stride,
            "augmentation_mode": augmentation_mode,
            "augmentation_seed": augmentation_seed,
            "max_tile_examples": max_tile_examples,
            "max_augmentation_tiles": max_augmentation_tiles,
        }
    )
    return payload


def _clear_pngs(directory: Path) -> None:
    for path in directory.glob("*.png"):
        path.unlink(missing_ok=True)


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None
