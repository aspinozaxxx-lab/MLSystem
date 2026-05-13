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
from .mask_rasterizer import rasterize_mask_for_window
from .raster_reader import read_rgb_window, to_rgb_uint8
from .records import TileSampleRecord
from .summary import summarize_tile_records
from .windows import build_tiling_check


RASTER_SUFFIXES = {".tif", ".tiff"}


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
) -> dict[str, Any]:
    scene, annotation = find_single_annotated_scene(
        input_dir,
        annotation_crs=annotation_crs,
        allow_inferred_annotation_crs=allow_inferred_annotation_crs,
    )
    config = TilePreparationConfig(
        tile_size=tile_size,
        stride=stride,
        positive_stride_factor=positive_stride_factor,
        hard_negative_stride_factor=hard_negative_stride_factor,
        negative_stride_factor=negative_stride_factor,
        min_positive_pixels=min_positive_pixels,
        hard_negative_context_px=max(1, int(tile_size) // 2),
    )
    result = build_tile_records([scene], annotation, config)
    with rasterio.open(scene.image_path) as ds:
        raster = _raster_metadata(ds, scene.image_path)
    base_summary = summarize_tile_records(result.base_records)
    response: dict[str, Any] = {
        "status": "ok",
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
    annotation_crs: str | None = "auto",
    allow_inferred_annotation_crs: bool = True,
    max_overview_size: int = 1600,
    max_tile_examples: int = 32,
    max_augmentation_tiles: int = 8,
    augmentation_mode: str = "all",
    augmentation_seed: int = 42,
) -> dict[str, Any]:
    root = Path(input_dir)
    scene, annotation = find_single_annotated_scene(
        root,
        annotation_crs=annotation_crs,
        allow_inferred_annotation_crs=allow_inferred_annotation_crs,
    )
    scene_dir = root / scene.resolved_scene_id()
    scene_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = scene_dir / "annotated_tiles"
    augmentations_dir = scene_dir / "annotated_augmentations"
    previews_dir.mkdir(parents=True, exist_ok=True)
    augmentations_dir.mkdir(parents=True, exist_ok=True)
    _clear_pngs(previews_dir)
    _clear_pngs(augmentations_dir)

    result = build_tile_records([scene], annotation, config)
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


def _write_overviews(
    ds: Any,
    geometries: list[Any],
    records: list[TileSampleRecord],
    scene_dir: Path,
    config: TilePreparationConfig,
    max_size: int,
) -> dict[str, str]:
    overview = _read_overview(ds, max_size)
    mask = _overview_mask(ds, geometries, overview.shape[1], overview.shape[0], config.all_touched)
    overlay = _overlay_mask(overview, mask)
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
        "overview_grid_positive_negative.png": "overview_grid_positive_negative.png",
    }
    base.convert("RGB").save(scene_dir / "overview_raster_grid_mask.png")
    Image.fromarray((mask * 255).astype("uint8"), mode="L").save(scene_dir / "overview_mask_only.png")
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
    max_augmentation_tiles: int,
    augmentation_mode: str,
    augmentation_seed: int,
) -> list[dict[str, Any]]:
    operations = resolve_augmentation_operations(augmentation_mode)
    rows: list[dict[str, Any]] = []
    augmented_positive_tiles = 0
    for index, record in enumerate(records):
        rgb = read_rgb_window(ds, record, bands=config.input_bands)
        mask, geom_count = rasterize_mask_for_window(ds, geometries, record, all_touched=config.all_touched)
        stem = f"tile_{index:03d}_{record.kind}"
        original_name = f"{stem}_rgb.png"
        mask_name = f"{stem}_mask.png"
        overlay_name = f"{stem}_overlay.png"
        Image.fromarray(rgb, mode="RGB").save(previews_dir / original_name)
        Image.fromarray((mask * 255).astype("uint8"), mode="L").save(previews_dir / mask_name)
        Image.fromarray(_overlay_mask(rgb, mask), mode="RGB").save(previews_dir / overlay_name)
        row = {
            **record.to_dict(),
            "geometries_intersecting": geom_count,
            "positive_pixels_runtime": int(mask.sum()),
            "preview_rgb": f"annotated_tiles/{original_name}",
            "preview_mask": f"annotated_tiles/{mask_name}",
            "preview_overlay": f"annotated_tiles/{overlay_name}",
            "augmentations": [],
        }
        should_write_aug = int(mask.sum()) > 0 and augmented_positive_tiles < max_augmentation_tiles
        if should_write_aug:
            augmented_positive_tiles += 1
            for aug_index, operation in enumerate(operations):
                seed = augmentation_seed + index * 1000 + aug_index
                aug_rgb, aug_mask, metadata = apply_debug_augmentation_with_mask(rgb, operation, seed=seed, mask=mask)
                aug_stem = f"tile_{index:03d}_{operation}"
                rgb_name = f"{aug_stem}_rgb.png"
                mask_aug_name = f"{aug_stem}_mask.png"
                overlay_aug_name = f"{aug_stem}_overlay.png"
                Image.fromarray(aug_rgb, mode="RGB").save(augmentations_dir / rgb_name)
                Image.fromarray(((aug_mask if aug_mask is not None else mask) * 255).astype("uint8"), mode="L").save(augmentations_dir / mask_aug_name)
                Image.fromarray(_overlay_mask(aug_rgb, aug_mask if aug_mask is not None else mask), mode="RGB").save(augmentations_dir / overlay_aug_name)
                row["augmentations"].append(
                    {
                        "rgb_path": f"annotated_augmentations/{rgb_name}",
                        "mask_path": f"annotated_augmentations/{mask_aug_name}",
                        "overlay_path": f"annotated_augmentations/{overlay_aug_name}",
                        **metadata,
                    }
                )
        rows.append(row)
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
        "cutout_mask_behavior": "unchanged, matching real_train.py production behavior",
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
        f"<figcaption>{html.escape(tile['kind'])}: {html.escape(tile['tile_id'])}, positive_pixels={tile['positive_pixels_runtime']}, geoms={tile['geometries_intersecting']}</figcaption>"
        "</figure>"
        for tile in summary["tile_examples"]
    )
    aug_rows = "\n".join(
        f"<tr><td>{item['operation']}</td><td>{item['training_key']}</td><td>{item['group']}</td><td>{html.escape(json.dumps(item.get('parameters') or {}, ensure_ascii=False))}</td><td>{item['status']}</td><td>{item.get('changed_pixels_fraction_mean')}</td></tr>"
        for item in summary["augmentation_report"]["operation_checks"]
    )
    check_rows = "\n".join(
        f"<tr><td>{aug['operation']}</td><td>{aug.get('training_key')}</td><td>{aug.get('checks', {}).get('shape_preserved')}</td><td>{aug.get('mask_checks', {}).get('mask_shape_preserved')}</td><td>{aug.get('mask_checks', {}).get('mask_binary')}</td><td>{aug.get('checks', {}).get('range_0_255')}</td><td>{aug.get('mask_alignment_check')}</td><td>{aug.get('changed_pixels_fraction')}</td><td>{aug.get('check_status')}</td></tr>"
        for tile in summary["tile_examples"]
        for aug in tile.get("augmentations", [])
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
  <h2>Scene-level mask overview</h2>
  <section class="gallery">{overview_html}</section>
  <h2>Tiling summary</h2>
  <table><tr><th>stride type</th><th>effective stride</th><th>expected</th><th>actual</th><th>coverage</th><th>out of bounds</th><th>status</th></tr>{tiling_rows}</table>
  <h2>Training tile classification summary</h2>
  <table><tr><th>metric</th><th>value</th></tr>{class_rows}</table>
  <h2>Positive / negative examples</h2>
  <section class="gallery">{tile_html}</section>
  <h2>Augmentation examples with masks</h2>
  <section class="gallery">{aug_gallery}</section>
  <h2>Augmentation checks</h2>
  <table><tr><th>operation</th><th>training_key</th><th>image shape</th><th>mask shape</th><th>mask binary</th><th>image range</th><th>mask alignment</th><th>changed pixels</th><th>status</th></tr>{check_rows}</table>
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


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_img = Image.fromarray((mask > 0).astype("uint8") * 135, mode="L")
    color = Image.new("RGBA", base.size, (255, 40, 40, 120))
    overlay.paste(color, (0, 0), mask_img)
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


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
