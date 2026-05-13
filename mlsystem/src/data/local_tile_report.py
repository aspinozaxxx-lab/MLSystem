from __future__ import annotations

import html
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from rasterio.windows import Window

from .debug_augmentations import (
    DEFAULT_AUGMENTATION_OPERATIONS,
    TRAINING_AUGMENTATION_KEYS,
    apply_debug_augmentation,
    augmentation_catalog,
    resolve_augmentation_operations,
)
from .virtual_tile_sampling import generate_window_grid_for_scene

RASTER_SUFFIXES = (".tif", ".tiff")
STRIDE_FACTORS = (1.0, 0.5, 0.25)


@dataclass
class LocalTileReportConfig:
    images_dir: Path
    tile_size: int = 768
    stride: int = 512
    positive_stride_factor: float = 0.5
    hard_negative_stride_factor: float = 0.5
    negative_stride_factor: float = 1.0
    positive_repeat_factor: int = 4
    hard_negative_repeat_factor: int = 2
    negative_repeat_factor: int = 1
    max_empty_tile_share: float | None = 0.35
    max_overview_size: int = 1600
    max_tile_examples: int = 24
    augment_examples_per_tile: int = 3
    max_augmentation_tiles: int = 8
    augmentation_mode: str = "all"
    augmentations: list[str] | None = None
    augmentation_seed: int = 42
    recursive: bool = False
    seed: int = 42


def find_local_rasters(images_dir: Path, *, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        (
            path
            for path in images_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in RASTER_SUFFIXES
        ),
        key=lambda item: str(item).lower(),
    )


def expected_axis_count(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return 1
    return int(math.ceil((length - tile_size) / max(1, stride))) + 1


def build_tiling_check(width: int, height: int, tile_size: int, base_stride: int, stride_factor: float) -> dict[str, Any]:
    effective_stride = max(1, int(base_stride * float(stride_factor)))
    windows = generate_window_grid_for_scene(width, height, tile_size, effective_stride, scene_id="local")
    xs = sorted({window.x for window in windows})
    ys = sorted({window.y for window in windows})
    expected_nx = expected_axis_count(width, tile_size, effective_stride)
    expected_ny = expected_axis_count(height, tile_size, effective_stride)
    out_of_bounds = [
        window
        for window in windows
        if window.x < 0 or window.y < 0 or window.x + window.width > width or window.y + window.height > height
    ]
    coverage_x = max((window.x + window.width for window in windows), default=0)
    coverage_y = max((window.y + window.height for window in windows), default=0)
    actual_nx = len(xs)
    actual_ny = len(ys)
    actual_total = len(windows)
    expected_total = expected_nx * expected_ny
    return {
        "stride_factor": float(stride_factor),
        "effective_stride": effective_stride,
        "expected_nx": expected_nx,
        "actual_nx": actual_nx,
        "expected_ny": expected_ny,
        "actual_ny": actual_ny,
        "expected_total": expected_total,
        "actual_total": actual_total,
        "coverage_x": coverage_x,
        "coverage_y": coverage_y,
        "last_x": xs[-1] if xs else None,
        "last_y": ys[-1] if ys else None,
        "out_of_bounds_windows": len(out_of_bounds),
        "pass": bool(
            expected_nx == actual_nx
            and expected_ny == actual_ny
            and expected_total == actual_total
            and coverage_x == width
            and coverage_y == height
            and not out_of_bounds
        ),
    }


def preview_local_tile_reports(
    images_dir: str | Path,
    *,
    tile_size: int,
    stride: int,
    stride_factors: list[float] | None = None,
    recursive: bool = False,
    max_scenes: int | None = None,
    max_records_preview: int = 20,
    include_augmentation_catalog: bool = False,
) -> dict[str, Any]:
    import rasterio

    root = Path(images_dir)
    rasters = find_local_rasters(root, recursive=recursive)
    if max_scenes is not None:
        rasters = rasters[: max(0, int(max_scenes))]
    factors = stride_factors or list(STRIDE_FACTORS)
    scenes: list[dict[str, Any]] = []
    for path in rasters:
        with rasterio.open(path) as ds:
            checks = [build_tiling_check(ds.width, ds.height, tile_size, stride, factor) for factor in factors]
            windows = generate_window_grid_for_scene(ds.width, ds.height, tile_size, stride, scene_id=path.stem)
            scenes.append(
                {
                    "scene": path.stem,
                    "path": str(path),
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "width": ds.width,
                    "height": ds.height,
                    "bands": ds.count,
                    "dtype": ",".join(ds.dtypes),
                    "crs": str(ds.crs) if ds.crs else None,
                    "bounds": _bounds_to_list(ds.bounds),
                    "nodata": ds.nodata,
                    "base_tile_count": checks[0]["actual_total"] if checks else len(windows),
                    "tiling_checks": checks,
                    "preview_records": [
                        {
                            "tile_id": f"tile_{index:03d}",
                            "x": window.x,
                            "y": window.y,
                            "width": window.width,
                            "height": window.height,
                            "kind": "unlabeled_image_only",
                        }
                        for index, window in enumerate(windows[: max(0, int(max_records_preview))])
                    ],
                    "limitations": [_image_only_limitation()],
                }
            )
    payload: dict[str, Any] = {
        "status": "ok",
        "summary": {
            "images_dir": str(root),
            "scene_count": len(scenes),
            "tile_size": int(tile_size),
            "stride": int(stride),
            "stride_factors": factors,
            "limitations": [_image_only_limitation()],
            "augmentation_report_available": False,
        },
        "scenes": scenes,
    }
    if include_augmentation_catalog:
        payload["summary"]["augmentation_operations_supported"] = list(DEFAULT_AUGMENTATION_OPERATIONS)
        payload["summary"]["training_augmentation_keys_supported"] = list(TRAINING_AUGMENTATION_KEYS)
        payload["augmentation_catalog"] = augmentation_catalog()
    return payload


def generate_local_tile_reports(config: LocalTileReportConfig) -> dict[str, Any]:
    import rasterio

    images_dir = Path(config.images_dir)
    rasters = find_local_rasters(images_dir, recursive=config.recursive)
    scenes: list[dict[str, Any]] = []
    for raster_path in rasters:
        with rasterio.open(raster_path) as ds:
            scene_dir = images_dir / raster_path.stem
            scene_dir.mkdir(parents=True, exist_ok=True)
            tiles_dir = scene_dir / "tiles"
            augmentations_dir = scene_dir / "augmentations"
            tiles_dir.mkdir(parents=True, exist_ok=True)
            augmentations_dir.mkdir(parents=True, exist_ok=True)
            _clear_debug_pngs(tiles_dir)
            _clear_debug_pngs(augmentations_dir)

            checks = [build_tiling_check(ds.width, ds.height, config.tile_size, config.stride, factor) for factor in STRIDE_FACTORS]
            metadata = _raster_metadata(ds, raster_path)
            overview_files = _write_overviews(ds, raster_path.stem, scene_dir, config, checks)
            windows = generate_window_grid_for_scene(ds.width, ds.height, config.tile_size, config.stride, scene_id=raster_path.stem)
            selected_windows = _select_example_windows(windows, config.max_tile_examples, seed=config.seed)
            example_tiles = _write_tile_and_augmentation_examples(ds, selected_windows, tiles_dir, augmentations_dir, config)
            augmentation_report = _build_augmentation_report(example_tiles, config)
            summary = {
                "scene": raster_path.stem,
                "image_path": str(raster_path),
                **metadata,
                "tile_size": config.tile_size,
                "base_stride": config.stride,
                "tiling_checks": checks,
                "parameter_matrix": _image_only_parameter_matrix(checks, config),
                "overview_images": overview_files,
                "example_tiles": example_tiles,
                "augmentation_report": augmentation_report,
                "limitations": [_image_only_limitation()],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            summary_path = scene_dir / "scene_summary.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
            html_path = scene_dir / "tile_sampling_report.html"
            html_path.write_text(_render_scene_html(summary, config, raster_path), encoding="utf-8")
            scenes.append(
                {
                    "scene": raster_path.stem,
                    "image_path": str(raster_path),
                    "size_bytes": raster_path.stat().st_size,
                    "width": metadata["width"],
                    "height": metadata["height"],
                    "bands": metadata["bands"],
                    "crs": metadata["crs"],
                    "tile_size": config.tile_size,
                    "stride": config.stride,
                    "base_tile_count": checks[0]["actual_total"],
                    "dense_0_5_tile_count": checks[1]["actual_total"],
                    "dense_0_25_tile_count": checks[2]["actual_total"],
                    "report_path": str(html_path),
                    "summary_path": str(summary_path),
                    "tile_preview_count": len(example_tiles),
                    "augmentation_preview_count": sum(len(item.get("augmentations") or []) for item in example_tiles),
                    "augmentation_report": {
                        "mode": augmentation_report["mode"],
                        "operation_count": len(augmentation_report["operations"]),
                        "checks_summary": augmentation_report["checks_summary"],
                    },
                    "tiling_checks": checks,
                }
            )

    index = {
        "status": "ok",
        "images_dir": str(images_dir),
        "scene_count": len(scenes),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": _config_parameters(config),
        "limitations": [_image_only_limitation()],
        "augmentation_operations_supported": list(DEFAULT_AUGMENTATION_OPERATIONS),
        "training_augmentation_keys_supported": list(TRAINING_AUGMENTATION_KEYS),
        "scenes": scenes,
    }
    (images_dir / "tile_sampling_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (images_dir / "tile_sampling_index.html").write_text(_render_index_html(index), encoding="utf-8")
    return index


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
        "bounds": _bounds_to_list(ds.bounds),
        "nodata": ds.nodata,
    }


def _write_overviews(ds: Any, scene_name: str, scene_dir: Path, config: LocalTileReportConfig, checks: list[dict[str, Any]]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    overview = _read_overview_rgb(ds, config.max_overview_size)
    scale_x = overview.shape[1] / float(ds.width)
    scale_y = overview.shape[0] / float(ds.height)
    for check, filename in zip(checks, ("overview_grid_base.png", "overview_grid_dense_0_5.png", "overview_grid_dense_0_25.png")):
        image = Image.fromarray(overview, mode="RGB").convert("RGBA")
        draw = ImageDraw.Draw(image)
        windows = generate_window_grid_for_scene(ds.width, ds.height, config.tile_size, int(check["effective_stride"]), scene_id=scene_name)
        for window in windows:
            draw.rectangle(
                (
                    int(window.x * scale_x),
                    int(window.y * scale_y),
                    int((window.x + window.width) * scale_x),
                    int((window.y + window.height) * scale_y),
                ),
                outline=(255, 220, 0, 210),
                width=1,
            )
        label = f"{scene_name} | {ds.width}x{ds.height} | tile={config.tile_size} | stride={check['effective_stride']} | tiles={check['actual_total']}"
        draw.rectangle((0, 0, min(image.width, max(460, len(label) * 8)), 28), fill=(0, 0, 0, 190))
        draw.text((8, 7), label, fill=(255, 255, 255, 255))
        out_path = scene_dir / filename
        image.convert("RGB").save(out_path)
        outputs[filename] = filename
    return outputs


def _read_overview_rgb(ds: Any, max_size: int) -> np.ndarray:
    scale = min(float(max_size) / max(ds.width, ds.height), 1.0)
    out_width = max(1, int(ds.width * scale))
    out_height = max(1, int(ds.height * scale))
    bands = list(range(1, min(3, ds.count) + 1))
    arr = ds.read(bands, out_shape=(len(bands), out_height, out_width), boundless=True, fill_value=0)
    return _to_rgb_uint8(arr)


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.shape[0] == 1:
        rgb = np.repeat(arr[:1], 3, axis=0)
    elif arr.shape[0] == 2:
        rgb = np.concatenate([arr, arr[:1]], axis=0)
    else:
        rgb = arr[:3]
    data = rgb.astype("float32")
    out = np.zeros_like(data, dtype="uint8")
    for idx in range(data.shape[0]):
        band = data[idx]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            lo, hi = float(valid.min()), float(valid.max() or 1.0)
        out[idx] = np.clip((band - lo) / max(1e-6, hi - lo) * 255.0, 0, 255).astype("uint8")
    return np.transpose(out, (1, 2, 0))


def _select_example_windows(windows: list[Any], max_count: int, *, seed: int) -> list[Any]:
    if not windows or max_count <= 0:
        return []
    max_x = max(window.x + window.width for window in windows)
    max_y = max(window.y + window.height for window in windows)
    edge = [window for window in windows if window.x == 0 or window.y == 0 or window.x + window.width == max_x or window.y + window.height == max_y]
    center_x = sum(window.x for window in windows) / len(windows)
    center_y = sum(window.y for window in windows) / len(windows)
    center = sorted(windows, key=lambda item: (item.x - center_x) ** 2 + (item.y - center_y) ** 2)[: max(1, max_count // 3)]
    random_windows = list(windows)
    random.Random(seed).shuffle(random_windows)
    selected: list[Any] = []
    seen: set[tuple[int, int]] = set()
    for bucket in (center, edge, random_windows):
        for window in bucket:
            key = (window.x, window.y)
            if key in seen:
                continue
            selected.append(window)
            seen.add(key)
            if len(selected) >= max_count:
                return selected
    return selected


def _write_tile_and_augmentation_examples(
    ds: Any,
    windows: list[Any],
    tiles_dir: Path,
    augmentations_dir: Path,
    config: LocalTileReportConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    operations = resolve_augmentation_operations(config.augmentation_mode, config.augmentations)
    matrix_limit = max(0, int(config.max_augmentation_tiles))
    for index, window in enumerate(windows):
        arr = ds.read(
            list(range(1, min(3, ds.count) + 1)),
            window=Window(window.x, window.y, window.width, window.height),
            boundless=True,
            fill_value=0,
        )
        rgb = _to_rgb_uint8(arr)
        original_name = f"tile_{index:03d}_original.png"
        Image.fromarray(rgb, mode="RGB").save(tiles_dir / original_name)
        row = {
            "tile_id": f"tile_{index:03d}",
            "x": int(window.x),
            "y": int(window.y),
            "width": int(window.width),
            "height": int(window.height),
            "is_edge": bool(window.x == 0 or window.y == 0 or window.x + window.width >= ds.width or window.y + window.height >= ds.height),
            "classification": "unlabeled_image_only",
            "nodata_share": _nodata_share(arr, ds.nodata),
            "preview_path": f"tiles/{original_name}",
            "augmentations": [],
        }
        for aug_index, operation in enumerate(operations if index < matrix_limit else []):
            seed = config.augmentation_seed + index * 1000 + aug_index
            aug_rgb, metadata = apply_debug_augmentation(rgb, operation, seed=seed)
            aug_name = f"tile_{index:03d}_{operation}.png"
            Image.fromarray(aug_rgb, mode="RGB").save(augmentations_dir / aug_name)
            row["augmentations"].append({"path": f"augmentations/{aug_name}", **metadata})
        rows.append(row)
    return rows


def _build_augmentation_report(example_tiles: list[dict[str, Any]], config: LocalTileReportConfig) -> dict[str, Any]:
    operations = resolve_augmentation_operations(config.augmentation_mode, config.augmentations)
    examples_by_operation: dict[str, list[dict[str, Any]]] = {operation: [] for operation in operations}
    status_counts = {"pass": 0, "warning": 0, "failed": 0}
    for tile in example_tiles:
        for aug in tile.get("augmentations") or []:
            status = str(aug.get("check_status") or "failed")
            status_counts[status] = status_counts.get(status, 0) + 1
            examples_by_operation.setdefault(str(aug.get("operation")), []).append(
                {
                    "tile_id": tile["tile_id"],
                    "path": aug.get("path"),
                    "seed": aug.get("seed"),
                    "check_status": status,
                    "changed_pixels_fraction": aug.get("changed_pixels_fraction"),
                    "checks": aug.get("checks"),
                }
            )
    operation_checks: list[dict[str, Any]] = []
    for operation in operations:
        catalog_item = augmentation_catalog([operation])[0]
        examples = examples_by_operation.get(operation) or []
        statuses = [str(item.get("check_status")) for item in examples]
        changed = [float(item.get("changed_pixels_fraction") or 0.0) for item in examples]
        if any(item == "failed" for item in statuses):
            status = "failed"
        elif any(item == "warning" for item in statuses):
            status = "warning"
        else:
            status = "pass"
        operation_checks.append(
            {
                **catalog_item,
                "status": status,
                "changed_pixels_fraction_mean": round(sum(changed) / len(changed), 6) if changed else None,
                "examples": examples[:10],
            }
        )
    return {
        "mode": config.augmentation_mode,
        "seed": config.augmentation_seed,
        "operations": operations,
        "tile_count_with_full_augmentation_matrix": min(config.max_augmentation_tiles, len(example_tiles)),
        "total_augmentation_previews": sum(len(tile.get("augmentations") or []) for tile in example_tiles),
        "checks_summary": {
            "passed": status_counts.get("pass", 0),
            "warnings": status_counts.get("warning", 0),
            "failed": status_counts.get("failed", 0),
        },
        "operation_checks": operation_checks,
    }


def _clear_debug_pngs(directory: Path) -> None:
    for path in directory.glob("*.png"):
        path.unlink(missing_ok=True)


def _nodata_share(arr: np.ndarray, nodata: Any) -> float | None:
    if nodata is None:
        empty = np.all(arr == 0, axis=0)
    else:
        empty = np.all(arr == nodata, axis=0)
    return round(float(np.count_nonzero(empty)) / float(empty.size), 6) if empty.size else None


def _image_only_parameter_matrix(checks: list[dict[str, Any]], config: LocalTileReportConfig) -> list[dict[str, Any]]:
    base_total = int(checks[0]["actual_total"]) if checks else 0
    return [
        {
            "case": "baseline_stride_factor_1_0",
            "base_window_count": base_total,
            "simulated_virtual_count": base_total,
            "note": "image-only: class-aware positive/hard_negative/negative repeats require annotation",
        },
        {
            "case": "dense_stride_factor_0_5",
            "base_window_count": int(checks[1]["actual_total"]) if len(checks) > 1 else None,
            "simulated_virtual_count": int(checks[1]["actual_total"]) if len(checks) > 1 else None,
            "note": "image-only dense window count",
        },
        {
            "case": "dense_stride_factor_0_25",
            "base_window_count": int(checks[2]["actual_total"]) if len(checks) > 2 else None,
            "simulated_virtual_count": int(checks[2]["actual_total"]) if len(checks) > 2 else None,
            "note": "image-only dense window count",
        },
        {
            "case": "repeat_factors_4_2_1",
            "base_window_count": base_total,
            "simulated_virtual_count_if_all_tiles_used_positive_repeat": base_total * config.positive_repeat_factor,
            "simulated_virtual_count_if_all_tiles_used_hard_negative_repeat": base_total * config.hard_negative_repeat_factor,
            "simulated_virtual_count_if_all_tiles_used_negative_repeat": base_total * config.negative_repeat_factor,
            "note": "not a real class-aware count because there is no mask/annotation",
        },
    ]


def _render_scene_html(summary: dict[str, Any], config: LocalTileReportConfig, raster_path: Path) -> str:
    rel = lambda value: html.escape(str(value).replace("\\", "/"))
    tiling_rows = "\n".join(
        "<tr>"
        f"<td>{check['stride_factor']}</td><td>{check['effective_stride']}</td>"
        f"<td>{check['expected_nx']}x{check['expected_ny']}={check['expected_total']}</td>"
        f"<td>{check['actual_nx']}x{check['actual_ny']}={check['actual_total']}</td>"
        f"<td>{check['coverage_x']} / {check['coverage_y']}</td><td>{check['last_x']} / {check['last_y']}</td>"
        f"<td>{check['out_of_bounds_windows']}</td><td>{'PASS' if check['pass'] else 'FAIL'}</td>"
        "</tr>"
        for check in summary["tiling_checks"]
    )
    param_rows = "\n".join(f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value))}</td></tr>" for key, value in _config_parameters(config).items())
    overview_html = "\n".join(f'<figure><img src="{rel(path)}"><figcaption>{html.escape(name)}</figcaption></figure>' for name, path in summary["overview_images"].items())
    tile_html = "\n".join(
        f'<figure><img src="{rel(tile["preview_path"])}"><figcaption>{tile["tile_id"]}: x={tile["x"]}, y={tile["y"]}, edge={tile["is_edge"]}, nodata={tile["nodata_share"]}</figcaption></figure>'
        for tile in summary["example_tiles"]
    )
    aug_report = summary.get("augmentation_report") or {}
    coverage_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(item['operation'])}</td><td>{html.escape(str(item.get('training_key')))}</td>"
        f"<td>{html.escape(str(item.get('group')))}</td><td>{html.escape(json.dumps(item.get('parameters') or {}, ensure_ascii=False))}</td>"
        f"<td>{html.escape(str(item.get('status')))}</td><td>{html.escape(str(item.get('changed_pixels_fraction_mean')))}</td>"
        f"<td>{html.escape(str(item.get('debug_note')))}</td>"
        "</tr>"
        for item in aug_report.get("operation_checks", [])
    )
    check_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(aug['operation'])}</td><td>{html.escape(str(aug.get('training_key')))}</td>"
        f"<td>{html.escape(str(aug.get('checks', {}).get('shape_preserved')))}</td>"
        f"<td>{html.escape(str(aug.get('checks', {}).get('dtype_uint8')))}</td>"
        f"<td>{html.escape(str(aug.get('checks', {}).get('range_0_255')))}</td>"
        f"<td>{html.escape(str(aug.get('checks', {}).get('non_empty_output')))}</td>"
        f"<td>{html.escape(str(aug.get('checks', {}).get('changed_when_expected')))}</td>"
        f"<td>{html.escape(str(aug.get('check_status')))}</td>"
        "</tr>"
        for tile in summary["example_tiles"]
        for aug in tile.get("augmentations", [])
    )
    grouped_tiles: dict[str, list[str]] = {}
    for tile in summary["example_tiles"]:
        for aug in tile.get("augmentations", []):
            group = str(aug.get("group") or "other")
            grouped_tiles.setdefault(group, []).append(
                f'<figure><img src="{rel(aug["path"])}"><figcaption>{tile["tile_id"]} | {html.escape(aug["operation"])} | seed={aug["seed"]} | changed={aug.get("changed_pixels_fraction")}</figcaption></figure>'
            )
    aug_sections = "\n".join(
        f"<details open><summary>{html.escape(group)}</summary><section class=\"gallery\">{''.join(items)}</section></details>"
        for group, items in grouped_tiles.items()
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Tile sampling report - {html.escape(summary['scene'])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f8fa; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d0d7de; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    details {{ margin: 12px 0 20px; }}
    summary {{ cursor: pointer; font-weight: bold; padding: 8px 0; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; margin-top: 4px; color: #4b5563; }}
    .warning {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px; }}
  </style>
</head>
<body>
  <h1>{html.escape(summary['scene'])}</h1>
  <p><b>Path:</b> {html.escape(str(raster_path))}</p>
  <p><b>Дата генерации:</b> {html.escape(summary['generated_at'])}</p>
  <h2>Raster metadata</h2>
  <table>
    <tr><th>width</th><td>{summary['width']}</td><th>height</th><td>{summary['height']}</td></tr>
    <tr><th>bands</th><td>{summary['bands']}</td><th>dtype</th><td>{html.escape(str(summary['dtype']))}</td></tr>
    <tr><th>crs</th><td colspan="3">{html.escape(str(summary['crs']))}</td></tr>
    <tr><th>transform</th><td colspan="3">{html.escape(str(summary['transform']))}</td></tr>
    <tr><th>bounds</th><td colspan="3">{html.escape(str(summary['bounds']))}</td></tr>
    <tr><th>nodata</th><td colspan="3">{html.escape(str(summary['nodata']))}</td></tr>
  </table>
  <h2>Parameters</h2>
  <table><tr><th>parameter</th><th>value</th></tr>{param_rows}</table>
  <h2>Tiling summary</h2>
  <table>
    <tr><th>stride_factor</th><th>effective_stride</th><th>expected</th><th>actual</th><th>coverage x/y</th><th>last x/y</th><th>out of bounds</th><th>status</th></tr>
    {tiling_rows}
  </table>
  <h2>Overview</h2>
  <section class="gallery">{overview_html}</section>
  <h2>Tile examples</h2>
  <section class="gallery">{tile_html}</section>
  <h2>Аугментации: покрытие методов</h2>
  <p>Показаны отдельные операции, production keys и debug metadata. Сводка проверок: {html.escape(str(aug_report.get('checks_summary')))}</p>
  <table>
    <tr><th>Operation</th><th>Training key</th><th>Group</th><th>Parameters</th><th>Status</th><th>Changed pixels mean</th><th>Notes</th></tr>
    {coverage_rows}
  </table>
  <h2>Примеры всех аугментаций</h2>
  {aug_sections}
  <h2>Проверки аугментаций</h2>
  <table>
    <tr><th>operation</th><th>training key</th><th>shape</th><th>dtype</th><th>range</th><th>non-empty</th><th>changed expected</th><th>status</th></tr>
    {check_rows}
  </table>
  <h2>Ограничение</h2>
  <p class="warning">{html.escape(_image_only_limitation())}</p>
  <p class="warning">На локальных GeoTIFF проверена визуальная аугментация RGB preview. Согласованность image+mask проверяется unit-test на synthetic mask.</p>
  <h2>JSON summary</h2>
  <p><a href="scene_summary.json">scene_summary.json</a></p>
</body>
</html>
"""


def _render_index_html(index: dict[str, Any]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(scene['scene'])}</td><td>{scene['width']}x{scene['height']}</td>"
        f"<td>{scene['bands']}</td><td>{scene['tile_size']}</td><td>{scene['stride']}</td>"
        f"<td>{scene['base_tile_count']}</td><td>{scene['dense_0_5_tile_count']}</td><td>{scene['dense_0_25_tile_count']}</td>"
        f"<td><a href=\"{html.escape(Path(scene['report_path']).parent.name)}/tile_sampling_report.html\">report</a></td>"
        "</tr>"
        for scene in index["scenes"]
    )
    return f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Local tile sampling index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    .warning {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Local tile sampling index</h1>
  <p><b>Images dir:</b> {html.escape(index['images_dir'])}</p>
  <p><b>Дата генерации:</b> {html.escape(index['generated_at'])}</p>
  <p class="warning">{html.escape(_image_only_limitation())}</p>
  <table>
    <tr><th>scene</th><th>size</th><th>bands</th><th>tile_size</th><th>stride</th><th>base</th><th>dense 0.5</th><th>dense 0.25</th><th>report</th></tr>
    {rows}
  </table>
  <p><a href="tile_sampling_index.json">tile_sampling_index.json</a></p>
</body>
</html>
"""


def _config_parameters(config: LocalTileReportConfig) -> dict[str, Any]:
    return {
        "tile_size": config.tile_size,
        "stride": config.stride,
        "positive_stride_factor": config.positive_stride_factor,
        "hard_negative_stride_factor": config.hard_negative_stride_factor,
        "negative_stride_factor": config.negative_stride_factor,
        "positive_repeat_factor": config.positive_repeat_factor,
        "hard_negative_repeat_factor": config.hard_negative_repeat_factor,
        "negative_repeat_factor": config.negative_repeat_factor,
        "max_empty_tile_share": config.max_empty_tile_share,
        "max_overview_size": config.max_overview_size,
        "max_tile_examples": config.max_tile_examples,
        "max_augmentation_tiles": config.max_augmentation_tiles,
        "augmentation_mode": config.augmentation_mode,
        "augmentations": ", ".join(resolve_augmentation_operations(config.augmentation_mode, config.augmentations)),
        "augmentation_seed": config.augmentation_seed,
        "seed": config.seed,
    }


def _bounds_to_list(bounds: Any) -> list[float]:
    return [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]


def _image_only_limitation() -> str:
    return (
        "В этой локальной проверке нет dataset_annotation.geojson, поэтому positive/partial_positive/"
        "hard_negative/negative классификация не проверялась на реальной разметке. Проверены raster read, "
        "window generation, edge coverage, dense stride и визуальная адекватность tile/augmentation."
    )
