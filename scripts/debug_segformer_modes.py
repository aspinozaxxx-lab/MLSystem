from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.features import shapes as raster_shapes
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union


MODES = [
    "hard_insert_full_tile",
    "hard_insert_center_crop",
    "weighted_overlap_full_tile",
    "weighted_overlap_center_crop",
]


@dataclass(frozen=True)
class ModeSpec:
    name: str
    full_tile: bool
    weighted: bool


MODE_SPECS = {
    "hard_insert_full_tile": ModeSpec("hard_insert_full_tile", full_tile=True, weighted=False),
    "hard_insert_center_crop": ModeSpec("hard_insert_center_crop", full_tile=False, weighted=False),
    "weighted_overlap_full_tile": ModeSpec("weighted_overlap_full_tile", full_tile=True, weighted=True),
    "weighted_overlap_center_crop": ModeSpec("weighted_overlap_center_crop", full_tile=False, weighted=True),
}


def _normalize_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    arr[~np.isfinite(arr)] = 0.0
    out = np.zeros_like(arr, dtype="float32")
    for band in range(arr.shape[0]):
        data = arr[band]
        valid = data[data != 0]
        if valid.size < 16:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        out[band] = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return out


def _build_model() -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=4, classes=1, activation=None)


def choose_device(requested: str) -> tuple[torch.device, dict[str, Any]]:
    requested = requested.lower()
    info: dict[str, Any] = {"requested": requested, "backend": "cpu", "device_name": "cpu", "gpu_used": False}
    if requested in {"auto", "directml", "dml"}:
        try:
            import torch_directml

            device = torch_directml.device()
            info.update({"backend": "directml", "device_name": str(device), "gpu_used": True})
            return device, info
        except Exception as exc:
            if requested in {"directml", "dml"}:
                raise
            info["directml_error"] = f"{type(exc).__name__}: {exc}"
    if requested == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        info.update({"backend": "cuda", "device_name": torch.cuda.get_device_name(0), "gpu_used": True})
        return device, info
    return torch.device("cpu"), info


def origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, max(1, length - tile + 1), max(1, stride)))
    edge = length - tile
    if values[-1] != edge:
        values.append(edge)
    return sorted(set(max(0, int(value)) for value in values))


def window_grid(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    return [(x, y) for y in origins(height, tile, stride) for x in origins(width, tile, stride)]


def center_insert_slices(x: int, y: int, actual_w: int, actual_h: int, scene_width: int, scene_height: int, context: int) -> dict[str, int]:
    crop_left = context if x > 0 else 0
    crop_top = context if y > 0 else 0
    crop_right = context if x + actual_w < scene_width else 0
    crop_bottom = context if y + actual_h < scene_height else 0
    if actual_w - crop_left - crop_right <= 0:
        crop_left = crop_right = 0
    if actual_h - crop_top - crop_bottom <= 0:
        crop_top = crop_bottom = 0
    return {
        "crop_x0": int(crop_left),
        "crop_y0": int(crop_top),
        "crop_x1": int(actual_w - crop_right),
        "crop_y1": int(actual_h - crop_bottom),
        "insert_x": int(x + crop_left),
        "insert_y": int(y + crop_top),
        "insert_width": int(actual_w - crop_left - crop_right),
        "insert_height": int(actual_h - crop_top - crop_bottom),
    }


def full_insert_slices(x: int, y: int, actual_w: int, actual_h: int) -> dict[str, int]:
    return {
        "crop_x0": 0,
        "crop_y0": 0,
        "crop_x1": int(actual_w),
        "crop_y1": int(actual_h),
        "insert_x": int(x),
        "insert_y": int(y),
        "insert_width": int(actual_w),
        "insert_height": int(actual_h),
    }


def weighted_window(actual_w: int, actual_h: int, x: int, y: int, scene_width: int, scene_height: int, context: int) -> np.ndarray:
    wx = np.ones(actual_w, dtype="float32")
    wy = np.ones(actual_h, dtype="float32")
    ramp = max(1, int(context))
    if x > 0:
        n = min(ramp, actual_w)
        wx[:n] = np.linspace(0.05, 1.0, n, dtype="float32")
    if x + actual_w < scene_width:
        n = min(ramp, actual_w)
        wx[-n:] = np.minimum(wx[-n:], np.linspace(1.0, 0.05, n, dtype="float32"))
    if y > 0:
        n = min(ramp, actual_h)
        wy[:n] = np.linspace(0.05, 1.0, n, dtype="float32")
    if y + actual_h < scene_height:
        n = min(ramp, actual_h)
        wy[-n:] = np.minimum(wy[-n:], np.linspace(1.0, 0.05, n, dtype="float32"))
    return wy[:, None] * wx[None, :]


def nrg_preview(arr: np.ndarray) -> Image.Image:
    indices = [3, 0, 1] if arr.shape[0] >= 4 else [0, 1, 2]
    rgb = np.stack([arr[i] for i in indices], axis=-1).astype("float32")
    out = np.zeros_like(rgb, dtype="uint8")
    for c in range(3):
        band = rgb[:, :, c]
        if np.any(band > 0):
            lo, hi = np.percentile(band[band > 0], [2, 98])
        else:
            lo, hi = 0, 1
        if hi <= lo:
            hi = lo + 1
        out[:, :, c] = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
    return Image.fromarray(out)


def heatmap_image(prob: np.ndarray) -> Image.Image:
    p = np.clip(prob, 0, 1)
    r = np.clip(2.0 * p, 0, 1)
    g = np.clip(2.0 * (1.0 - np.abs(p - 0.5) * 2.0), 0, 1)
    b = np.clip(2.0 * (1.0 - p), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype("uint8"))


def overlay_prob(base: Image.Image, prob: np.ndarray, alpha: float = 0.48) -> Image.Image:
    base = base.convert("RGBA")
    hm = heatmap_image(prob).resize(base.size).convert("RGBA")
    hm.putalpha(int(255 * alpha))
    return Image.alpha_composite(base, hm)


def save_probability(path: Path, prob: np.ndarray, profile: dict[str, Any]) -> None:
    out_profile = profile.copy()
    out_profile.pop("photometric", None)
    out_profile.update(count=1, dtype="float32", compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(prob.astype("float32"), 1)


def save_mask(path: Path, mask: np.ndarray, profile: dict[str, Any]) -> None:
    out_profile = profile.copy()
    out_profile.pop("photometric", None)
    out_profile.update(count=1, dtype="uint8", compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_geojson(path: Path, features: list[dict[str, Any]], crs_name: str) -> None:
    write_json(path, {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs_name}}, "features": features})


def vertex_count(geom_mapping: dict[str, Any]) -> int:
    coords = geom_mapping.get("coordinates") or []
    if geom_mapping.get("type") == "Polygon":
        return sum(len(ring) for ring in coords)
    if geom_mapping.get("type") == "MultiPolygon":
        return sum(len(ring) for poly in coords for ring in poly)
    return 0


def vectorize_mask(mask: np.ndarray, transform: Any, crs_name: str, min_area: float, simplify: float, max_objects: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_features: list[dict[str, Any]] = []
    post_features: list[dict[str, Any]] = []
    vertices_before = 0
    vertices_after = 0
    for geom, value in raster_shapes(mask, mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        mapped_raw = mapping(poly)
        vertices_before += vertex_count(mapped_raw)
        raw_features.append({"type": "Feature", "properties": {"area_m2": float(poly.area)}, "geometry": mapped_raw})
        if float(poly.area) < min_area:
            continue
        post_poly = poly.simplify(simplify, preserve_topology=True) if simplify > 0 else poly
        if post_poly.is_empty:
            continue
        mapped_post = mapping(post_poly)
        vertices_after += vertex_count(mapped_post)
        post_features.append({"type": "Feature", "properties": {"area_m2": float(post_poly.area)}, "geometry": mapped_post})
    post_features.sort(key=lambda item: float(item["properties"].get("area_m2") or 0), reverse=True)
    accepted = post_features[:max_objects]
    stats = {
        "object_count_before_postprocess": len(raw_features),
        "object_count_after_postprocess": len(accepted),
        "objects_after_filter_before_top": len(post_features),
        "top_limit": max_objects,
        "top_limit_applied": len(post_features) > max_objects,
        "total_area_before": float(sum(float(f["properties"].get("area_m2") or 0) for f in raw_features)),
        "total_area_after": float(sum(float(f["properties"].get("area_m2") or 0) for f in accepted)),
        "vertices_before": vertices_before,
        "vertices_after": vertices_after,
    }
    return raw_features, accepted, stats


def seam_score(prob: np.ndarray, xs: list[int], ys: list[int], sample_step: int = 8) -> dict[str, float]:
    vals = []
    view = prob
    for x in xs:
        if 1 <= x < view.shape[1]:
            vals.append(float(np.mean(np.abs(view[::sample_step, x] - view[::sample_step, x - 1]))))
    for y in ys:
        if 1 <= y < view.shape[0]:
            vals.append(float(np.mean(np.abs(view[y, ::sample_step] - view[y - 1, ::sample_step]))))
    return {"mean_seam_jump": float(np.mean(vals)) if vals else 0.0, "max_seam_jump": float(np.max(vals)) if vals else 0.0}


def geojson_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def feature_union_bounds(features: list[dict[str, Any]]) -> list[float] | None:
    geoms = [shape(item["geometry"]) for item in features if item.get("geometry")]
    if not geoms:
        return None
    return [float(v) for v in unary_union(geoms).bounds]


def preview_from_full(arr: np.ndarray, width: int = 1600) -> Image.Image:
    if arr.ndim == 2:
        scale = max(1, int(math.ceil(arr.shape[1] / width)))
        return heatmap_image(arr[::scale, ::scale])
    scale = max(1, int(math.ceil(arr.shape[-1] / width)))
    return nrg_preview(arr[:, ::scale, ::scale])


def draw_feature_overlay(base: Image.Image, features: list[dict[str, Any]], transform: Any, color: tuple[int, int, int], width: int = 2) -> Image.Image:
    img = base.convert("RGBA")
    draw = ImageDraw.Draw(img)
    sx = base.width
    sy = base.height
    # Invert affine approximately by rasterio rowcol via ~transform.
    inv = ~transform
    for feature in features[:2000]:
        geom = shape(feature["geometry"])
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in parts:
            coords = list(poly.exterior.coords)
            pts = []
            for gx, gy in coords:
                px, py = inv * (gx, gy)
                pts.append((int(px / feature["_scene_width"] * sx) if "_scene_width" in feature else int(px), int(py / feature["_scene_height"] * sy) if "_scene_height" in feature else int(py)))
            if len(pts) > 1:
                draw.line(pts, fill=(*color, 220), width=width)
    return img


def add_scene_dims(features: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    for feature in features:
        feature["_scene_width"] = width
        feature["_scene_height"] = height
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--debug-root", default=r"E:\Projects\debug")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min-area", type=float, default=500)
    parser.add_argument("--simplify", type=float, default=2)
    parser.add_argument("--max-objects", type=int, default=500)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "directml", "dml", "cuda"])
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    modes_root = debug_root / "modes"
    input_tiles_root = debug_root / "input_tiles"
    work_root = debug_root / "work" / "modes"
    for root in [modes_root, input_tiles_root, work_root]:
        root.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        (modes_root / mode).mkdir(parents=True, exist_ok=True)

    device, device_info = choose_device(args.device)
    model = _build_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    started = time.time()
    tile_rows: list[dict[str, Any]] = []
    input_debug_written = 0
    mode_sums: dict[str, np.memmap] = {}
    mode_counts: dict[str, np.memmap] = {}
    mode_insert_features: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    tile_grid_features: list[dict[str, Any]] = []
    seam_x: set[int] = set()
    seam_y: set[int] = set()

    try:
        with rasterio.open(args.image) as ds:
            profile = ds.profile
            profile.pop("photometric", None)
            crs_name = ds.crs.to_string() if ds.crs else ""
            windows = window_grid(ds.width, ds.height, args.tile_size, args.stride)
            for mode in MODES:
                mode_sums[mode] = np.memmap(work_root / f"{mode}_sum.dat", dtype="float32", mode="w+", shape=(ds.height, ds.width))
                mode_counts[mode] = np.memmap(work_root / f"{mode}_count.dat", dtype="float32", mode="w+", shape=(ds.height, ds.width))
                mode_sums[mode][:] = 0
                mode_counts[mode][:] = 0

            source_preview_arr = ds.read([1, 2, 3, 4], out_shape=(4, min(ds.height, 2200), min(ds.width, 1800)))
            source_preview = nrg_preview(source_preview_arr)
            source_preview.save(work_root / "scene_nrg_preview.jpg", quality=90)

            selected_input_tiles = set(list(range(min(5, len(windows)))) + [len(windows) // 2, max(0, len(windows) - 1)])
            for idx, (x, y) in enumerate(windows):
                window = Window(x, y, args.tile_size, args.tile_size)
                actual_w = min(args.tile_size, ds.width - x)
                actual_h = min(args.tile_size, ds.height - y)
                tile_feature_props = {
                    "tile_index": idx,
                    "x": int(x),
                    "y": int(y),
                    "tile_width": int(actual_w),
                    "tile_height": int(actual_h),
                }
                tile_grid_features.append({"type": "Feature", "properties": tile_feature_props, "geometry": mapping(box(*ds.window_bounds(window)))})
                arr = ds.read([1, 2, 3, 4], window=window, boundless=True, fill_value=0)
                if np.count_nonzero(arr) == 0:
                    tile_rows.append({**tile_feature_props, "predicted": False, "skipped_reason": "all_zero"})
                    continue
                norm = _normalize_image(arr)
                if idx in selected_input_tiles and input_debug_written < 10:
                    input_debug_written += 1
                    tile_prefix = input_tiles_root / f"tile_{idx:04d}"
                    nrg_preview(arr).save(f"{tile_prefix}_input_nrg.png")
                    stats = {
                        "tile_index": idx,
                        "x": int(x),
                        "y": int(y),
                        "tile_size": args.tile_size,
                        "actual_width": int(actual_w),
                        "actual_height": int(actual_h),
                        "input_bands": [1, 2, 3, 4],
                        "preview_bands": [4, 1, 2],
                        "raw_band_stats": [
                            {
                                "band": b + 1,
                                "min": float(arr[b].min()),
                                "max": float(arr[b].max()),
                                "mean": float(arr[b].mean()),
                                "nonzero_fraction": float(np.count_nonzero(arr[b]) / arr[b].size),
                            }
                            for b in range(arr.shape[0])
                        ],
                        "normalized_band_stats": [
                            {"band": b + 1, "min": float(norm[b].min()), "max": float(norm[b].max()), "mean": float(norm[b].mean())}
                            for b in range(norm.shape[0])
                        ],
                    }
                    write_json(Path(f"{tile_prefix}_band_stats.json"), stats)
                    write_json(
                        Path(f"{tile_prefix}_model_input_shape.json"),
                        {"model_input_shape": [1, 4, args.tile_size, args.tile_size], "context": args.context, "center_size": args.tile_size - 2 * args.context},
                    )

                sample = torch.from_numpy(norm[None, ...]).to(device)
                with torch.no_grad():
                    logits = model(sample)
                    model_output_shape = list(logits.shape)
                    if tuple(logits.shape[-2:]) != tuple(arr.shape[-2:]):
                        logits = torch.nn.functional.interpolate(logits, size=arr.shape[-2:], mode="bilinear", align_corners=False)
                    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()[:actual_h, :actual_w]

                tile_rows.append(
                    {
                        **tile_feature_props,
                        "predicted": True,
                        "model_input_shape": list(sample.shape),
                        "model_output_shape": model_output_shape,
                        "prob_mean": float(prob.mean()),
                        "prob_max": float(prob.max()),
                    }
                )

                full_insert = full_insert_slices(x, y, actual_w, actual_h)
                center_insert = center_insert_slices(x, y, actual_w, actual_h, ds.width, ds.height, args.context)
                weights_full = weighted_window(actual_w, actual_h, x, y, ds.width, ds.height, args.context)

                for mode, spec in MODE_SPECS.items():
                    insert = full_insert if spec.full_tile else center_insert
                    prob_insert = prob[insert["crop_y0"] : insert["crop_y1"], insert["crop_x0"] : insert["crop_x1"]]
                    iy = insert["insert_y"]
                    ix = insert["insert_x"]
                    ih = insert["insert_height"]
                    iw = insert["insert_width"]
                    if spec.weighted:
                        if spec.full_tile:
                            weights = weights_full
                        else:
                            weights = weights_full[insert["crop_y0"] : insert["crop_y1"], insert["crop_x0"] : insert["crop_x1"]]
                        mode_sums[mode][iy : iy + ih, ix : ix + iw] += prob_insert * weights
                        mode_counts[mode][iy : iy + ih, ix : ix + iw] += weights
                    else:
                        mode_sums[mode][iy : iy + ih, ix : ix + iw] += prob_insert
                        mode_counts[mode][iy : iy + ih, ix : ix + iw] += 1.0
                    props = {
                        **tile_feature_props,
                        "mode": mode,
                        "insert_x": ix,
                        "insert_y": iy,
                        "insert_width": iw,
                        "insert_height": ih,
                        "crop_x0": insert["crop_x0"],
                        "crop_y0": insert["crop_y0"],
                        "crop_x1": insert["crop_x1"],
                        "crop_y1": insert["crop_y1"],
                    }
                    mode_insert_features[mode].append({"type": "Feature", "properties": props, "geometry": mapping(box(*ds.window_bounds(Window(ix, iy, iw, ih))))})
                seam_x.add(int(x))
                seam_x.add(int(x + actual_w))
                seam_y.add(int(y))
                seam_y.add(int(y + actual_h))

            summaries: list[dict[str, Any]] = []
            overview_cards = []
            selected_tiles = sorted([row for row in tile_rows if row.get("predicted")], key=lambda row: row.get("prob_max", 0), reverse=True)[:20]
            selected_tiles += [row for row in tile_rows if row.get("predicted")][:5]
            selected_tiles += [row for row in tile_rows if row.get("predicted")][-5:]
            selected_tiles = selected_tiles[:30]

            for mode in MODES:
                mode_dir = modes_root / mode
                prob = np.zeros((ds.height, ds.width), dtype="float32")
                coverage = mode_counts[mode] > 0
                prob[coverage] = mode_sums[mode][coverage] / np.maximum(mode_counts[mode][coverage], 1e-6)
                mask = (prob >= args.threshold).astype("uint8")
                raw_features, accepted_features, post_stats = vectorize_mask(mask, ds.transform, crs_name, args.min_area, args.simplify, args.max_objects)

                save_probability(mode_dir / "probability.tif", prob, profile)
                save_mask(mode_dir / "binary_mask_before_postprocess.tif", mask, profile)
                write_geojson(mode_dir / "polygons_before_postprocess.geojson", raw_features, crs_name)
                write_geojson(mode_dir / "polygons_after_postprocess.geojson", accepted_features, crs_name)
                shutil.copyfile(mode_dir / "polygons_after_postprocess.geojson", mode_dir / "accepted.geojson")
                write_geojson(mode_dir / "tile_grid.geojson", tile_grid_features, crs_name)
                write_geojson(mode_dir / "insert_regions.geojson", mode_insert_features[mode], crs_name)

                seam = seam_score(prob, sorted(seam_x), sorted(seam_y))
                write_json(mode_dir / "seam_metrics.json", seam)
                summary = {
                    "mode": mode,
                    "expected_windows": len(windows),
                    "predicted_windows": int(sum(1 for row in tile_rows if row.get("predicted"))),
                    "coverage_fraction": float(np.count_nonzero(coverage) / max(1, ds.width * ds.height)),
                    "mean_probability": float(prob[coverage].mean()) if np.any(coverage) else 0.0,
                    "mean_seam_jump": seam["mean_seam_jump"],
                    "max_seam_jump": seam["max_seam_jump"],
                    "geojson_size_before_mb": geojson_size_mb(mode_dir / "polygons_before_postprocess.geojson"),
                    "geojson_size_after_mb": geojson_size_mb(mode_dir / "polygons_after_postprocess.geojson"),
                    "vector_bounds_before": feature_union_bounds(raw_features),
                    "vector_bounds_after": feature_union_bounds(accepted_features),
                    **post_stats,
                }
                write_json(mode_dir / "mode_summary.json", summary)
                summaries.append(summary)

                scale = max(1, int(max(ds.width, ds.height) / 1600))
                base = nrg_preview(ds.read([1, 2, 3, 4], out_shape=(4, ds.height // scale, ds.width // scale)))
                prob_small = prob[::scale, ::scale]
                mask_small = mask[::scale, ::scale]
                heat = overlay_prob(base, prob_small)
                heat.save(mode_dir / "heatmap_preview.png")
                Image.fromarray((mask_small * 255).astype("uint8")).save(mode_dir / "binary_mask_preview.png")
                before_preview = draw_feature_overlay(base, add_scene_dims(raw_features.copy(), ds.width, ds.height), ds.transform, (255, 210, 0), 1)
                after_preview = draw_feature_overlay(base, add_scene_dims(accepted_features.copy(), ds.width, ds.height), ds.transform, (0, 160, 255), 2)
                before_preview.save(mode_dir / "polygons_before_preview.png")
                after_preview.save(mode_dir / "polygons_after_preview.png")
                overview_cards.append((mode, summary))

                tile_html_rows = []
                tile_mode_dir = mode_dir / "tiles"
                tile_mode_dir.mkdir(exist_ok=True)
                for n, row in enumerate(selected_tiles, start=1):
                    x, y, w, h = int(row["x"]), int(row["y"]), int(row["tile_width"]), int(row["tile_height"])
                    arr = ds.read([1, 2, 3, 4], window=Window(x, y, w, h), boundless=True, fill_value=0)
                    tile_base = nrg_preview(arr).resize((220, 220))
                    tile_heat = overlay_prob(tile_base, prob[y : y + h, x : x + w]).resize((220, 220))
                    tile_mask = Image.fromarray((mask[y : y + h, x : x + w] * 255).astype("uint8")).resize((220, 220))
                    grid = tile_base.copy().convert("RGBA")
                    draw = ImageDraw.Draw(grid)
                    draw.rectangle([0, 0, 219, 219], outline=(255, 255, 0, 255), width=3)
                    files = {}
                    for label, image in [("source", tile_base), ("heatmap", tile_heat), ("mask", tile_mask), ("grid", grid)]:
                        rel = tile_mode_dir / f"tile_{n:02d}_{label}.jpg"
                        image.convert("RGB").save(rel, quality=88)
                        files[label] = f"modes/{mode}/tiles/{rel.name}"
                    tile_html_rows.append(
                        f"<tr><td>{n}</td><td>{row['tile_index']}<br>x={x}, y={y}</td>"
                        f"<td><img src='{files['source']}'></td><td><img src='{files['heatmap']}'></td>"
                        f"<td><img src='{files['mask']}'></td><td><img src='{files['grid']}'></td></tr>"
                    )
                (mode_dir / "tile_table.html").write_text("\n".join(tile_html_rows), encoding="utf-8")
                del prob

            mode_summary = {"device": device_info, "image": Path(args.image).name, "checkpoint": str(args.checkpoint), "modes": summaries}
            write_json(debug_root / "mode_comparison_summary.json", mode_summary)

            reference_overlay_html = ""
            if args.reference and Path(args.reference).exists():
                reference_overlay_html = "<p>Reference server accepted.geojson was loaded for manual comparison: source/reference_accepted_scn08.geojson.</p>"

            summary_rows = "\n".join(
                "<tr>"
                f"<td>{s['mode']}</td><td>{s['predicted_windows']} / {s['expected_windows']}</td>"
                f"<td>{s['coverage_fraction']:.4f}</td><td>{s['object_count_before_postprocess']}</td>"
                f"<td>{s['object_count_after_postprocess']}</td><td>{s['total_area_before']:.0f}</td>"
                f"<td>{s['total_area_after']:.0f}</td><td>{s['geojson_size_before_mb']:.3f}</td>"
                f"<td>{s['geojson_size_after_mb']:.3f}</td><td>{s['mean_seam_jump']:.6f}</td>"
                f"<td>{s['max_seam_jump']:.6f}</td></tr>"
                for s in summaries
            )
            mode_sections = []
            for mode in MODES:
                tile_table = (modes_root / mode / "tile_table.html").read_text(encoding="utf-8")
                mode_sections.append(
                    f"<h2>{mode}</h2>"
                    f"<p><a href='modes/{mode}/probability.tif'>probability.tif</a> | "
                    f"<a href='modes/{mode}/binary_mask_before_postprocess.tif'>binary mask tif</a> | "
                    f"<a href='modes/{mode}/polygons_before_postprocess.geojson'>before polygons</a> | "
                    f"<a href='modes/{mode}/polygons_after_postprocess.geojson'>after polygons</a></p>"
                    f"<div class='previews'><div><b>heatmap</b><br><img src='modes/{mode}/heatmap_preview.png'></div>"
                    f"<div><b>binary mask</b><br><img src='modes/{mode}/binary_mask_preview.png'></div>"
                    f"<div><b>before polygons</b><br><img src='modes/{mode}/polygons_before_preview.png'></div>"
                    f"<div><b>after polygons</b><br><img src='modes/{mode}/polygons_after_preview.png'></div></div>"
                    f"<table><thead><tr><th>#</th><th>tile</th><th>source</th><th>heatmap</th><th>mask before post</th><th>grid/insert</th></tr></thead><tbody>{tile_table}</tbody></table>"
                )
            html = "\n".join(
                [
                    "<!doctype html><html><head><meta charset='utf-8'><title>SegFormer mode comparison</title>",
                    "<style>body{font-family:Arial,sans-serif;margin:18px} table{border-collapse:collapse;margin:12px 0} td,th{border:1px solid #ccc;padding:6px;vertical-align:top} img{max-width:260px}.previews{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:10px}.previews img{width:100%;max-width:380px}</style>",
                    "</head><body>",
                    "<h1>SegFormer pseudolabel stitching mode comparison</h1>",
                    f"<p><b>Scene:</b> {Path(args.image).name}<br><b>Checkpoint:</b> {Path(args.checkpoint).name}<br><b>Device:</b> {device_info}</p>",
                    reference_overlay_html,
                    "<h2>Summary</h2>",
                    "<table><thead><tr><th>mode</th><th>windows</th><th>coverage</th><th>objects before</th><th>objects after</th><th>area before</th><th>area after</th><th>before MB</th><th>after MB</th><th>mean seam</th><th>max seam</th></tr></thead><tbody>",
                    summary_rows,
                    "</tbody></table>",
                    *mode_sections,
                    "</body></html>",
                ]
            )
            (debug_root / "report_compare_modes.html").write_text(html, encoding="utf-8")

            best = sorted(summaries, key=lambda s: (s["mean_seam_jump"], abs(s["total_area_after"] - summaries[1]["total_area_after"])))[0]
            md = [
                "# SegFormer Pseudolabel Mode Debug",
                "",
                f"- image: `{Path(args.image).name}`",
                f"- checkpoint: `{Path(args.checkpoint).name}`",
                f"- device: `{device_info}`",
                f"- tile_size: {args.tile_size}",
                f"- stride: {args.stride}",
                f"- context: {args.context}",
                "",
                "## Modes",
                "",
            ]
            for s in summaries:
                md.append(
                    f"- `{s['mode']}`: coverage={s['coverage_fraction']:.4f}, objects_before={s['object_count_before_postprocess']}, "
                    f"objects_after={s['object_count_after_postprocess']}, area_after={s['total_area_after']:.0f}, "
                    f"mean_seam_jump={s['mean_seam_jump']:.6f}, after_mb={s['geojson_size_after_mb']:.3f}"
                )
            md.extend(
                [
                    "",
                    "## Finding",
                    "",
                    "The rectangular edges are visible before postprocess when inserted regions meet with hard boundaries. "
                    "The comparison separates probability stitching from postprocess: all modes use the same threshold, min area, simplify and top limit.",
                    "",
                    f"Current automatic best by seam score is `{best['mode']}`. This still requires manual visual review in `report_compare_modes.html`; it should not be accepted only because object count decreased.",
                    "",
                    "## Outputs",
                    "",
                    "- `report_compare_modes.html`",
                    "- `modes/<mode>/probability.tif`",
                    "- `modes/<mode>/binary_mask_before_postprocess.tif`",
                    "- `modes/<mode>/polygons_before_postprocess.geojson`",
                    "- `modes/<mode>/polygons_after_postprocess.geojson`",
                    "- `input_tiles/`",
                ]
            )
            (debug_root / "debug_summary.md").write_text("\n".join(md), encoding="utf-8")

    finally:
        for arr in list(mode_sums.values()) + list(mode_counts.values()):
            try:
                arr.flush()
            except Exception:
                pass

    print(f"wrote {debug_root / 'report_compare_modes.html'} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
