from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.features import shapes as raster_shapes
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape


STITCH_MODES = [
    "hard_insert_full_tile",
    "hard_insert_center_crop",
    "weighted_overlap_full_tile",
    "weighted_overlap_center_crop",
]
NORM_MODES = ["per_tile_percentile", "scene_global_percentile"]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model() -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=4, classes=1, activation=None)


def choose_device(requested: str) -> tuple[torch.device, dict[str, Any]]:
    if requested in {"auto", "directml", "dml"}:
        try:
            import torch_directml

            device = torch_directml.device()
            return device, {"requested": requested, "backend": "directml", "device_name": str(device), "gpu_used": True}
        except Exception as exc:
            if requested != "auto":
                raise
            return torch.device("cpu"), {"requested": requested, "backend": "cpu", "device_name": "cpu", "gpu_used": False, "directml_error": f"{type(exc).__name__}: {exc}"}
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda"), {"requested": requested, "backend": "cuda", "device_name": torch.cuda.get_device_name(0), "gpu_used": True}
    return torch.device("cpu"), {"requested": requested, "backend": "cpu", "device_name": "cpu", "gpu_used": False}


def origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, max(1, length - tile + 1), max(1, stride)))
    edge = length - tile
    if values[-1] != edge:
        values.append(edge)
    return sorted(set(max(0, int(v)) for v in values))


def window_grid(width: int, height: int, tile: int, stride: int) -> list[tuple[int, int]]:
    return [(x, y) for y in origins(height, tile, stride) for x in origins(width, tile, stride)]


def center_insert(x: int, y: int, actual_w: int, actual_h: int, scene_w: int, scene_h: int, context: int) -> dict[str, int]:
    left = context if x > 0 else 0
    top = context if y > 0 else 0
    right = context if x + actual_w < scene_w else 0
    bottom = context if y + actual_h < scene_h else 0
    if actual_w - left - right <= 0:
        left = right = 0
    if actual_h - top - bottom <= 0:
        top = bottom = 0
    return {
        "crop_x0": int(left),
        "crop_y0": int(top),
        "crop_x1": int(actual_w - right),
        "crop_y1": int(actual_h - bottom),
        "insert_x": int(x + left),
        "insert_y": int(y + top),
        "insert_width": int(actual_w - left - right),
        "insert_height": int(actual_h - top - bottom),
    }


def full_insert(x: int, y: int, actual_w: int, actual_h: int) -> dict[str, int]:
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


def edge_weights(actual_w: int, actual_h: int, x: int, y: int, scene_w: int, scene_h: int, context: int) -> np.ndarray:
    wx = np.ones(actual_w, dtype="float32")
    wy = np.ones(actual_h, dtype="float32")
    ramp = max(1, int(context))
    if x > 0:
        n = min(ramp, actual_w)
        wx[:n] = np.linspace(0.05, 1.0, n, dtype="float32")
    if x + actual_w < scene_w:
        n = min(ramp, actual_w)
        wx[-n:] = np.minimum(wx[-n:], np.linspace(1.0, 0.05, n, dtype="float32"))
    if y > 0:
        n = min(ramp, actual_h)
        wy[:n] = np.linspace(0.05, 1.0, n, dtype="float32")
    if y + actual_h < scene_h:
        n = min(ramp, actual_h)
        wy[-n:] = np.minimum(wy[-n:], np.linspace(1.0, 0.05, n, dtype="float32"))
    return wy[:, None] * wx[None, :]


def nrg_preview(arr: np.ndarray) -> Image.Image:
    idx = [3, 0, 1] if arr.shape[0] >= 4 else [0, 1, 2]
    rgb = np.stack([arr[i] for i in idx], axis=-1).astype("float32")
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


def heatmap(prob: np.ndarray) -> Image.Image:
    p = np.clip(prob, 0, 1)
    rgb = np.stack([np.clip(2 * p, 0, 1), np.clip(2 * (1 - np.abs(p - 0.5) * 2), 0, 1), np.clip(2 * (1 - p), 0, 1)], axis=-1)
    return Image.fromarray((rgb * 255).astype("uint8"))


def overlay_heatmap(base: Image.Image, prob: np.ndarray) -> Image.Image:
    base = base.convert("RGBA")
    hm = heatmap(prob).resize(base.size).convert("RGBA")
    hm.putalpha(125)
    return Image.alpha_composite(base, hm).convert("RGB")


def compute_global_percentiles(ds: rasterio.DatasetReader, bands: list[int]) -> dict[str, Any]:
    if ds.dtypes[0] == "uint8":
        hists = np.zeros((len(bands), 256), dtype="int64")
        for _, window in ds.block_windows(1):
            arr = ds.read(bands, window=window)
            for i in range(arr.shape[0]):
                vals = arr[i].ravel()
                vals = vals[vals != 0]
                if vals.size:
                    hists[i] += np.bincount(vals, minlength=256)
        lohi = []
        for hist in hists:
            total = int(hist.sum())
            if total <= 0:
                lohi.append([0.0, 1.0])
                continue
            cdf = np.cumsum(hist)
            lo = int(np.searchsorted(cdf, total * 0.02))
            hi = int(np.searchsorted(cdf, total * 0.98))
            if hi <= lo:
                hi = lo + 1
            lohi.append([float(lo), float(hi)])
        return {"method": "global_uint8_histogram_p02_p98", "lo_hi": lohi}
    samples = []
    scale = max(1, int(max(ds.width, ds.height) / 4096))
    arr = ds.read(bands, out_shape=(len(bands), max(1, ds.height // scale), max(1, ds.width // scale)))
    for i in range(arr.shape[0]):
        vals = arr[i][arr[i] != 0]
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
        else:
            lo, hi = 0.0, 1.0
        if hi <= lo:
            hi = lo + 1
        samples.append([float(lo), float(hi)])
    return {"method": "global_downsample_p02_p98", "lo_hi": samples}


def normalize_per_tile(arr: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    arr = arr.astype("float32", copy=False)
    out = np.zeros_like(arr, dtype="float32")
    lohi = []
    for i in range(arr.shape[0]):
        vals = arr[i][arr[i] != 0]
        if vals.size < 16:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1
        out[i] = np.clip((arr[i] - lo) / (hi - lo), 0, 1)
        lohi.append([float(lo), float(hi)])
    return out, lohi


def normalize_global(arr: np.ndarray, lohi: list[list[float]]) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    out = np.zeros_like(arr, dtype="float32")
    for i, (lo, hi) in enumerate(lohi):
        if hi <= lo:
            hi = lo + 1
        out[i] = np.clip((arr[i] - lo) / (hi - lo), 0, 1)
    return out


def save_raster(path: Path, arr: np.ndarray, profile: dict[str, Any], dtype: str = "float32") -> None:
    profile = profile.copy()
    profile.pop("photometric", None)
    profile.update(count=1, dtype=dtype, compress="deflate", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype), 1)


def write_geojson(path: Path, features: list[dict[str, Any]], crs_name: str) -> None:
    write_json(path, {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs_name}}, "features": features})


def vectorize(mask: np.ndarray, transform: Any, min_area: float, simplify: float, max_objects: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    raw = []
    post = []
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
        raw.append({"type": "Feature", "properties": {"area_m2": float(poly.area)}, "geometry": mapping(poly)})
        if float(poly.area) < min_area:
            continue
        if simplify > 0:
            poly = poly.simplify(simplify, preserve_topology=True)
        if not poly.is_empty:
            post.append({"type": "Feature", "properties": {"area_m2": float(poly.area)}, "geometry": mapping(poly)})
    post.sort(key=lambda f: float(f["properties"].get("area_m2") or 0), reverse=True)
    accepted = post[:max_objects]
    return raw, accepted, {
        "object_count_before_postprocess": len(raw),
        "object_count_after_postprocess": len(accepted),
        "objects_after_filter_before_top": len(post),
        "total_area_before": float(sum(float(f["properties"].get("area_m2") or 0) for f in raw)),
        "total_area_after": float(sum(float(f["properties"].get("area_m2") or 0) for f in accepted)),
    }


def seam_metrics(prob: np.ndarray, xs: list[int], ys: list[int], sample: int = 8) -> dict[str, float]:
    vals = []
    for x in xs:
        if 1 <= x < prob.shape[1]:
            vals.append(float(np.mean(np.abs(prob[::sample, x] - prob[::sample, x - 1]))))
    for y in ys:
        if 1 <= y < prob.shape[0]:
            vals.append(float(np.mean(np.abs(prob[y, ::sample] - prob[y - 1, ::sample]))))
    return {"mean_seam_jump": float(np.mean(vals)) if vals else 0.0, "max_seam_jump": float(np.max(vals)) if vals else 0.0}


def select_tile_indices(tile_rows: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    predicted = [r for r in tile_rows if r.get("predicted")]
    ranked = sorted(predicted, key=lambda r: (float(r.get("prob_max", 0)), float(r.get("prob_mean", 0))), reverse=True)
    selected = ranked[: max(0, limit - 6)] + predicted[:3] + predicted[-3:]
    seen = set()
    out = []
    for row in selected:
        if row["tile_index"] in seen:
            continue
        seen.add(row["tile_index"])
        out.append(row)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
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

    root = Path(args.debug_root)
    out_root = root / "stitching_internals"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    raw_root = out_root / "raw_tiles"
    raw_root.mkdir()
    work_root = out_root / "work"
    work_root.mkdir()

    device, device_info = choose_device(args.device)
    model = build_model().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()

    summaries = []
    started = time.time()
    with rasterio.open(args.image) as ds:
        profile = ds.profile
        crs_name = ds.crs.to_string() if ds.crs else ""
        windows = window_grid(ds.width, ds.height, args.tile_size, args.stride)
        global_norm = compute_global_percentiles(ds, [1, 2, 3, 4])
        tile_grid_features = [
            {
                "type": "Feature",
                "properties": {"tile_index": i, "x": int(x), "y": int(y), "tile_width": int(min(args.tile_size, ds.width - x)), "tile_height": int(min(args.tile_size, ds.height - y))},
                "geometry": mapping(box(*ds.window_bounds(Window(x, y, args.tile_size, args.tile_size)))),
            }
            for i, (x, y) in enumerate(windows)
        ]
        seam_x = sorted(set([x for x, _ in windows] + [min(ds.width, x + args.tile_size) for x, _ in windows]))
        seam_y = sorted(set([y for _, y in windows] + [min(ds.height, y + args.tile_size) for _, y in windows]))

        for norm_mode in NORM_MODES:
            norm_root = out_root / norm_mode
            norm_root.mkdir()
            arrays = {}
            for mode in STITCH_MODES:
                arrays[mode] = {
                    "sum": np.memmap(work_root / f"{norm_mode}_{mode}_sum.dat", dtype="float32", mode="w+", shape=(ds.height, ds.width)),
                    "count": np.memmap(work_root / f"{norm_mode}_{mode}_count.dat", dtype="uint16", mode="w+", shape=(ds.height, ds.width)),
                    "weight": np.memmap(work_root / f"{norm_mode}_{mode}_weight.dat", dtype="float32", mode="w+", shape=(ds.height, ds.width)),
                    "inserts": [],
                }
                arrays[mode]["sum"][:] = 0
                arrays[mode]["count"][:] = 0
                arrays[mode]["weight"][:] = 0
            tile_rows = []
            selected_raw_written = 0
            for tile_index, (x, y) in enumerate(windows):
                actual_w = min(args.tile_size, ds.width - x)
                actual_h = min(args.tile_size, ds.height - y)
                win = Window(x, y, args.tile_size, args.tile_size)
                arr = ds.read([1, 2, 3, 4], window=win, boundless=True, fill_value=0)
                if np.count_nonzero(arr) == 0:
                    tile_rows.append({"tile_index": tile_index, "x": int(x), "y": int(y), "predicted": False, "skipped_reason": "all_zero"})
                    continue
                if norm_mode == "per_tile_percentile":
                    norm, lohi = normalize_per_tile(arr)
                else:
                    norm = normalize_global(arr, global_norm["lo_hi"])
                    lohi = global_norm["lo_hi"]
                sample = torch.from_numpy(norm[None, ...]).to(device)
                with torch.no_grad():
                    logits = model(sample)
                    input_shape = list(sample.shape)
                    output_shape = list(logits.shape)
                    if tuple(logits.shape[-2:]) != tuple(arr.shape[-2:]):
                        logits = torch.nn.functional.interpolate(logits, size=arr.shape[-2:], mode="bilinear", align_corners=False)
                    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()[:actual_h, :actual_w]
                tile_rows.append(
                    {
                        "tile_index": tile_index,
                        "x": int(x),
                        "y": int(y),
                        "tile_width": int(actual_w),
                        "tile_height": int(actual_h),
                        "predicted": True,
                        "prob_mean": float(prob.mean()),
                        "prob_max": float(prob.max()),
                        "normalization": norm_mode,
                        "normalization_lo_hi": lohi,
                        "model_input_shape": input_shape,
                        "model_output_shape": output_shape,
                    }
                )
                if selected_raw_written < 30 and (prob.max() > args.threshold or tile_index in {0, len(windows) // 2, len(windows) - 1}):
                    selected_raw_written += 1
                    raw_dir = raw_root / norm_mode
                    raw_dir.mkdir(exist_ok=True)
                    prefix = raw_dir / f"tile_{tile_index:04d}"
                    np.savez_compressed(prefix.with_suffix(".probability.npz"), probability=prob.astype("float32"), x=x, y=y, tile_width=actual_w, tile_height=actual_h)
                    nrg_preview(arr).save(f"{prefix}.input_nrg.png")
                    heatmap(prob).save(f"{prefix}.raw_heatmap.png")
                    write_json(
                        Path(f"{prefix}.stats.json"),
                        {
                            "tile_index": tile_index,
                            "x": int(x),
                            "y": int(y),
                            "tile_width": int(actual_w),
                            "tile_height": int(actual_h),
                            "input_bands": [1, 2, 3, 4],
                            "preview_bands": [4, 1, 2],
                            "normalization": norm_mode,
                            "normalization_lo_hi": lohi,
                            "model_input_shape": input_shape,
                            "model_output_shape": output_shape,
                            "prob_mean": float(prob.mean()),
                            "prob_max": float(prob.max()),
                        },
                    )
                inserts = {
                    "full": full_insert(x, y, actual_w, actual_h),
                    "center": center_insert(x, y, actual_w, actual_h, ds.width, ds.height, args.context),
                }
                weights_full = edge_weights(actual_w, actual_h, x, y, ds.width, ds.height, args.context)
                for mode in STITCH_MODES:
                    use_center = mode.endswith("center_crop")
                    use_weight = mode.startswith("weighted")
                    insert = inserts["center" if use_center else "full"]
                    crop = prob[insert["crop_y0"] : insert["crop_y1"], insert["crop_x0"] : insert["crop_x1"]]
                    iy, ix = insert["insert_y"], insert["insert_x"]
                    ih, iw = insert["insert_height"], insert["insert_width"]
                    if use_weight:
                        w = weights_full if not use_center else weights_full[insert["crop_y0"] : insert["crop_y1"], insert["crop_x0"] : insert["crop_x1"]]
                    else:
                        w = np.ones_like(crop, dtype="float32")
                    arrays[mode]["sum"][iy : iy + ih, ix : ix + iw] += crop * w
                    arrays[mode]["weight"][iy : iy + ih, ix : ix + iw] += w
                    arrays[mode]["count"][iy : iy + ih, ix : ix + iw] += 1
                    props = {
                        "tile_index": tile_index,
                        "mode": mode,
                        "normalization": norm_mode,
                        "x": int(x),
                        "y": int(y),
                        "tile_width": int(actual_w),
                        "tile_height": int(actual_h),
                        **insert,
                    }
                    arrays[mode]["inserts"].append({"type": "Feature", "properties": props, "geometry": mapping(box(*ds.window_bounds(Window(ix, iy, iw, ih))))})
            write_json(norm_root / "tile_predictions.json", tile_rows)
            write_geojson(norm_root / "tile_grid.geojson", tile_grid_features, crs_name)
            selected_tiles = select_tile_indices(tile_rows, 30)
            for mode in STITCH_MODES:
                mode_dir = norm_root / mode
                mode_dir.mkdir()
                sum_arr = np.asarray(arrays[mode]["sum"])
                weight_arr = np.asarray(arrays[mode]["weight"])
                count_arr = np.asarray(arrays[mode]["count"])
                avg = np.zeros((ds.height, ds.width), dtype="float32")
                valid = weight_arr > 0
                avg[valid] = sum_arr[valid] / np.maximum(weight_arr[valid], 1e-6)
                mask = (avg >= args.threshold).astype("uint8")
                raw_features, accepted_features, stats = vectorize(mask, ds.transform, args.min_area, args.simplify, args.max_objects)
                save_raster(mode_dir / "prob_sum.tif", sum_arr, profile, "float32")
                save_raster(mode_dir / "prob_count.tif", count_arr, profile, "uint16")
                save_raster(mode_dir / "weight_sum.tif", weight_arr, profile, "float32")
                save_raster(mode_dir / "prob_avg.tif", avg, profile, "float32")
                shutil.copyfile(mode_dir / "prob_avg.tif", mode_dir / "probability.tif")
                save_raster(mode_dir / "binary_mask_before_postprocess.tif", mask, profile, "uint8")
                write_geojson(mode_dir / "tile_grid.geojson", tile_grid_features, crs_name)
                write_geojson(mode_dir / "insert_regions.geojson", arrays[mode]["inserts"], crs_name)
                write_geojson(mode_dir / "polygons_before_postprocess.geojson", raw_features, crs_name)
                write_geojson(mode_dir / "polygons_after_postprocess.geojson", accepted_features, crs_name)
                shutil.copyfile(mode_dir / "polygons_after_postprocess.geojson", mode_dir / "accepted.geojson")
                seam = seam_metrics(avg, seam_x, seam_y)
                write_json(mode_dir / "seam_metrics.json", seam)
                mode_summary = {
                    "normalization": norm_mode,
                    "mode": mode,
                    "tile_size": args.tile_size,
                    "stride": args.stride,
                    "context": args.context,
                    "expected_windows": len(windows),
                    "predicted_windows": int(sum(1 for r in tile_rows if r.get("predicted"))),
                    "coverage_fraction": float(np.count_nonzero(valid) / max(1, ds.width * ds.height)),
                    "mean_probability": float(avg[valid].mean()) if np.any(valid) else 0.0,
                    "geojson_size_before_mb": (mode_dir / "polygons_before_postprocess.geojson").stat().st_size / (1024 * 1024),
                    "geojson_size_after_mb": (mode_dir / "polygons_after_postprocess.geojson").stat().st_size / (1024 * 1024),
                    **stats,
                    **seam,
                }
                write_json(mode_dir / "mode_summary.json", mode_summary)
                summaries.append(mode_summary)
                scale = max(1, int(max(ds.width, ds.height) / 1600))
                base = nrg_preview(ds.read([1, 2, 3, 4], out_shape=(4, ds.height // scale, ds.width // scale)))
                overlay_heatmap(base, avg[::scale, ::scale]).save(mode_dir / "heatmap_preview.png")
                Image.fromarray((mask[::scale, ::scale] * 255).astype("uint8")).save(mode_dir / "binary_mask_preview.png")

                tile_dir = mode_dir / "tiles"
                tile_dir.mkdir()
                rows = []
                with rasterio.open(mode_dir / "prob_avg.tif") as avg_ds, rasterio.open(mode_dir / "binary_mask_before_postprocess.tif") as mask_ds:
                    for n, row in enumerate(selected_tiles, start=1):
                        x, y = int(row["x"]), int(row["y"])
                        tw, th = int(row["tile_width"]), int(row["tile_height"])
                        arr = ds.read([1, 2, 3, 4], window=Window(x, y, tw, th), boundless=True, fill_value=0)
                        base_tile = nrg_preview(arr).resize((220, 220))
                        prob_tile = avg_ds.read(1, window=Window(x, y, tw, th))
                        mask_tile = mask_ds.read(1, window=Window(x, y, tw, th))
                        heat = overlay_heatmap(base_tile, prob_tile).resize((220, 220))
                        mask_img = Image.fromarray((mask_tile * 255).astype("uint8")).resize((220, 220)).convert("RGB")
                        raw_tile_heat = raw_root / norm_mode / f"tile_{int(row['tile_index']):04d}.raw_heatmap.png"
                        if not raw_tile_heat.exists():
                            raw_tile_heat = ""
                        grid = base_tile.copy().convert("RGBA")
                        draw = ImageDraw.Draw(grid)
                        draw.rectangle([0, 0, 219, 219], outline=(255, 255, 0, 255), width=3)
                        for feature in arrays[mode]["inserts"]:
                            props = feature["properties"]
                            if props["tile_index"] != row["tile_index"]:
                                continue
                            ix = (props["insert_x"] - x) / max(1, tw) * 220
                            iy = (props["insert_y"] - y) / max(1, th) * 220
                            iw = props["insert_width"] / max(1, tw) * 220
                            ih = props["insert_height"] / max(1, th) * 220
                            draw.rectangle([ix, iy, ix + iw, iy + ih], outline=(255, 0, 0, 240), width=2)
                            break
                        files = {}
                        for label, image in [("input", base_tile), ("stitched_heat", heat), ("mask", mask_img), ("insert", grid.convert("RGB"))]:
                            path = tile_dir / f"tile_{n:02d}_{label}.jpg"
                            image.save(path, quality=88)
                            files[label] = path.name
                        rows.append(
                            {
                                "n": n,
                                "tile_index": int(row["tile_index"]),
                                "x": x,
                                "y": y,
                                "raw_tile_heatmap": str(raw_tile_heat),
                                "files": files,
                            }
                        )
                write_json(tile_dir / "tile_rows.json", rows)
                del avg

    write_json(
        out_root / "stitching_summary.json",
        {
            "device": device_info,
            "global_normalization": global_norm,
            "image": Path(args.image).name,
            "checkpoint": str(args.checkpoint),
            "duration_sec": round(time.time() - started, 3),
            "summaries": summaries,
        },
    )

    summary_rows = "\n".join(
        "<tr>"
        f"<td>{s['normalization']}</td><td>{s['mode']}</td><td>{s['predicted_windows']} / {s['expected_windows']}</td>"
        f"<td>{s['coverage_fraction']:.4f}</td><td>{s['object_count_before_postprocess']}</td><td>{s['object_count_after_postprocess']}</td>"
        f"<td>{s['total_area_before']:.0f}</td><td>{s['total_area_after']:.0f}</td><td>{s['mean_seam_jump']:.6f}</td><td>{s['max_seam_jump']:.6f}</td>"
        f"<td>{s['geojson_size_before_mb']:.3f}</td><td>{s['geojson_size_after_mb']:.3f}</td></tr>"
        for s in summaries
    )
    sections = []
    for norm_mode in NORM_MODES:
        for mode in STITCH_MODES:
            mode_dir = out_root / norm_mode / mode
            rows = json.loads((mode_dir / "tiles" / "tile_rows.json").read_text(encoding="utf-8"))
            tile_html = []
            for row in rows:
                files = row["files"]
                raw_rel = ""
                if row.get("raw_tile_heatmap"):
                    raw_path = Path(row["raw_tile_heatmap"])
                    raw_rel = raw_path.relative_to(out_root).as_posix()
                raw_html = f"<img src='{raw_rel}'>" if raw_rel else "not saved"
                tile_html.append(
                    f"<tr><td>{row['n']}</td><td>{row['tile_index']}<br>x={row['x']} y={row['y']}</td>"
                    f"<td><img src='{norm_mode}/{mode}/tiles/{files['input']}'></td>"
                    f"<td>{raw_html}</td>"
                    f"<td><img src='{norm_mode}/{mode}/tiles/{files['insert']}'></td>"
                    f"<td><img src='{norm_mode}/{mode}/tiles/{files['stitched_heat']}'></td>"
                    f"<td><img src='{norm_mode}/{mode}/tiles/{files['mask']}'></td>"
                    f"<td><a href='{norm_mode}/{mode}/polygons_before_postprocess.geojson'>geojson</a></td>"
                    f"<td><a href='{norm_mode}/{mode}/polygons_after_postprocess.geojson'>geojson</a></td></tr>"
                )
            sections.append(
                f"<h2>{norm_mode} / {mode}</h2>"
                f"<p><a href='{norm_mode}/{mode}/prob_sum.tif'>prob_sum</a> | <a href='{norm_mode}/{mode}/prob_count.tif'>prob_count</a> | "
                f"<a href='{norm_mode}/{mode}/weight_sum.tif'>weight_sum</a> | <a href='{norm_mode}/{mode}/prob_avg.tif'>prob_avg</a> | "
                f"<a href='{norm_mode}/{mode}/tile_grid.geojson'>tile_grid</a> | <a href='{norm_mode}/{mode}/insert_regions.geojson'>insert_regions</a></p>"
                f"<p><img src='{norm_mode}/{mode}/heatmap_preview.png' style='max-width:700px'> <img src='{norm_mode}/{mode}/binary_mask_preview.png' style='max-width:700px'></p>"
                "<table><thead><tr><th>#</th><th>tile</th><th>input tile</th><th>raw tile heatmap</th><th>inserted region</th><th>stitched heatmap fragment</th><th>binary mask</th><th>polygons before</th><th>polygons after</th></tr></thead><tbody>"
                + "\n".join(tile_html)
                + "</tbody></table>"
            )
    html = "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'><title>SegFormer stitching internals</title>",
            "<style>body{font-family:Arial,sans-serif;margin:18px} table{border-collapse:collapse;margin:12px 0} td,th{border:1px solid #ccc;padding:6px;vertical-align:top} img{max-width:220px}</style>",
            "</head><body>",
            "<h1>SegFormer stitching internals and normalization comparison</h1>",
            f"<p><b>Device:</b> {device_info}</p>",
            f"<p><b>Global normalization:</b> {global_norm}</p>",
            "<table><thead><tr><th>normalization</th><th>mode</th><th>windows</th><th>coverage</th><th>objects before</th><th>objects after</th><th>area before</th><th>area after</th><th>mean seam</th><th>max seam</th><th>before MB</th><th>after MB</th></tr></thead><tbody>",
            summary_rows,
            "</tbody></table>",
            *sections,
            "</body></html>",
        ]
    )
    (out_root / "report_stitching_internals.html").write_text(html, encoding="utf-8")
    print(f"wrote {out_root / 'report_stitching_internals.html'} in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
