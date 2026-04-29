from __future__ import annotations

import argparse
import json
import math
import time
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw
from rasterio.features import shapes as raster_shapes
from rasterio.windows import Window
from shapely.geometry import box, mapping, shape

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _build_model(model_name: str, in_channels: int, out_channels: int, base_channels: int) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    if model_name.lower() == "segformer_b0":
        return smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=in_channels, classes=out_channels, activation=None)
    raise ValueError(f"Local debug script supports only segformer_b0, got {model_name}")


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
        "crop_x0": crop_left,
        "crop_y0": crop_top,
        "crop_x1": actual_w - crop_right,
        "crop_y1": actual_h - crop_bottom,
        "insert_x": x + crop_left,
        "insert_y": y + crop_top,
        "insert_width": actual_w - crop_left - crop_right,
        "insert_height": actual_h - crop_top - crop_bottom,
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


def vectorize(prob: np.ndarray, transform: Any, threshold: float, min_area: float, simplify: float, max_objects: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask = (prob >= threshold).astype("uint8")
    raw = 0
    after_filter = 0
    vertices_before = 0
    vertices_after = 0
    features: list[dict[str, Any]] = []
    for geom, value in raster_shapes(mask, mask=mask.astype(bool), transform=transform):
        if value != 1:
            continue
        poly = shape(geom)
        if poly.is_empty:
            continue
        raw += 1
        vertices_before += len(mapping(poly).get("coordinates", [[]])[0]) if poly.geom_type == "Polygon" else 0
        if float(poly.area) < min_area:
            continue
        after_filter += 1
        if simplify > 0:
            poly = poly.simplify(simplify, preserve_topology=True)
        mapped = mapping(poly)
        vertices_after += len(mapped.get("coordinates", [[]])[0]) if mapped.get("type") == "Polygon" else 0
        features.append({"type": "Feature", "properties": {"area_m2": float(poly.area), "threshold": threshold}, "geometry": mapped})
    features.sort(key=lambda f: float(f["properties"].get("area_m2") or 0), reverse=True)
    after_top = min(len(features), max_objects)
    return features[:max_objects], {
        "threshold": threshold,
        "min_area_m2": min_area,
        "simplify_tolerance": simplify,
        "objects_before_filter": raw,
        "objects_after_filter": after_filter,
        "objects_after_top500": after_top,
        "vertices_before": vertices_before,
        "vertices_after": vertices_after,
        "max_objects": max_objects,
    }


def write_geojson(path: Path, features: list[dict[str, Any]], crs_name: str = "urn:ogc:def:crs:EPSG::3857") -> None:
    payload = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs_name}}, "features": features}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def save_probability(path: Path, prob: np.ndarray, profile: dict[str, Any]) -> None:
    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", compress="deflate", predictor=3, tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(prob.astype("float32"), 1)


def nrg_preview(arr: np.ndarray) -> Image.Image:
    # NRG preview: bands [4,1,2] if available; arr is [bands,h,w].
    indices = [3, 0, 1] if arr.shape[0] >= 4 else [0, 1, 2]
    rgb = np.stack([arr[i] for i in indices], axis=-1).astype("float32")
    out = np.zeros_like(rgb, dtype="uint8")
    for c in range(3):
        band = rgb[:, :, c]
        lo, hi = np.percentile(band[band > 0], [2, 98]) if np.any(band > 0) else (0, 1)
        if hi <= lo:
            hi = lo + 1
        out[:, :, c] = np.clip((band - lo) / (hi - lo) * 255, 0, 255).astype("uint8")
    return Image.fromarray(out)


def overlay_mask(base: Image.Image, prob: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = base.convert("RGBA")
    mask = Image.fromarray(np.clip(prob * 255, 0, 255).astype("uint8")).resize(base.size)
    layer = Image.new("RGBA", base.size, (*color, 0))
    layer.putalpha(mask.point(lambda v: int(v * alpha)))
    return Image.alpha_composite(base, layer)


def seam_score(prob: np.ndarray, xs: list[int], ys: list[int]) -> dict[str, float]:
    vals = []
    for x in xs:
        if 1 <= x < prob.shape[1]:
            vals.append(float(np.mean(np.abs(prob[:, x] - prob[:, x - 1]))))
    for y in ys:
        if 1 <= y < prob.shape[0]:
            vals.append(float(np.mean(np.abs(prob[y, :] - prob[y - 1, :]))))
    return {"mean_abs_jump": float(np.mean(vals)) if vals else 0.0, "max_abs_jump": float(np.max(vals)) if vals else 0.0}


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
    parser.add_argument("--max-tiles", type=int, default=0, help="0 means full scene")
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    before_dir = debug_root / "before"
    after_dir = debug_root / "after"
    work_dir = debug_root / "work"
    for d in (before_dir, after_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = _build_model("segformer_b0", 4, 1, 8).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint)
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    started = time.time()
    tile_rows = []
    seam_x_before: set[int] = set()
    seam_y_before: set[int] = set()
    with rasterio.open(args.image) as ds:
        windows = window_grid(ds.width, ds.height, args.tile_size, args.stride)
        if args.max_tiles and args.max_tiles > 0:
            windows = windows[: args.max_tiles]
        profile = ds.profile
        profile.pop("photometric", None)
        prob_sum_before = np.zeros((ds.height, ds.width), dtype="float32")
        prob_count_before = np.zeros((ds.height, ds.width), dtype="uint16")
        prob_sum_after = np.zeros((ds.height, ds.width), dtype="float32")
        weight_sum_after = np.zeros((ds.height, ds.width), dtype="float32")
        source_preview_arr = ds.read([1, 2, 3, 4], out_shape=(4, min(ds.height, 2200), min(ds.width, 1800)))
        source_preview = nrg_preview(source_preview_arr)
        source_preview.save(work_dir / "scene_nrg_preview.jpg", quality=90)

        for idx, (x, y) in enumerate(windows):
            win_started = time.time()
            window = Window(x, y, args.tile_size, args.tile_size)
            actual_w = min(args.tile_size, ds.width - x)
            actual_h = min(args.tile_size, ds.height - y)
            arr = ds.read([1, 2, 3, 4], window=window, boundless=True, fill_value=0)
            if np.count_nonzero(arr) == 0:
                tile_rows.append({"tile_index": idx, "x": x, "y": y, "predicted": False, "reason": "all_zero"})
                continue
            sample = torch.from_numpy(_normalize_image(arr)[None, ...]).to(device)
            with torch.no_grad():
                logits = model(sample)
                if tuple(logits.shape[-2:]) != tuple(arr.shape[-2:]):
                    logits = torch.nn.functional.interpolate(logits, size=arr.shape[-2:], mode="bilinear", align_corners=False)
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()[:actual_h, :actual_w]

            insert = center_insert_slices(x, y, actual_w, actual_h, ds.width, ds.height, args.context)
            ix, iy, iw, ih = insert["insert_x"], insert["insert_y"], insert["insert_width"], insert["insert_height"]
            prob_insert = prob[insert["crop_y0"] : insert["crop_y1"], insert["crop_x0"] : insert["crop_x1"]]
            prob_sum_before[iy : iy + ih, ix : ix + iw] += prob_insert
            prob_count_before[iy : iy + ih, ix : ix + iw] += 1
            seam_x_before.add(ix)
            seam_x_before.add(ix + iw)
            seam_y_before.add(iy)
            seam_y_before.add(iy + ih)

            weights = weighted_window(actual_w, actual_h, x, y, ds.width, ds.height, args.context)
            prob_sum_after[y : y + actual_h, x : x + actual_w] += prob * weights
            weight_sum_after[y : y + actual_h, x : x + actual_w] += weights

            tile_rows.append(
                {
                    "tile_index": idx,
                    "x": x,
                    "y": y,
                    "width": actual_w,
                    "height": actual_h,
                    "predicted": True,
                    "before_insert": insert,
                    "duration_sec": round(time.time() - win_started, 3),
                    "prob_mean": float(prob.mean()),
                    "prob_max": float(prob.max()),
                }
            )

        before_mask = prob_count_before > 0
        after_mask = weight_sum_after > 0
        prob_before = np.zeros_like(prob_sum_before)
        prob_after = np.zeros_like(prob_sum_after)
        prob_before[before_mask] = prob_sum_before[before_mask] / np.maximum(prob_count_before[before_mask], 1)
        prob_after[after_mask] = prob_sum_after[after_mask] / np.maximum(weight_sum_after[after_mask], 1e-6)

        before_features, before_post = vectorize(prob_before, ds.transform, args.threshold, args.min_area, args.simplify, args.max_objects)
        after_features, after_post = vectorize(prob_after, ds.transform, args.threshold, args.min_area, args.simplify, args.max_objects)

        write_geojson(before_dir / "accepted_before.geojson", before_features)
        write_geojson(after_dir / "accepted_after.geojson", after_features)
        save_probability(before_dir / "probability_before.tif", prob_before, profile)
        save_probability(after_dir / "probability_after.tif", prob_after, profile)
        Image.fromarray((np.clip(prob_before, 0, 1)[::8, ::8] * 255).astype("uint8")).save(before_dir / "probability_before_preview.png")
        Image.fromarray((np.clip(prob_after, 0, 1)[::8, ::8] * 255).astype("uint8")).save(after_dir / "probability_after_preview.png")
        overlay_mask(source_preview, prob_before[:: max(1, ds.height // source_preview.height), :: max(1, ds.width // source_preview.width)], (255, 0, 0)).save(before_dir / "whole_scene_before_overlay.png")
        overlay_mask(source_preview, prob_after[:: max(1, ds.height // source_preview.height), :: max(1, ds.width // source_preview.width)], (0, 120, 255)).save(after_dir / "whole_scene_after_overlay.png")

        # Pick 30 informative tiles: high before/after difference first, then seam/positive regions.
        scored_tiles = []
        for row in tile_rows:
            if not row.get("predicted"):
                continue
            x, y, w, h = row["x"], row["y"], row["width"], row["height"]
            diff = float(np.mean(np.abs(prob_after[y:y+h, x:x+w] - prob_before[y:y+h, x:x+w])))
            score = diff + 0.25 * float(np.max(prob_before[y:y+h, x:x+w])) + 0.25 * float(np.max(prob_after[y:y+h, x:x+w]))
            scored_tiles.append((score, diff, row))
        selected = [row for _, _, row in sorted(scored_tiles, key=lambda item: item[0], reverse=True)[:30]]
        thumb_rows = []
        for n, row in enumerate(selected, start=1):
            x, y, w, h = row["x"], row["y"], row["width"], row["height"]
            arr = ds.read([1, 2, 3, 4], window=Window(x, y, w, h), boundless=True, fill_value=0)
            base = nrg_preview(arr).resize((220, 220))
            before_img = overlay_mask(base, prob_before[y:y+h, x:x+w], (255, 0, 0)).resize((220, 220))
            after_img = overlay_mask(base, prob_after[y:y+h, x:x+w], (0, 120, 255)).resize((220, 220))
            diff_img = Image.fromarray((np.clip(np.abs(prob_after[y:y+h, x:x+w] - prob_before[y:y+h, x:x+w]) * 255 * 3, 0, 255)).astype("uint8")).resize((220, 220))
            grid = base.copy().convert("RGBA")
            draw = ImageDraw.Draw(grid)
            draw.rectangle([0, 0, 219, 219], outline=(255, 255, 0, 255), width=3)
            files = {}
            for label, img in [("source", base), ("grid", grid), ("before", before_img), ("after", after_img), ("diff", diff_img)]:
                f = work_dir / f"tile_{n:02d}_{label}.jpg"
                img.convert("RGB").save(f, quality=88)
                files[label] = f.name
            thumb_rows.append({"n": n, "tile_index": row["tile_index"], "x": x, "y": y, "diff_mean": float(np.mean(np.asarray(diff_img))), "files": files})

        before_post["geojson_size_mb"] = (before_dir / "accepted_before.geojson").stat().st_size / (1024 * 1024)
        after_post["geojson_size_mb"] = (after_dir / "accepted_after.geojson").stat().st_size / (1024 * 1024)
        before_debug = {
            "mode": "before_center_crop_hard_insert",
            "scene": Path(args.image).name,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "context": args.context,
            "expected_window_count": len(window_grid(ds.width, ds.height, args.tile_size, args.stride)),
            "actual_window_count": len(windows),
            "predicted_window_count": int(sum(1 for r in tile_rows if r.get("predicted"))),
            "coverage_fraction": float(np.count_nonzero(before_mask) / max(1, ds.width * ds.height)),
            "seam_score": seam_score(prob_before, sorted(seam_x_before), sorted(seam_y_before)),
            "postprocess": before_post,
            "tiles": tile_rows,
        }
        after_debug = {
            "mode": "after_weighted_overlap_full_tile",
            "scene": Path(args.image).name,
            "tile_size": args.tile_size,
            "stride": args.stride,
            "context": args.context,
            "expected_window_count": len(window_grid(ds.width, ds.height, args.tile_size, args.stride)),
            "actual_window_count": len(windows),
            "predicted_window_count": int(sum(1 for r in tile_rows if r.get("predicted"))),
            "coverage_fraction": float(np.count_nonzero(after_mask) / max(1, ds.width * ds.height)),
            "seam_score": seam_score(prob_after, sorted(seam_x_before), sorted(seam_y_before)),
            "postprocess": after_post,
            "tiles": tile_rows,
        }
        (before_dir / "tiling_debug_before.json").write_text(json.dumps(before_debug, ensure_ascii=False, indent=2), encoding="utf-8")
        (after_dir / "tiling_debug_after.json").write_text(json.dumps(after_debug, ensure_ascii=False, indent=2), encoding="utf-8")
        (before_dir / "postprocess_debug_before.json").write_text(json.dumps(before_post, ensure_ascii=False, indent=2), encoding="utf-8")
        (after_dir / "postprocess_debug_after.json").write_text(json.dumps(after_post, ensure_ascii=False, indent=2), encoding="utf-8")

    report_rows = "\n".join(
        "<tr>"
        f"<td>{r['n']}<br>tile {r['tile_index']}<br>x={r['x']} y={r['y']}</td>"
        f"<td><img src='work/{r['files']['source']}'></td>"
        f"<td><img src='work/{r['files']['grid']}'></td>"
        f"<td><img src='work/{r['files']['before']}'></td>"
        f"<td><img src='work/{r['files']['after']}'></td>"
        f"<td><img src='work/{r['files']['diff']}'><br>diff={r['diff_mean']:.3f}</td>"
        "</tr>"
        for r in thumb_rows
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Pseudolabel stitching debug</title>
<style>body{{font-family:Arial,sans-serif;margin:20px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ddd;padding:6px;vertical-align:top}} img{{max-width:220px}} .big img{{max-width:48%;margin:6px;border:1px solid #ccc}}</style>
</head><body>
<h1>Pseudolabel stitching debug: before vs after</h1>
<h2>Краткий итог</h2>
<ul>
<li>Scene: <code>{Path(args.image).name}</code></li>
<li>Checkpoint: <code>{Path(args.checkpoint).name}</code></li>
<li>Параметры: tile={args.tile_size}, stride={args.stride}, context={args.context}, threshold={args.threshold}, min_area={args.min_area}, simplify={args.simplify}, max_objects={args.max_objects}</li>
<li>Ошибка: жесткая center-crop вставка SegFormer создает разрывы/скачки вероятности на границах вставленных областей. Векторизация затем превращает эти скачки в искусственные горизонтальные/вертикальные отсечения.</li>
<li>Исправление: использовать взвешенное overlap-stitching полного tile prediction с плавными весами на краях вместо hard center crop.</li>
</ul>
<h2>Summary</h2>
<table><tr><th>metric</th><th>before</th><th>after</th></tr>
<tr><td>coverage</td><td>{before_debug['coverage_fraction']:.6f}</td><td>{after_debug['coverage_fraction']:.6f}</td></tr>
<tr><td>predicted windows</td><td>{before_debug['predicted_window_count']}</td><td>{after_debug['predicted_window_count']}</td></tr>
<tr><td>objects after top</td><td>{before_post['objects_after_top500']}</td><td>{after_post['objects_after_top500']}</td></tr>
<tr><td>geojson MB</td><td>{before_post['geojson_size_mb']:.3f}</td><td>{after_post['geojson_size_mb']:.3f}</td></tr>
<tr><td>mean seam abs jump</td><td>{before_debug['seam_score']['mean_abs_jump']:.6f}</td><td>{after_debug['seam_score']['mean_abs_jump']:.6f}</td></tr>
<tr><td>max seam abs jump</td><td>{before_debug['seam_score']['max_abs_jump']:.6f}</td><td>{after_debug['seam_score']['max_abs_jump']:.6f}</td></tr>
</table>
<h2>Whole scene previews</h2>
<div class="big">
<img src="before/whole_scene_before_overlay.png"><img src="after/whole_scene_after_overlay.png">
<br><img src="before/probability_before_preview.png"><img src="after/probability_after_preview.png">
</div>
<h2>30 tile comparisons</h2>
<table><tr><th>#</th><th>source</th><th>tile grid</th><th>before</th><th>after</th><th>difference</th></tr>
{report_rows}
</table>
<p>Runtime: {time.time() - started:.1f} sec</p>
</body></html>"""
    (debug_root / "report_compare_before_after.html").write_text(html, encoding="utf-8")
    summary = f"""# Pseudolabel Stitching Debug

## Downloaded source

- checkpoint: `{args.checkpoint}`
- image: `{args.image}`
- reference accepted: `E:\\Projects\\debug\\source\\reference_accepted.geojson`

## Reproduction

The local script runs SegFormer B0 on one scene with tile={args.tile_size}, stride={args.stride}, context={args.context}, threshold={args.threshold}.

## Root cause

The current SegFormer pseudolabel stitching uses hard center-crop insertion. Adjacent inserted regions meet without overlap/blending, so probability jumps at insert boundaries. Thresholding/vectorization turns these jumps into artificial horizontal/vertical cuts.

## Local fix

Use weighted overlap-stitching over full tile predictions. Edge pixels receive lower weight for interior tiles, overlapping predictions are averaged smoothly.

## Results

- before accepted: `E:\\Projects\\debug\\before\\accepted_before.geojson`
- after accepted: `E:\\Projects\\debug\\after\\accepted_after.geojson`
- before probability: `E:\\Projects\\debug\\before\\probability_before.tif`
- after probability: `E:\\Projects\\debug\\after\\probability_after.tif`
- HTML report: `E:\\Projects\\debug\\report_compare_before_after.html`

## Key numbers

- before coverage: {before_debug['coverage_fraction']:.6f}
- after coverage: {after_debug['coverage_fraction']:.6f}
- before objects: {before_post['objects_after_top500']}
- after objects: {after_post['objects_after_top500']}
- before geojson MB: {before_post['geojson_size_mb']:.3f}
- after geojson MB: {after_post['geojson_size_mb']:.3f}
- before mean seam jump: {before_debug['seam_score']['mean_abs_jump']:.6f}
- after mean seam jump: {after_debug['seam_score']['mean_abs_jump']:.6f}

## Ready for push/deploy

The local code path is ready to port into `mlsystem/src/real_train.py` as the SegFormer default stitching mode after visual review.
"""
    (debug_root / "debug_summary.md").write_text(summary, encoding="utf-8")
    print(debug_root / "report_compare_before_after.html")
    print(debug_root / "debug_summary.md")


if __name__ == "__main__":
    main()
