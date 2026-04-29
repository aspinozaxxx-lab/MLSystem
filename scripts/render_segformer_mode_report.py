from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window
from shapely.geometry import box, shape


MODES = [
    "hard_insert_full_tile",
    "hard_insert_center_crop",
    "weighted_overlap_full_tile",
    "weighted_overlap_center_crop",
]


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


def normalize_image(arr: np.ndarray) -> np.ndarray:
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


def heatmap_image(prob: np.ndarray) -> Image.Image:
    p = np.clip(prob, 0, 1)
    rgb = np.stack(
        [
            np.clip(2.0 * p, 0, 1),
            np.clip(2.0 * (1.0 - np.abs(p - 0.5) * 2.0), 0, 1),
            np.clip(2.0 * (1.0 - p), 0, 1),
        ],
        axis=-1,
    )
    return Image.fromarray((rgb * 255).astype("uint8"))


def overlay_prob(base: Image.Image, prob: np.ndarray, alpha: float = 0.48) -> Image.Image:
    base = base.convert("RGBA")
    hm = heatmap_image(prob).resize(base.size).convert("RGBA")
    hm.putalpha(int(255 * alpha))
    return Image.alpha_composite(base, hm).convert("RGB")


def read_geojson(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("features") or []


def draw_features(
    base: Image.Image,
    features: list[dict[str, Any]],
    transform: Any,
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    color: tuple[int, int, int],
    max_features: int = 1500,
) -> Image.Image:
    img = base.convert("RGBA")
    draw = ImageDraw.Draw(img)
    inv = ~transform
    tile_poly = None
    drawn = 0
    for feature in features:
        try:
            geom = shape(feature["geometry"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        if tile_poly is not None and not geom.intersects(tile_poly):
            continue
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in parts:
            coords = list(poly.exterior.coords)
            pts = []
            for gx, gy in coords:
                px, py = inv * (gx, gy)
                local_x = (px - tile_x) / max(1, tile_w) * base.width
                local_y = (py - tile_y) / max(1, tile_h) * base.height
                if -base.width <= local_x <= base.width * 2 and -base.height <= local_y <= base.height * 2:
                    pts.append((int(local_x), int(local_y)))
            if len(pts) > 1:
                draw.line(pts, fill=(*color, 230), width=2)
                drawn += 1
                if drawn >= max_features:
                    return img.convert("RGB")
    return img.convert("RGB")


def draw_full_scene_features(base: Image.Image, features: list[dict[str, Any]], transform: Any, scene_w: int, scene_h: int, color: tuple[int, int, int]) -> Image.Image:
    img = base.convert("RGBA")
    draw = ImageDraw.Draw(img)
    inv = ~transform
    for feature in features[:2000]:
        try:
            geom = shape(feature["geometry"])
        except Exception:
            continue
        parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
        for poly in parts:
            pts = []
            for gx, gy in poly.exterior.coords:
                px, py = inv * (gx, gy)
                pts.append((int(px / scene_w * base.width), int(py / scene_h * base.height)))
            if len(pts) > 1:
                draw.line(pts, fill=(*color, 230), width=2)
    return img


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


def select_tiles(ds: rasterio.DatasetReader, debug_root: Path, tile_size: int, stride: int, limit: int = 30) -> list[tuple[int, int, int]]:
    center_path = debug_root / "modes" / "hard_insert_center_crop" / "probability.tif"
    full_path = debug_root / "modes" / "weighted_overlap_full_tile" / "probability.tif"
    windows = window_grid(ds.width, ds.height, tile_size, stride)
    scored = []
    with rasterio.open(center_path) as center, rasterio.open(full_path) as full:
        for idx, (x, y) in enumerate(windows):
            w = min(tile_size, ds.width - x)
            h = min(tile_size, ds.height - y)
            win = Window(x, y, w, h)
            a = center.read(1, window=win)
            b = full.read(1, window=win)
            score = float(np.max(a) + np.max(b) + np.mean(np.abs(a - b)) * 2.0)
            if x == 0 or y == 0 or x + w >= ds.width or y + h >= ds.height:
                score += 0.2
            scored.append((score, idx, x, y))
    selected = sorted(scored, reverse=True)[: max(0, limit - 6)]
    selected += [(10.0, 0, windows[0][0], windows[0][1]), (10.0, len(windows) - 1, windows[-1][0], windows[-1][1])]
    seen = set()
    out = []
    for _, idx, x, y in selected:
        if idx in seen:
            continue
        seen.add(idx)
        out.append((idx, x, y))
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--debug-root", default=r"E:\Projects\debug")
    parser.add_argument("--reference", default="")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    args = parser.parse_args()

    debug_root = Path(args.debug_root)
    modes_root = debug_root / "modes"
    summaries = [json.loads((modes_root / mode / "mode_summary.json").read_text(encoding="utf-8")) for mode in MODES]
    device = json.loads((debug_root / "mode_comparison_summary.json").read_text(encoding="utf-8")).get("device", {})

    with rasterio.open(args.image) as ds:
        selected = select_tiles(ds, debug_root, args.tile_size, args.stride, limit=30)
        input_tiles = debug_root / "input_tiles"
        input_tiles.mkdir(exist_ok=True)
        for tile_idx, x, y in selected[:10]:
            w = min(args.tile_size, ds.width - x)
            h = min(args.tile_size, ds.height - y)
            arr = ds.read([1, 2, 3, 4], window=Window(x, y, w, h), boundless=True, fill_value=0)
            norm = normalize_image(arr)
            prefix = input_tiles / f"tile_{tile_idx:04d}"
            nrg_preview(arr).save(f"{prefix}_input_nrg.png")
            stats = {
                "tile_index": int(tile_idx),
                "x": int(x),
                "y": int(y),
                "tile_size": args.tile_size,
                "actual_width": int(w),
                "actual_height": int(h),
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
            (Path(f"{prefix}_band_stats.json")).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
            (Path(f"{prefix}_model_input_shape.json")).write_text(
                json.dumps(
                    {
                        "model_input_shape": [1, 4, args.tile_size, args.tile_size],
                        "input_channels": 4,
                        "input_bands": [1, 2, 3, 4],
                        "preview_bands": [4, 1, 2],
                        "context": 128,
                        "center_size": args.tile_size - 2 * 128,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        reference_features = read_geojson(Path(args.reference)) if args.reference else []
        old_features = read_geojson(modes_root / "hard_insert_center_crop" / "polygons_after_postprocess.geojson")
        best_features = read_geojson(modes_root / "weighted_overlap_full_tile" / "polygons_after_postprocess.geojson")
        scale = max(1, int(max(ds.width, ds.height) / 1600))
        overview_arr = ds.read([1, 2, 3, 4], out_shape=(4, ds.height // scale, ds.width // scale))
        overview = nrg_preview(overview_arr)
        cmp = draw_full_scene_features(overview, reference_features, ds.transform, ds.width, ds.height, (255, 0, 0))
        cmp = draw_full_scene_features(cmp, old_features, ds.transform, ds.width, ds.height, (255, 210, 0))
        cmp = draw_full_scene_features(cmp, best_features, ds.transform, ds.width, ds.height, (0, 180, 255))
        cmp.save(debug_root / "reference_old_best_overlay.png")

        mode_sections = []
        for mode in MODES:
            mode_dir = modes_root / mode
            raw_features = read_geojson(mode_dir / "polygons_before_postprocess.geojson")
            post_features = read_geojson(mode_dir / "polygons_after_postprocess.geojson")
            insert_features = read_geojson(mode_dir / "insert_regions.geojson")
            tile_dir = mode_dir / "rich_tiles"
            tile_dir.mkdir(exist_ok=True)
            tile_rows = []
            with rasterio.open(mode_dir / "probability.tif") as prob_ds, rasterio.open(mode_dir / "binary_mask_before_postprocess.tif") as mask_ds:
                for n, (tile_idx, x, y) in enumerate(selected, start=1):
                    w = min(args.tile_size, ds.width - x)
                    h = min(args.tile_size, ds.height - y)
                    win = Window(x, y, w, h)
                    arr = ds.read([1, 2, 3, 4], window=win, boundless=True, fill_value=0)
                    base = nrg_preview(arr).resize((220, 220))
                    prob = prob_ds.read(1, window=win)
                    mask = mask_ds.read(1, window=win)
                    heat = overlay_prob(base, prob).resize((220, 220))
                    mask_img = Image.fromarray((mask * 255).astype("uint8")).resize((220, 220)).convert("RGB")
                    before_poly = draw_features(base, raw_features, ds.transform, x, y, w, h, (255, 210, 0))
                    after_poly = draw_features(base, post_features, ds.transform, x, y, w, h, (0, 160, 255))
                    grid = base.copy().convert("RGBA")
                    draw = ImageDraw.Draw(grid)
                    draw.rectangle([0, 0, 219, 219], outline=(255, 255, 0, 255), width=3)
                    for feature in insert_features:
                        props = feature.get("properties") or {}
                        if int(props.get("tile_index", -1)) != int(tile_idx):
                            continue
                        ix = (int(props["insert_x"]) - x) / max(1, w) * 220
                        iy = (int(props["insert_y"]) - y) / max(1, h) * 220
                        iw = int(props["insert_width"]) / max(1, w) * 220
                        ih = int(props["insert_height"]) / max(1, h) * 220
                        draw.rectangle([ix, iy, ix + iw, iy + ih], outline=(255, 0, 0, 240), width=2)
                        break
                    files = {}
                    for label, image in [
                        ("source", base),
                        ("heatmap", heat),
                        ("mask", mask_img),
                        ("poly_before", before_poly),
                        ("poly_after", after_poly),
                        ("grid_insert", grid.convert("RGB")),
                    ]:
                        path = tile_dir / f"tile_{n:02d}_{label}.jpg"
                        image.save(path, quality=88)
                        files[label] = f"modes/{mode}/rich_tiles/{path.name}"
                    tile_rows.append(
                        f"<tr><td>{n}</td><td>{tile_idx}<br>x={x}, y={y}</td>"
                        f"<td><img src='{files['source']}'></td>"
                        f"<td><img src='{files['heatmap']}'></td>"
                        f"<td><img src='{files['mask']}'></td>"
                        f"<td><img src='{files['poly_before']}'></td>"
                        f"<td><img src='{files['poly_after']}'></td>"
                        f"<td><img src='{files['grid_insert']}'></td></tr>"
                    )
            mode_sections.append(
                f"<h2>{mode}</h2>"
                f"<p><a href='modes/{mode}/probability.tif'>probability.tif</a> | "
                f"<a href='modes/{mode}/binary_mask_before_postprocess.tif'>binary mask tif</a> | "
                f"<a href='modes/{mode}/polygons_before_postprocess.geojson'>polygons before postprocess</a> | "
                f"<a href='modes/{mode}/polygons_after_postprocess.geojson'>polygons after postprocess</a> | "
                f"<a href='modes/{mode}/insert_regions.geojson'>insert regions</a></p>"
                f"<div class='previews'><div><b>heatmap</b><br><img src='modes/{mode}/heatmap_preview.png'></div>"
                f"<div><b>binary mask</b><br><img src='modes/{mode}/binary_mask_preview.png'></div>"
                f"<div><b>polygons before</b><br><img src='modes/{mode}/polygons_before_preview.png'></div>"
                f"<div><b>polygons after</b><br><img src='modes/{mode}/polygons_after_preview.png'></div></div>"
                "<table><thead><tr><th>#</th><th>tile</th><th>source NRG</th><th>heatmap</th><th>binary before post</th><th>polygons before post</th><th>polygons after post</th><th>grid/insert</th></tr></thead><tbody>"
                + "\n".join(tile_rows)
                + "</tbody></table>"
            )

    summary_rows = "\n".join(
        "<tr>"
        f"<td>{s['mode']}</td><td>{s['predicted_windows']} / {s['expected_windows']}</td>"
        f"<td>{s['coverage_fraction']:.4f}</td><td>{s['object_count_before_postprocess']}</td>"
        f"<td>{s['object_count_after_postprocess']}</td><td>{s['total_area_before']:.0f}</td>"
        f"<td>{s['total_area_after']:.0f}</td><td>{s['geojson_size_before_mb']:.3f}</td>"
        f"<td>{s['geojson_size_after_mb']:.3f}</td><td>{s['mean_seam_jump']:.6f}</td>"
        f"<td>{s['max_seam_jump']:.6f}</td><td>{'center crop has rectangular insert regions' if 'center_crop' in s['mode'] else 'full tile stitching'}</td></tr>"
        for s in summaries
    )
    html = "\n".join(
        [
            "<!doctype html><html><head><meta charset='utf-8'><title>SegFormer rich mode comparison</title>",
            "<style>body{font-family:Arial,sans-serif;margin:18px} table{border-collapse:collapse;margin:12px 0} td,th{border:1px solid #ccc;padding:6px;vertical-align:top} img{max-width:220px}.previews{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:10px}.previews img{width:100%;max-width:380px}.legend span{display:inline-block;margin-right:18px}</style>",
            "</head><body>",
            "<h1>SegFormer pseudolabel stitching: heatmaps, masks and polygons</h1>",
            f"<p><b>Device:</b> {device}</p>",
            "<h2>Summary</h2>",
            "<table><thead><tr><th>mode</th><th>windows</th><th>coverage</th><th>objects before</th><th>objects after</th><th>area before</th><th>area after</th><th>before MB</th><th>after MB</th><th>mean seam</th><th>max seam</th><th>comment</th></tr></thead><tbody>",
            summary_rows,
            "</tbody></table>",
            "<h2>Reference vs local modes</h2>",
            "<p class='legend'><span style='color:red'>red: server accepted</span><span style='color:#c90'>yellow: old local center crop</span><span style='color:#08c'>blue: weighted full tile</span></p>",
            "<img src='reference_old_best_overlay.png' style='max-width:100%;width:1200px'>",
            *mode_sections,
            "</body></html>",
        ]
    )
    (debug_root / "report_compare_modes.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
