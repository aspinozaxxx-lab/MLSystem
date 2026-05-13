from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlsystem.src.data.virtual_tile_sampling import preview_from_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview virtual tile sampling records without training.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing dataset_manifest.json and dataset_annotation.geojson.")
    parser.add_argument("--dataset-manifest", default=None, help="Path to dataset_manifest.json.")
    parser.add_argument("--annotation", default=None, help="Path to GeoJSON annotation.")
    parser.add_argument("--images-dir", required=True, help="Directory with local GeoTIFF scenes.")
    parser.add_argument("--output-json", default=None, help="Optional lightweight JSON output path.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Limit train and val scenes for preview.")
    parser.add_argument("--max-records-preview", type=int, default=20, help="Number of sample records to include in preview.")
    parser.add_argument("--config-json", default=None, help="JSON file containing train_sampling config.")
    parser.add_argument("--config-yaml", default=None, help="YAML file containing train_sampling config.")
    parser.add_argument("--write-preview-overlays", default=None, help="Optional directory for debug PNG overlays.")
    parser.add_argument("--max-preview-overlays", type=int, default=20, help="Maximum overlay PNGs to write.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    dataset_manifest = Path(args.dataset_manifest) if args.dataset_manifest else (run_dir / "dataset_manifest.json" if run_dir else None)
    annotation = Path(args.annotation) if args.annotation else (run_dir / "dataset_annotation.geojson" if run_dir else None)
    if dataset_manifest is None:
        parser.error("--dataset-manifest is required when --run-dir is not provided")
    if annotation is None:
        parser.error("--annotation is required when --run-dir is not provided")
    if not dataset_manifest.exists():
        raise FileNotFoundError(f"dataset manifest does not exist: {dataset_manifest}")
    if not annotation.exists():
        raise FileNotFoundError(f"annotation does not exist: {annotation}")

    config = _load_config(args.config_json, args.config_yaml)
    payload = preview_from_manifest(
        dataset_manifest=dataset_manifest,
        annotation=annotation,
        images_dir=args.images_dir,
        config=config,
        max_scenes=args.max_scenes,
        max_records_preview=args.max_records_preview,
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if args.write_preview_overlays:
        written = _write_preview_overlays(
            payload.get("preview") or [],
            annotation,
            Path(args.write_preview_overlays),
            max_count=max(0, int(args.max_preview_overlays)),
        )
        payload["summary"]["preview_overlays_written"] = written
        if args.output_json:
            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2, default=str))
    return 0


def _load_config(config_json: str | None, config_yaml: str | None) -> dict[str, Any]:
    if config_json and config_yaml:
        raise ValueError("Use only one of --config-json or --config-yaml")
    if config_json:
        return json.loads(Path(config_json).read_text(encoding="utf-8"))
    if config_yaml:
        import yaml

        return yaml.safe_load(Path(config_yaml).read_text(encoding="utf-8")) or {}
    return {}


def _write_preview_overlays(records: list[dict[str, Any]], annotation_path: Path, output_dir: Path, *, max_count: int) -> int:
    if max_count <= 0:
        return 0
    try:
        import numpy as np
        import rasterio
        from PIL import Image, ImageDraw
        from rasterio.features import rasterize
        from rasterio.windows import Window
        from shapely.geometry import box, shape
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"preview overlays require rasterio, shapely, numpy, and Pillow: {exc}") from exc

    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    shapes = [shape(feature["geometry"]) for feature in annotation.get("features") or [] if feature.get("geometry")]
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, record in enumerate(records[:max_count]):
        image_path = record.get("source_image_path")
        window_info = record.get("window") or {}
        if not image_path or not window_info:
            continue
        x = int(window_info.get("x") or 0)
        y = int(window_info.get("y") or 0)
        width = int(window_info.get("width") or record.get("width") or 0)
        height = int(window_info.get("height") or record.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        with rasterio.open(str(image_path)) as ds:
            window = Window(x, y, width, height)
            band_count = min(3, int(ds.count))
            arr = ds.read(list(range(1, band_count + 1)), window=window, boundless=True, fill_value=0)
            if band_count == 1:
                rgb = np.repeat(arr[:1], 3, axis=0)
            elif band_count == 2:
                rgb = np.concatenate([arr, arr[:1]], axis=0)
            else:
                rgb = arr[:3]
            rgb = _normalize_rgb(rgb)
            mask = rasterize(
                [(geom, 1) for geom in shapes if geom.intersects(box(*ds.window_bounds(window)))],
                out_shape=(height, width),
                transform=ds.window_transform(window),
                fill=0,
                dtype="uint8",
            )
        image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB").convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_arr = np.asarray(overlay).copy()
        overlay_arr[mask > 0] = (255, 64, 64, 90)
        image = Image.alpha_composite(image, Image.fromarray(overlay_arr, mode="RGBA"))
        draw = ImageDraw.Draw(image)
        label = (
            f"{record.get('scene_id') or record.get('scene')} "
            f"x={x} y={y} {record.get('kind')} px={record.get('positive_pixels') or record.get('gt_positive_pixels')}"
        )
        draw.rectangle((0, 0, min(image.width, max(220, len(label) * 7)), 22), fill=(0, 0, 0, 180))
        draw.text((4, 4), label, fill=(255, 255, 255, 255))
        safe_scene = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(record.get("scene_id") or "scene"))
        image.convert("RGB").save(output_dir / f"{index:03d}_{safe_scene}_x{x}_y{y}_{record.get('kind')}.png")
        written += 1
    return written


def _normalize_rgb(arr: Any) -> Any:
    import numpy as np

    data = arr.astype("float32")
    result = np.zeros_like(data, dtype="uint8")
    for idx in range(data.shape[0]):
        band = data[idx]
        lo, hi = np.percentile(band, [2, 98])
        if hi <= lo:
            hi = float(band.max() or 1.0)
            lo = float(band.min())
        scaled = (band - lo) / max(1e-6, hi - lo)
        result[idx] = np.clip(scaled * 255.0, 0, 255).astype("uint8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
