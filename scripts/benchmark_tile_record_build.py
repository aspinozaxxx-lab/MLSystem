from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlsystem.src.tile_preparation.api import build_datasets  # noqa: E402
from mlsystem.src.tile_preparation.contracts import SceneInputContract as SceneInput, TileDatasetRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark tile_preparation record build with footprint filtering.")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--train-scene-list", required=True)
    parser.add_argument("--val-scene-list", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--augmentation-level", type=int, default=1)
    parser.add_argument("--record-workers", default="0,1,2,4")
    parser.add_argument("--normalization-mode", default="uint8_255", choices=["uint8_255", "tile_percentile", "scene_percentile"])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    images_root = Path(args.images_root)
    train_scenes = _read_scenes(Path(args.train_scene_list), images_root)
    val_scenes = _read_scenes(Path(args.val_scene_list), images_root)
    worker_values = [int(item.strip()) for item in str(args.record_workers).split(",") if item.strip()]
    if not worker_values:
        worker_values = [0]
    old_env = os.environ.get("MLSYSTEM_TILE_RECORD_WORKERS")
    results: list[dict[str, Any]] = []
    try:
        for workers in worker_values:
            os.environ["MLSYSTEM_TILE_RECORD_WORKERS"] = str(workers)
            started = time.perf_counter()
            bundle = build_datasets(
                TileDatasetRequest(
                    train_scenes=train_scenes,
                    val_scenes=val_scenes,
                    annotation_path=Path(args.annotation),
                    tile_size=int(args.tile_size),
                    stride=int(args.stride),
                    augmentation_level=int(args.augmentation_level),
                    normalization_mode=str(args.normalization_mode),
                )
            )
            total_sec = time.perf_counter() - started
            try:
                result = _result_row(workers, total_sec, bundle)
                results.append(result)
                print(
                    "record_workers={record_workers} total_sec={total_sec:.3f} train_records={train_records} "
                    "kept_windows={windows_intersecting_footprint} skipped_outside={skipped_outside_footprint}".format(**result),
                    flush=True,
                )
            finally:
                bundle.train_dataset.close()
                bundle.val_dataset.close()
    finally:
        if old_env is None:
            os.environ.pop("MLSYSTEM_TILE_RECORD_WORKERS", None)
        else:
            os.environ["MLSYSTEM_TILE_RECORD_WORKERS"] = old_env

    report = {
        "schema_version": 1,
        "config": {
            "images_root": str(images_root),
            "train_scene_list": str(args.train_scene_list),
            "val_scene_list": str(args.val_scene_list),
            "annotation": str(args.annotation),
            "tile_size": int(args.tile_size),
            "stride": int(args.stride),
            "augmentation_level": int(args.augmentation_level),
            "record_workers": str(args.record_workers),
            "normalization_mode": str(args.normalization_mode),
        },
        "results": results,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _result_row(workers: int, total_sec: float, bundle: Any) -> dict[str, Any]:
    train_meta = bundle.train_dataset.metadata
    val_meta = bundle.val_dataset.metadata
    train_records = len(bundle.train_dataset.records)
    val_records = len(bundle.val_dataset.records)
    counts = _record_counts(bundle.train_dataset.records)
    return {
        "record_workers": int(workers),
        "workers": int(workers),
        "scenes": int(len(bundle.train_dataset.scenes) + len(bundle.val_dataset.scenes)),
        "scenes_count": int(len(bundle.train_dataset.scenes) + len(bundle.val_dataset.scenes)),
        "train_scene_count": int(len(bundle.train_dataset.scenes)),
        "val_scene_count": int(len(bundle.val_dataset.scenes)),
        "train_records": int(train_records),
        "val_records": int(val_records),
        "total_sec": float(total_sec),
        "train_build_sec": float(train_meta.get("build_records_sec", 0.0) or 0.0),
        "val_build_sec": float(val_meta.get("build_records_sec", 0.0) or 0.0),
        "train_build_footprint_sec": float(train_meta.get("build_footprint_sec", 0.0) or 0.0),
        "val_build_footprint_sec": float(val_meta.get("build_footprint_sec", 0.0) or 0.0),
        "build_footprint_sec": float(train_meta.get("build_footprint_sec", 0.0) or 0.0) + float(val_meta.get("build_footprint_sec", 0.0) or 0.0),
        "footprint_build_sec_total": float(train_meta.get("build_footprint_sec", 0.0) or 0.0) + float(val_meta.get("build_footprint_sec", 0.0) or 0.0),
        "footprint_build_sec_mean": (
            (float(train_meta.get("build_footprint_sec", 0.0) or 0.0) + float(val_meta.get("build_footprint_sec", 0.0) or 0.0))
            / max(1, int(len(bundle.train_dataset.scenes) + len(bundle.val_dataset.scenes)))
        ),
        "build_records_sec": float(train_meta.get("build_records_sec", 0.0) or 0.0) + float(val_meta.get("build_records_sec", 0.0) or 0.0),
        "records_per_sec": float(train_records + val_records) / max(1e-9, float(total_sec)),
        "candidate_windows_rectangular": _meta_sum(train_meta, val_meta, "candidate_windows_rectangular"),
        "windows_intersecting_footprint": _meta_sum(train_meta, val_meta, "windows_intersecting_footprint"),
        "kept_footprint_windows": _meta_sum(train_meta, val_meta, "windows_intersecting_footprint"),
        "skipped_outside_footprint": _meta_sum(train_meta, val_meta, "skipped_outside_footprint"),
        "boundary_windows": _meta_sum(train_meta, val_meta, "boundary_footprint_windows"),
        "fully_inside_windows": _meta_sum(train_meta, val_meta, "fully_inside_footprint_windows"),
        "read_valid_mask_calls": _meta_sum(train_meta, val_meta, "read_valid_mask_calls"),
        "warning_count": int(len(bundle.warnings)),
        **counts,
    }


def _meta_sum(left: dict[str, Any], right: dict[str, Any], key: str) -> int:
    return int(left.get(key, 0) or 0) + int(right.get(key, 0) or 0)


def _record_counts(records: list[Any]) -> dict[str, int]:
    positive = sum(1 for record in records if record.kind == "positive")
    partial_positive = sum(1 for record in records if record.kind == "partial_positive")
    hard_negative = sum(1 for record in records if record.kind == "hard_negative")
    negative = sum(1 for record in records if record.kind == "negative")
    return {
        "positive_records": int(positive + partial_positive),
        "strict_positive_records": int(positive),
        "partial_positive_records": int(partial_positive),
        "hard_negative_records": int(hard_negative),
        "negative_records": int(negative),
    }


def _read_scenes(path: Path, images_root: Path) -> list[SceneInput]:
    scenes: list[SceneInput] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        image_path = _resolve_scene_path(line, images_root)
        scenes.append(SceneInput(image_path, Path(line).stem))
    return scenes


def _resolve_scene_path(line: str, images_root: Path) -> Path:
    raw = Path(line)
    candidates = [raw] if raw.is_absolute() else [images_root / line]
    if raw.suffix.lower() not in {".tif", ".tiff"}:
        candidates.extend([images_root / f"{line}.tif", images_root / f"{line}.tiff"])
        candidates.extend(sorted(images_root.glob(f"{line}*.tif")))
        candidates.extend(sorted(images_root.glob(f"{line}*.tiff")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


if __name__ == "__main__":
    raise SystemExit(main())
