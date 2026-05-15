from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlsystem.src.tile_preparation import SceneInput, TilePreparationFacade  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark tile_preparation DataLoader throughput without running a model.")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--train-scene-list", required=True)
    parser.add_argument("--val-scene-list", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--augmentation-level", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", default="0,2,4,8")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--normalization-mode", default="uint8_255", choices=["uint8_255", "tile_percentile", "scene_percentile"])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    images_root = Path(args.images_root)
    train_scenes = _read_scenes(Path(args.train_scene_list), images_root)
    val_scenes = _read_scenes(Path(args.val_scene_list), images_root)
    bundle = TilePreparationFacade.build_datasets(
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        annotation_path=Path(args.annotation),
        tile_size=args.tile_size,
        stride=args.stride,
        augmentation_level=args.augmentation_level,
        normalization_mode=args.normalization_mode,
    )
    try:
        worker_values = [int(item.strip()) for item in str(args.workers).split(",") if item.strip()]
        results = []
        for workers in worker_values:
            result = _benchmark_workers(
                bundle,
                batch_size=args.batch_size,
                workers=workers,
                prefetch_factor=args.prefetch_factor,
                batches=args.batches,
            )
            results.append(result)
            print(
                "workers={workers} batch_size={batch_size} samples/sec={samples_per_sec:.3f} "
                "batches/sec={batches_per_sec:.3f} median_batch_sec={median_batch_sec:.4f} p95_batch_sec={p95_batch_sec:.4f}".format(
                    **result
                ),
                flush=True,
            )

        report: dict[str, Any] = {
            "schema_version": 1,
            "config": {
                "images_root": str(images_root),
                "train_scene_list": str(args.train_scene_list),
                "val_scene_list": str(args.val_scene_list),
                "annotation": str(args.annotation),
                "tile_size": int(args.tile_size),
                "stride": int(args.stride),
                "augmentation_level": int(args.augmentation_level),
                "batch_size": int(args.batch_size),
                "prefetch_factor": int(args.prefetch_factor),
                "batches_requested": int(args.batches),
                "normalization_mode": args.normalization_mode,
            },
            "dataset": {
                "train_records": len(bundle.train_dataset),
                "val_records": len(bundle.val_dataset),
                **_record_counts(bundle.train_dataset.records),
                "skipped_fully_invalid_tiles": int(bundle.train_dataset.metadata.get("skipped_fully_invalid_tiles", 0)),
                "skipped_low_valid_share_tiles": int(bundle.train_dataset.metadata.get("skipped_low_valid_share_tiles", 0)),
                "warnings": bundle.warnings,
            },
            "results": results,
        }
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        bundle.train_dataset.close()
        bundle.val_dataset.close()


def _benchmark_workers(bundle: Any, *, batch_size: int, workers: int, prefetch_factor: int, batches: int) -> dict[str, Any]:
    loader = TilePreparationFacade.train_dataloader(
        bundle,
        batch_size=batch_size,
        workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=False,
    )
    batch_times: list[float] = []
    sample_count = 0
    batch_count = 0
    started = time.perf_counter()
    iterator = iter(loader)
    while batch_count < max(1, int(batches)):
        batch_started = time.perf_counter()
        try:
            _indices, _x, y = next(iterator)
        except StopIteration:
            break
        batch_times.append(time.perf_counter() - batch_started)
        sample_count += int(y.shape[0])
        batch_count += 1
    wall_sec = time.perf_counter() - started
    ordered = sorted(batch_times)
    return {
        "workers": int(workers),
        "batch_size": int(batch_size),
        "wall_sec": round(float(wall_sec), 6),
        "batches": int(batch_count),
        "samples": int(sample_count),
        "samples_per_sec": float(sample_count) / max(1e-9, wall_sec),
        "batches_per_sec": float(batch_count) / max(1e-9, wall_sec),
        "first_batch_sec": batch_times[0] if batch_times else 0.0,
        "median_batch_sec": statistics.median(ordered) if ordered else 0.0,
        "p95_batch_sec": _percentile(ordered, 0.95) if ordered else 0.0,
        "train_records": len(bundle.train_dataset),
        **_record_counts(bundle.train_dataset.records),
        "skipped_fully_invalid_tiles": int(bundle.train_dataset.metadata.get("skipped_fully_invalid_tiles", 0)),
        "skipped_low_valid_share_tiles": int(bundle.train_dataset.metadata.get("skipped_low_valid_share_tiles", 0)),
        "warnings": bundle.warnings,
    }


def _read_scenes(path: Path, images_root: Path) -> list[SceneInput]:
    scenes: list[SceneInput] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        image_path = _resolve_scene_path(line, images_root)
        scenes.append(SceneInput(image_path=image_path, scene_id=Path(line).stem))
    return scenes


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


def _resolve_scene_path(line: str, images_root: Path) -> Path:
    raw = Path(line)
    candidates = [raw] if raw.is_absolute() else [images_root / line]
    if raw.suffix.lower() not in {".tif", ".tiff"}:
        candidates.extend([images_root / f"{line}.tif", images_root / f"{line}.tiff"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


if __name__ == "__main__":
    main()
