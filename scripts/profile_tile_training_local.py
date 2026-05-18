from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mlsystem.src.tile_preparation.api import build_datasets  # noqa: E402
from mlsystem.src.tile_preparation.contracts import SceneInputContract as SceneInput, TileDatasetRequest  # noqa: E402
from mlsystem.src.tile_preparation.dataloader import make_tile_dataloader, tile_collate_with_metadata_fn  # noqa: E402


TILE_FUNCTION_KEYS = [
    "read_valid_mask_sec",
    "read_image_sec",
    "normalize_sec",
    "geometry_index_query_sec",
    "rasterize_burn_sec",
    "valid_clip_sec",
    "rasterize_sec",
    "augmentation_sec",
    "mosaic_sec",
    "total_getitem_sec",
]

STEP_KEYS = [
    "batch_wait_sec",
    "cpu_to_gpu_sec",
    "forward_sec",
    "loss_sec",
    "backward_sec",
    "optimizer_step_sec",
    "step_total_sec",
]


class TinyUNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1, base: int = 8) -> None:
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, base, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base, base, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.down = torch.nn.MaxPool2d(2)
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(base, base * 2, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base * 2, base * 2, 3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.up = torch.nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec = torch.nn.Sequential(
            torch.nn.Conv2d(base * 2, base, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(base, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc1(x)
        x = self.enc2(self.down(skip))
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec(torch.cat([x, skip], dim=1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Short local profiler for tile_preparation + one training step.")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--train-scene-list", required=True)
    parser.add_argument("--val-scene-list", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--augmentation-level", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", default="0,2")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-runtime-sec", type=float, default=45.0)
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--normalization-mode", default="uint8_255", choices=["uint8_255", "tile_percentile", "scene_percentile"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    os.environ["MLSYSTEM_TILE_PREP_PROFILE"] = "1"
    requested_device = str(args.device).lower()
    if requested_device == "cuda" and not torch.cuda.is_available():
        actual_device = torch.device("cpu")
        cuda_fallback_reason = "torch.cuda.is_available() is false"
    else:
        actual_device = torch.device(requested_device)
        cuda_fallback_reason = None

    images_root = Path(args.images_root)
    train_scenes = _read_scenes(Path(args.train_scene_list), images_root)
    val_scenes = _read_scenes(Path(args.val_scene_list), images_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_started = time.perf_counter()
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
    build_datasets_sec = time.perf_counter() - build_started
    try:
        worker_values = [int(item.strip()) for item in str(args.workers).split(",") if item.strip()]
        if not worker_values:
            worker_values = [0]
        deadline = time.perf_counter() + max(1.0, float(args.max_runtime_sec))
        batch_rows: list[dict[str, Any]] = []
        function_rows: list[dict[str, Any]] = []
        runs: list[dict[str, Any]] = []
        for workers in worker_values:
            remaining = deadline - time.perf_counter()
            if remaining <= 1.0:
                break
            run = _profile_workers(
                bundle,
                workers=workers,
                batch_size=int(args.batch_size),
                prefetch_factor=int(args.prefetch_factor),
                max_batches=int(args.max_batches),
                max_runtime_sec=remaining,
                warmup_batches=int(args.warmup_batches),
                device=actual_device,
            )
            runs.append(run["summary"])
            batch_rows.extend(run["batch_rows"])
            function_rows.extend(run["function_rows"])
            print(
                "workers={workers} device={device} batches={batches} "
                "batch_wait_mean={batch_wait_mean_sec:.4f}s step_mean={step_total_mean_sec:.4f}s "
                "forward_mean={forward_mean_sec:.4f}s backward_mean={backward_mean_sec:.4f}s".format(**run["summary"]),
                flush=True,
            )

        summary = _build_summary(
            args=args,
            bundle=bundle,
            requested_device=requested_device,
            actual_device=str(actual_device),
            cuda_fallback_reason=cuda_fallback_reason,
            runs=runs,
            batch_rows=batch_rows,
            function_rows=function_rows,
            build_datasets_sec=build_datasets_sec,
        )
        _write_outputs(output_dir, summary, batch_rows, function_rows)
    finally:
        bundle.train_dataset.close()
        bundle.val_dataset.close()
    return 0


def _profile_workers(
    bundle: Any,
    *,
    workers: int,
    batch_size: int,
    prefetch_factor: int,
    max_batches: int,
    max_runtime_sec: float,
    warmup_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    bundle.train_dataset.set_epoch(1)
    loader = make_tile_dataloader(
        bundle.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=int(bundle.config.seed) + 1,
        workers=workers,
        prefetch_factor=prefetch_factor,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        collate_fn=tile_collate_with_metadata_fn,
    )
    iterator = iter(loader)
    model = TinyUNet(in_channels=_input_channels(bundle), base=8).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(max(0, int(warmup_batches))):
        try:
            _indices, x_cpu, y_cpu, _metadata = next(iterator)
        except StopIteration:
            break
        _run_training_step(model, optimizer, loss_fn, x_cpu, y_cpu, device, record_timing=False)

    rows: list[dict[str, Any]] = []
    function_values: dict[str, list[float]] = {key: [] for key in TILE_FUNCTION_KEYS}
    measured_started = time.perf_counter()
    batch_index = 0
    while batch_index < max(1, int(max_batches)) and (time.perf_counter() - measured_started) < max_runtime_sec:
        wait_started = time.perf_counter()
        try:
            indices, x_cpu, y_cpu, metadata = next(iterator)
        except StopIteration:
            break
        batch_wait_sec = time.perf_counter() - wait_started
        row = _run_training_step(model, optimizer, loss_fn, x_cpu, y_cpu, device, record_timing=True)
        row["gpu_step_sec"] = float(row.get("step_total_sec", 0.0) or 0.0)
        row["step_total_sec"] = float(row["gpu_step_sec"]) + float(batch_wait_sec)
        row.update(
            {
                "workers": int(workers),
                "batch_index": int(batch_index),
                "batch_wait_sec": float(batch_wait_sec),
                "batch_size": int(y_cpu.shape[0]),
                "sample_indices": ";".join(str(item) for item in indices),
                "x_shape": "x".join(str(item) for item in x_cpu.shape),
                "y_shape": "x".join(str(item) for item in y_cpu.shape),
                "y_positive_pixels": int(torch.count_nonzero(y_cpu).item()),
            }
        )
        _accumulate_tile_profiles(metadata, function_values)
        rows.append(row)
        batch_index += 1

    function_rows = _function_rows(workers, function_values)
    summary = _run_summary(workers, str(device), rows, function_rows)
    return {"summary": summary, "batch_rows": rows, "function_rows": function_rows}


def _run_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    x_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    device: torch.device,
    *,
    record_timing: bool,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    step_started = time.perf_counter()

    def timed(name: str, func: Any) -> Any:
        _sync(device)
        started = time.perf_counter()
        result = func()
        _sync(device)
        if record_timing:
            timings[name] = time.perf_counter() - started
        return result

    x, y = timed("cpu_to_gpu_sec", lambda: (x_cpu.to(device, non_blocking=device.type == "cuda"), y_cpu.to(device, non_blocking=device.type == "cuda")))
    optimizer.zero_grad(set_to_none=True)
    logits = timed("forward_sec", lambda: model(x))
    loss = timed("loss_sec", lambda: loss_fn(logits, y))
    timed("backward_sec", lambda: loss.backward())
    timed("optimizer_step_sec", optimizer.step)
    _sync(device)
    if record_timing:
        timings["step_total_sec"] = time.perf_counter() - step_started
        timings["loss_value"] = float(loss.detach().cpu().item())
        timings.update(_gpu_memory(device))
    return timings


def _input_channels(bundle: Any) -> int:
    if bundle.train_dataset.records:
        sample = bundle.train_dataset[0]
        return int(sample.image.shape[0])
    configured = bundle.config.input_bands
    return len(configured) if configured else 4


def _accumulate_tile_profiles(metadata: list[dict[str, Any]], function_values: dict[str, list[float]]) -> None:
    for item in metadata:
        profile = item.get("tile_prep_profile") if isinstance(item, dict) else None
        if not isinstance(profile, dict):
            continue
        for key in TILE_FUNCTION_KEYS:
            function_values[key].append(float(profile.get(key, 0.0) or 0.0))


def _function_rows(workers: int, values: dict[str, list[float]]) -> list[dict[str, Any]]:
    total_getitem = sum(values.get("total_getitem_sec", []))
    rows: list[dict[str, Any]] = []
    for key in TILE_FUNCTION_KEYS:
        samples = values.get(key, [])
        stats = _stats(samples)
        rows.append(
            {
                "workers": int(workers),
                "function": key.removesuffix("_sec"),
                "count": int(len(samples)),
                "mean_sample_sec": stats["mean"],
                "median_sample_sec": stats["median"],
                "p95_sample_sec": stats["p95"],
                "max_sample_sec": stats["max"],
                "total_sec": stats["total"],
                "percent_of_getitem": (stats["total"] / total_getitem * 100.0) if total_getitem > 0 else 0.0,
            }
        )
    return rows


def _run_summary(workers: int, device: str, rows: list[dict[str, Any]], function_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "workers": int(workers),
        "device": device,
        "batches": len(rows),
        "samples": sum(int(row.get("batch_size", 0)) for row in rows),
    }
    for key in STEP_KEYS:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        stats = _stats(values)
        clean = key.removesuffix("_sec")
        summary[f"{clean}_mean_sec"] = stats["mean"]
        summary[f"{clean}_median_sec"] = stats["median"]
        summary[f"{clean}_p95_sec"] = stats["p95"]
        summary[f"{clean}_max_sec"] = stats["max"]
    step_mean = float(summary.get("step_total_mean_sec", 0.0) or 0.0)
    compute_mean = sum(float(summary.get(f"{key.removesuffix('_sec')}_mean_sec", 0.0) or 0.0) for key in ["forward_sec", "loss_sec", "backward_sec", "optimizer_step_sec"])
    summary["gpu_compute_mean_sec"] = compute_mean
    summary["batches_per_sec"] = (len(rows) / sum(float(row.get("step_total_sec", 0.0) or 0.0) for row in rows)) if rows else 0.0
    summary["samples_per_sec"] = (summary["samples"] / sum(float(row.get("step_total_sec", 0.0) or 0.0) for row in rows)) if rows else 0.0
    summary["bottleneck"] = _diagnose(summary, function_rows, step_mean)
    return summary


def _build_summary(
    *,
    args: argparse.Namespace,
    bundle: Any,
    requested_device: str,
    actual_device: str,
    cuda_fallback_reason: str | None,
    runs: list[dict[str, Any]],
    batch_rows: list[dict[str, Any]],
    function_rows: list[dict[str, Any]],
    build_datasets_sec: float,
) -> dict[str, Any]:
    combined = _run_summary(-1, actual_device, batch_rows, function_rows)
    return {
        "schema_version": 1,
        "requested_device": requested_device,
        "actual_device": actual_device,
        "cuda_fallback_reason": cuda_fallback_reason,
        "config": {
            "images_root": str(args.images_root),
            "train_scene_list": str(args.train_scene_list),
            "val_scene_list": str(args.val_scene_list),
            "annotation": str(args.annotation),
            "tile_size": int(args.tile_size),
            "stride": int(args.stride),
            "augmentation_level": int(args.augmentation_level),
            "batch_size": int(args.batch_size),
            "workers": str(args.workers),
            "prefetch_factor": int(args.prefetch_factor),
            "max_runtime_sec": float(args.max_runtime_sec),
            "max_batches": int(args.max_batches),
            "normalization_mode": str(args.normalization_mode),
        },
        "dataset": {
            "train_records": len(bundle.train_dataset),
            "val_records": len(bundle.val_dataset),
            "build_datasets_sec": float(build_datasets_sec),
            "warnings": list(bundle.warnings),
            "train_summary": dict(bundle.train_summary),
            "val_summary": dict(bundle.val_summary),
        },
        "runs": runs,
        "batch_wait_mean_sec": combined["batch_wait_mean_sec"],
        "forward_mean_sec": combined["forward_mean_sec"],
        "backward_mean_sec": combined["backward_mean_sec"],
        "total_step_mean_sec": combined["step_total_mean_sec"],
        "combined": combined,
    }


def _write_outputs(output_dir: Path, summary: dict[str, Any], batch_rows: list[dict[str, Any]], function_rows: list[dict[str, Any]]) -> None:
    (output_dir / "profile_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "profile_batches.csv", batch_rows)
    _write_csv(output_dir / "profile_functions.csv", function_rows)
    (output_dir / "profile_report.md").write_text(_report_markdown(summary, function_rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(summary: dict[str, Any], function_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Local Tile Training Profile",
        "",
        f"- requested_device: `{summary['requested_device']}`",
        f"- actual_device: `{summary['actual_device']}`",
        f"- cuda_fallback_reason: `{summary.get('cuda_fallback_reason')}`",
        f"- build_datasets_sec: `{summary['dataset']['build_datasets_sec']:.3f}`",
        f"- train_records: `{summary['dataset']['train_records']}`",
        f"- val_records: `{summary['dataset']['val_records']}`",
        "",
        "## Batch Wait",
        "",
        "| workers | mean | median | p95 | max |",
        "|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['workers']} | {_fmt(run['batch_wait_mean_sec'])} | {_fmt(run['batch_wait_median_sec'])} | "
            f"{_fmt(run['batch_wait_p95_sec'])} | {_fmt(run['batch_wait_max_sec'])} |"
        )
    lines.extend(["", "## Tile Prep Function Breakdown", "", "| workers | function | mean/sample | p95/sample | total | percent of getitem |", "|---:|---|---:|---:|---:|---:|"])
    for row in function_rows:
        lines.append(
            f"| {row['workers']} | {row['function']} | {_fmt(row['mean_sample_sec'])} | {_fmt(row['p95_sample_sec'])} | "
            f"{_fmt(row['total_sec'])} | {row['percent_of_getitem']:.1f}% |"
        )
    lines.extend(["", "## GPU Step Breakdown", "", "| workers | section | mean/batch | p95/batch | percent of step |", "|---:|---|---:|---:|---:|"])
    for run in summary["runs"]:
        step_mean = float(run.get("step_total_mean_sec", 0.0) or 0.0)
        for section in ["batch_wait", "cpu_to_gpu", "forward", "loss", "backward", "optimizer_step", "step_total"]:
            mean = float(run.get(f"{section}_mean_sec", 0.0) or 0.0)
            p95 = float(run.get(f"{section}_p95_sec", 0.0) or 0.0)
            percent = mean / step_mean * 100.0 if step_mean > 0 and section != "step_total" else 100.0 if section == "step_total" else 0.0
            lines.append(f"| {run['workers']} | {section} | {_fmt(mean)} | {_fmt(p95)} | {percent:.1f}% |")
    lines.extend(["", "## Diagnosis", ""])
    for run in summary["runs"]:
        lines.append(f"- workers={run['workers']}: bottleneck=`{run['bottleneck']}`.")
    lines.append("")
    return "\n".join(lines)


def _diagnose(summary: dict[str, Any], function_rows: list[dict[str, Any]], step_mean: float) -> str:
    if not summary.get("batches"):
        return "unknown"
    batch_wait = float(summary.get("batch_wait_mean_sec", 0.0) or 0.0)
    gpu_compute = float(summary.get("gpu_compute_mean_sec", 0.0) or 0.0)
    cpu_to_gpu = float(summary.get("cpu_to_gpu_mean_sec", 0.0) or 0.0)
    if step_mean > 0 and batch_wait / step_mean >= 0.35:
        top_tile = _top_tile_function(function_rows, int(summary.get("workers", -1)))
        return top_tile or "batch_wait"
    if step_mean > 0 and gpu_compute / step_mean >= 0.5:
        return "gpu_compute"
    if step_mean > 0 and cpu_to_gpu / step_mean >= 0.25:
        return "cpu_to_gpu"
    return "unknown"


def _top_tile_function(function_rows: list[dict[str, Any]], workers: int) -> str | None:
    candidates = [row for row in function_rows if int(row.get("workers", -2)) == workers and row.get("function") != "total_getitem"]
    if not candidates:
        return None
    top = max(candidates, key=lambda row: float(row.get("mean_sample_sec", 0.0) or 0.0))
    return str(top.get("function") or "batch_wait")


def _stats(values: list[float]) -> dict[str, float]:
    cleaned = [float(value) for value in values]
    ordered = sorted(cleaned)
    return {
        "count": float(len(ordered)),
        "total": float(sum(ordered)),
        "mean": float(sum(ordered) / len(ordered)) if ordered else 0.0,
        "median": float(statistics.median(ordered)) if ordered else 0.0,
        "p95": float(_percentile(ordered, 0.95)) if ordered else 0.0,
        "max": float(max(ordered)) if ordered else 0.0,
    }


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


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"gpu_memory_used_mb": None, "gpu_memory_allocated_mb": None}
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu_memory_used_mb": round((total - free) / 1024 / 1024, 3),
        "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 3),
        "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 3),
    }


def _fmt(value: Any) -> str:
    return f"{float(value or 0.0):.6f}"


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    raise SystemExit(main())
