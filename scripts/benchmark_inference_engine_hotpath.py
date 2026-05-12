from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from InferenceEngine.src.inference_engine.api.schemas import JobRequest, SceneInput  # noqa: E402
from InferenceEngine.src.inference_engine.planning.planner import build_scene_plan  # noqa: E402
from InferenceEngine.src.inference_engine.triton.client import TritonEndpoint, infer_segmentation_batch  # noqa: E402
from InferenceEngine.src.inference_engine.workers.tile_workers import (  # noqa: E402
    infer_prepared_tile_batch_multi_scene,
    infer_tile_batch_multi_scene,
    preprocess_tile_descriptor,
    read_normalized_tile,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MLSystem InferenceEngine GPU hot path")
    parser.add_argument("--triton-url", default=os.getenv("INFERENCE_ENGINE_TRITON_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--triton-grpc-url", default=os.getenv("INFERENCE_ENGINE_TRITON_GRPC_URL", "127.0.0.1:8001"))
    parser.add_argument("--model-name", default=os.getenv("INFERENCE_ENGINE_TRITON_MODEL_NAME", "segformer_b2"))
    parser.add_argument("--model-version", default=None)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--input-bands", type=int, default=4)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--duration-sec", type=float, default=10.0)
    parser.add_argument("--warmup-sec", type=float, default=2.0)
    parser.add_argument("--bench-tiles", type=int, default=64)
    parser.add_argument("--modes", default="direct,perf,current,nodisk")
    parser.add_argument("--transports", default="http")
    parser.add_argument("--output-dir", default="/data/mlsystem/reports/inference_throughput")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.batch_sizes = "1,4,8"
        args.concurrency = "1,4,8"
        args.duration_sec = min(args.duration_sec, 4.0)
        args.warmup_sec = min(args.warmup_sec, 1.0)
        args.bench_tiles = min(args.bench_tiles, 16)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": _hostname(),
        "config": vars(args),
        "results": {},
    }

    modes = {item.strip() for item in args.modes.split(",") if item.strip()}
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    concurrencies = [int(item) for item in args.concurrency.split(",") if item.strip()]
    transports = [item.strip().lower() for item in args.transports.split(",") if item.strip()]

    if "direct" in modes:
        report["results"]["direct_triton_synthetic"] = []
        for transport in transports:
            for batch_size in batch_sizes:
                for concurrency in concurrencies:
                    result = benchmark_direct_triton(args, batch_size=batch_size, concurrency=concurrency, transport=transport)
                    report["results"]["direct_triton_synthetic"].append(result)
                    _write_reports(output_dir, timestamp, report)

    if "perf" in modes:
        report["results"]["triton_perf_analyzer"] = benchmark_perf_analyzer(args, batch_sizes=batch_sizes, concurrencies=concurrencies)
        _write_reports(output_dir, timestamp, report)

    if "current" in modes or "nodisk" in modes:
        with tempfile.TemporaryDirectory(prefix="ie-hotpath-") as tmp:
            root = Path(tmp)
            request, plan = _benchmark_plan(root, args)
            if "current" in modes:
                report["results"]["current_ie_disk_spool"] = benchmark_current_disk_spool(root, request, plan, args)
                _write_reports(output_dir, timestamp, report)
            if "nodisk" in modes:
                report["results"]["ie_no_disk_input_spool"] = benchmark_no_disk(root, request, plan, args)
                _write_reports(output_dir, timestamp, report)

    json_path, md_path = _write_reports(output_dir, timestamp, report)
    print(json_path)
    print(md_path)


def benchmark_direct_triton(args: argparse.Namespace, *, batch_size: int, concurrency: int, transport: str) -> dict[str, Any]:
    endpoint = _endpoint(args, transport=transport)
    sampler = GpuSampler()
    latencies: list[float] = []
    lock = threading.Lock()
    stop_at = time.monotonic() + float(args.warmup_sec) + float(args.duration_sec)
    measure_at = time.monotonic() + float(args.warmup_sec)
    counters = {"requests": 0, "tiles": 0, "errors": 0}

    def worker(seed: int) -> None:
        rng = np.random.default_rng(seed)
        batch = rng.random((batch_size, int(args.input_bands), int(args.tile_size), int(args.tile_size)), dtype=np.float32)
        while time.monotonic() < stop_at:
            started = time.perf_counter()
            try:
                infer_segmentation_batch(endpoint, batch)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if time.monotonic() >= measure_at:
                    with lock:
                        latencies.append(elapsed_ms)
                        counters["requests"] += 1
                        counters["tiles"] += batch_size
            except Exception:
                with lock:
                    counters["errors"] += 1
                raise

    started = time.monotonic()
    sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker, idx) for idx in range(concurrency)]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        gpu = sampler.stop()
    duration = max(0.001, time.monotonic() - started - float(args.warmup_sec))
    return {
        "mode": "direct_triton_synthetic",
        "transport": transport,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "duration_sec": duration,
        "requests": counters["requests"],
        "tiles": counters["tiles"],
        "request_per_sec": counters["requests"] / duration,
        "tiles_per_sec": counters["tiles"] / duration,
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p95": _percentile(latencies, 95),
        "latency_ms_mean": statistics.mean(latencies) if latencies else None,
        "errors": counters["errors"],
        "gpu": gpu,
    }


def benchmark_perf_analyzer(args: argparse.Namespace, *, batch_sizes: list[int], concurrencies: list[int]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not _command_exists("docker"):
        return [{"status": "skipped", "reason": "docker command is unavailable"}]
    for batch_size in batch_sizes:
        for concurrency in concurrencies:
            cmd = [
                "docker",
                "exec",
                "mlsystem-gpu-triton",
                "perf_analyzer",
                "-m",
                args.model_name,
                "-b",
                str(batch_size),
                "--concurrency-range",
                f"{concurrency}:{concurrency}",
                "--shape",
                f"INPUT__0:{int(args.input_bands)},{int(args.tile_size)},{int(args.tile_size)}",
                "--measurement-interval",
                "5000",
            ]
            started = time.monotonic()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
            results.append(
                {
                    "mode": "triton_perf_analyzer",
                    "batch_size": batch_size,
                    "concurrency": concurrency,
                    "returncode": proc.returncode,
                    "duration_sec": time.monotonic() - started,
                    "stdout_tail": proc.stdout[-4000:],
                    "stderr_tail": proc.stderr[-4000:],
                    "parsed": _parse_perf_analyzer(proc.stdout),
                }
            )
    return results


def benchmark_current_disk_spool(root: Path, request: JobRequest, plan: Any, args: argparse.Namespace) -> dict[str, Any]:
    endpoint = _endpoint(args, transport=str(args.transports).split(",")[0].strip() or "http")
    sampler = GpuSampler()
    timings: list[dict[str, Any]] = []
    tiles = plan.tiles[: int(args.bench_tiles)]
    tiles_by_scene = {plan.scene_id: {tile.tile_id: tile for tile in plan.tiles}}
    scene_plans = {plan.scene_id: plan}
    spool_dir = root / "spool" / "bench" / plan.scene_id
    sampler.start()
    started = time.monotonic()
    try:
        for offset in range(0, len(tiles), int(args.batch_sizes.split(",")[0] or 1)):
            batch_tiles = tiles[offset : offset + int(args.batch_sizes.split(",")[0] or 1)]
            descriptors = []
            for tile in batch_tiles:
                descriptor = preprocess_tile_descriptor(tile=tile, scene_plan=plan, request=request, spool_dir=spool_dir)
                descriptors.append({"scene_id": plan.scene_id, **descriptor})
            outputs = infer_tile_batch_multi_scene(descriptors=descriptors, tiles_by_scene=tiles_by_scene, scene_plans=scene_plans, request=request, endpoint=endpoint)
            timings.extend(output.get("timings") or {} for output in outputs)
    finally:
        gpu = sampler.stop()
    duration = time.monotonic() - started
    return _ie_result("current_ie_disk_spool", len(tiles), duration, timings, gpu)


def benchmark_no_disk(root: Path, request: JobRequest, plan: Any, args: argparse.Namespace) -> dict[str, Any]:
    endpoint = _endpoint(args, transport=str(args.transports).split(",")[0].strip() or "http")
    sampler = GpuSampler()
    timings: list[dict[str, Any]] = []
    tiles = plan.tiles[: int(args.bench_tiles)]
    sampler.start()
    started = time.monotonic()
    try:
        for offset in range(0, len(tiles), int(args.batch_sizes.split(",")[0] or 1)):
            batch_tiles = tiles[offset : offset + int(args.batch_sizes.split(",")[0] or 1)]
            prepared = []
            for tile in batch_tiles:
                array, item_timings = read_normalized_tile(tile, plan, request)
                prepared.append({"scene_id": plan.scene_id, "tile_id": tile.tile_id, "tile": tile, "scene_plan": plan, "array": array, "timings": item_timings})
            outputs = infer_prepared_tile_batch_multi_scene(prepared_rows=prepared, request=request, endpoint=endpoint)
            timings.extend(output.get("timings") or {} for output in outputs)
    finally:
        gpu = sampler.stop()
    duration = time.monotonic() - started
    return _ie_result("ie_no_disk_input_spool", len(tiles), duration, timings, gpu)


class GpuSampler:
    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = interval_sec
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        values = [float(item["gpu_util_pct"]) for item in self.samples if item.get("gpu_util_pct") is not None]
        return {
            "samples": len(values),
            "gpu_util_mean": statistics.mean(values) if values else None,
            "gpu_util_max": max(values) if values else None,
            "raw_tail": self.samples[-20:],
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if proc.returncode == 0:
                    for line in proc.stdout.splitlines():
                        parts = [part.strip() for part in line.split(",")]
                        if len(parts) >= 6:
                            self.samples.append(
                                {
                                    "timestamp": parts[0],
                                    "index": parts[1],
                                    "gpu_util_pct": _float(parts[2]),
                                    "mem_util_pct": _float(parts[3]),
                                    "memory_used_mb": _float(parts[4]),
                                    "power_w": _float(parts[5]),
                                }
                            )
            except Exception:
                pass
            self._stop.wait(self.interval_sec)


def _benchmark_plan(root: Path, args: argparse.Namespace) -> tuple[JobRequest, Any]:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except Exception as exc:
        raise RuntimeError(f"rasterio is required for IE path benchmarks: {exc}") from exc
    image_path = root / "benchmark_scene.tif"
    width = max(int(args.tile_size) * 4, int(args.tile_size) + 1)
    height = max(int(args.tile_size) * 4, int(args.tile_size) + 1)
    rng = np.random.default_rng(123)
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=int(args.input_bands),
        dtype="uint16",
        crs="EPSG:3857",
        transform=from_origin(0.0, float(height), 1.0, 1.0),
    ) as ds:
        block = rng.integers(0, 4095, size=(int(args.input_bands), height, width), dtype=np.uint16)
        ds.write(block)
    request = JobRequest(
        experiment_id="hotpath_benchmark",
        scenes=[SceneInput(scene_id="bench_scene", name="bench_scene", path=str(image_path))],
        preprocess={"patch_size": int(args.tile_size), "stride": int(args.tile_size), "input_bands": list(range(1, int(args.input_bands) + 1))},
        resource={"triton_batch_size": int(str(args.batch_sizes).split(",")[0] or 1), "max_preprocess_queue": 4096},
        model={"triton_model_name": args.model_name, "triton_model_version": args.model_version},
    )
    plan = build_scene_plan("hotpath_benchmark", 0, request.scenes[0], request, root / "jobs")
    return request, plan


def _endpoint(args: argparse.Namespace, *, transport: str) -> TritonEndpoint:
    return TritonEndpoint(
        url=args.triton_url,
        grpc_url=args.triton_grpc_url,
        transport=transport,
        model_name=args.model_name,
        model_version=args.model_version,
    )


def _ie_result(mode: str, tiles: int, duration: float, timings: list[dict[str, Any]], gpu: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    keys = sorted({key for row in timings for key, value in row.items() if isinstance(value, (int, float))})
    for key in keys:
        values = [float(row[key]) for row in timings if isinstance(row.get(key), (int, float))]
        summary[key] = statistics.mean(values) if values else None
    return {
        "mode": mode,
        "tiles": tiles,
        "duration_sec": duration,
        "tiles_per_sec": tiles / max(0.001, duration),
        "timing_ms_mean": summary,
        "gpu": gpu,
    }


def _write_reports(output_dir: Path, timestamp: str, report: dict[str, Any]) -> tuple[Path, Path]:
    json_path = output_dir / f"benchmark_{timestamp}.json"
    md_path = output_dir / f"benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return json_path, md_path


def _render_markdown(report: dict[str, Any]) -> str:
    lines = ["# InferenceEngine hotpath benchmark", "", f"Created: {report.get('created_at')}", ""]
    results = report.get("results") or {}
    rows = []
    for name, payload in results.items():
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if not isinstance(item, dict):
                continue
            gpu = item.get("gpu") or {}
            rows.append(
                [
                    item.get("mode") or name,
                    item.get("transport", ""),
                    item.get("batch_size", ""),
                    item.get("concurrency", ""),
                    _fmt(item.get("tiles_per_sec")),
                    _fmt(gpu.get("gpu_util_mean")),
                    _fmt(gpu.get("gpu_util_max")),
                ]
            )
    lines.append("| mode | transport | batch | concurrency | tiles/sec | gpu mean | gpu max |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    lines.append("")
    lines.append("## Raw Summary")
    lines.append("```json")
    lines.append(json.dumps(report.get("results") or {}, ensure_ascii=False, indent=2, sort_keys=True)[:20000])
    lines.append("```")
    return "\n".join(lines)


def _parse_perf_analyzer(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in stdout.splitlines():
        text = line.strip()
        for key in ("Throughput", "Avg latency", "p50 latency", "p95 latency", "GPU Utilization"):
            if text.lower().startswith(key.lower()):
                parsed[key] = text
    return parsed


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((percentile / 100.0) * (len(values) - 1))))
    return values[index]


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return ""


def _hostname() -> str:
    try:
        return subprocess.check_output(["hostname"], text=True, timeout=3).strip()
    except Exception:
        return "unknown"


def _command_exists(name: str) -> bool:
    from shutil import which

    return which(name) is not None


if __name__ == "__main__":
    main()
