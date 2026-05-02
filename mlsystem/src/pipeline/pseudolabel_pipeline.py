from __future__ import annotations

import time
import json
import gc
import os
import subprocess
import sys
import pickle
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from affine import Affine
from shapely.geometry import shape

from ..debug.pseudolabel_debug import build_postprocess_debug, write_probability_preview
from ..inference.scene_inference import SceneInferenceConfig, SceneInferenceRunner
from ..job_schema import JobSpec
from ..metrics.object_metric_artifacts import write_object_metrics_artifacts
from ..metrics.object_metrics import compute_object_f1
from ..pipeline.contracts import ProbabilityMap, VectorizationResult
from ..pipeline_config import PipelineConfig
from ..postprocessing.filtering import postprocess_vectorization_result
from ..postprocessing.pseudolabel_export import export_pseudolabel_artifacts
from ..postprocessing.vectorization import vertex_count, vectorize_probability_map
from ..reporting.prediction_examples import write_prediction_examples_report
from ..storage.local_io import write_json


def _scene_file_names(matches: list[Any]) -> list[str]:
    return [PurePosixPath(match.name or match.key).name for match in matches]


def write_scene_list(path: Path, matches: list[Any]) -> Path:
    path.write_text("\n".join(_scene_file_names(matches)) + ("\n" if matches else ""), encoding="utf-8")
    return path


def _geojson_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _features_geojson_size_mb(features: list[dict[str, Any]]) -> float:
    payload = {"type": "FeatureCollection", "features": features}
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) / (1024 * 1024)


def _scene_area_m2(row: dict[str, Any]) -> float:
    width = float(row.get("width") or 0)
    height = float(row.get("height") or 0)
    transform = row.get("transform") or []
    if width <= 0 or height <= 0 or len(transform) < 6:
        return 0.0
    a, b, _c, d, e, _f = [float(item) for item in transform[:6]]
    pixel_area = abs(a * e - b * d)
    return width * height * pixel_area


def _effective_max_geojson_mb(
    post_cfg: dict[str, Any],
    scene_rows: list[dict[str, Any]],
    configured_max_geojson_mb: float,
) -> tuple[float, dict[str, Any]]:
    adaptive_cfg = post_cfg.get("adaptive_max_geojson_mb") or post_cfg.get("adaptive_geojson_limit")
    if not adaptive_cfg:
        return configured_max_geojson_mb, {"enabled": False, "configured_max_geojson_mb": configured_max_geojson_mb}
    if adaptive_cfg is True:
        adaptive_cfg = {}
    if not isinstance(adaptive_cfg, dict):
        return configured_max_geojson_mb, {
            "enabled": False,
            "configured_max_geojson_mb": configured_max_geojson_mb,
            "warning": "adaptive_max_geojson_mb must be a boolean or object",
        }

    reference_mb = float(adaptive_cfg.get("reference_max_geojson_mb") or configured_max_geojson_mb or 20)
    coefficient = float(
        adaptive_cfg.get("area_scale_coefficient")
        or adaptive_cfg.get("coefficient")
        or post_cfg.get("geojson_area_scale_coefficient")
        or 1.0
    )
    target_area_m2 = sum(_scene_area_m2(row) for row in scene_rows)
    reference_area_m2_raw = adaptive_cfg.get("reference_area_m2")
    reference_area_m2 = float(reference_area_m2_raw) if reference_area_m2_raw else 0.0
    reference_match = str(
        adaptive_cfg.get("reference_uri_contains")
        or adaptive_cfg.get("reference_area_uri_contains")
        or adaptive_cfg.get("reference_area_match")
        or "/irkutsk/"
    ).lower()
    if reference_area_m2 <= 0 and reference_match:
        reference_area_m2 = sum(
            _scene_area_m2(row)
            for row in scene_rows
            if reference_match in str(row.get("image_uri") or row.get("scene_id") or "").lower()
        )

    if reference_area_m2 <= 0 or target_area_m2 <= 0:
        return configured_max_geojson_mb, {
            "enabled": True,
            "configured_max_geojson_mb": configured_max_geojson_mb,
            "effective_max_geojson_mb": configured_max_geojson_mb,
            "reference_max_geojson_mb": reference_mb,
            "area_scale_coefficient": coefficient,
            "target_area_m2": target_area_m2,
            "reference_area_m2": reference_area_m2,
            "reference_uri_contains": reference_match,
            "warning": "adaptive reference or target area is empty; fixed max_geojson_mb was used",
        }

    effective = reference_mb * coefficient * (target_area_m2 / reference_area_m2)
    min_mb = adaptive_cfg.get("min_geojson_mb")
    max_mb = adaptive_cfg.get("max_geojson_mb_cap") or adaptive_cfg.get("max_geojson_mb")
    if min_mb is not None:
        effective = max(float(min_mb), effective)
    if max_mb is not None:
        effective = min(float(max_mb), effective)
    return effective, {
        "enabled": True,
        "configured_max_geojson_mb": configured_max_geojson_mb,
        "effective_max_geojson_mb": effective,
        "reference_max_geojson_mb": reference_mb,
        "area_scale_coefficient": coefficient,
        "target_area_m2": target_area_m2,
        "reference_area_m2": reference_area_m2,
        "area_ratio": target_area_m2 / reference_area_m2,
        "reference_uri_contains": reference_match,
        "min_geojson_mb": min_mb,
        "max_geojson_mb_cap": max_mb,
    }


def _pseudolabel_runtime_options(
    job: JobSpec,
    matches: list[Any],
    patch_size: int,
    threshold: float,
) -> dict[str, Any]:
    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    post_cfg = job.postprocess or {}
    thresholds = post_cfg.get("thresholds") or [threshold]
    model_cfg = (job.params.get("model") if isinstance(job.params.get("model"), dict) else {}) or {}
    model_cfg = {**model_cfg, **(job.predict.get("model") or {}), **(job.train.get("model") or {})}
    model_name = str(model_cfg.get("name") or job.train.get("model_name") or job.predict.get("model_name") or "").lower()
    full_scene = bool(pseudolabel_cfg.get("full_scene", True))
    max_windows_per_scene = pseudolabel_cfg.get("max_windows_per_scene")
    batch_size_raw = pseudolabel_cfg.get("batch_size") or pseudolabel_cfg.get("inference_batch_size")
    if batch_size_raw is not None:
        inference_batch_size = max(1, int(batch_size_raw))
    elif patch_size <= 512:
        inference_batch_size = 128
    elif patch_size <= 768:
        inference_batch_size = 32 if "deeplab" in model_name else 48
    else:
        if model_name.startswith("segformer_b3"):
            inference_batch_size = 4
        elif model_name.startswith("segformer_b2"):
            inference_batch_size = 8
        elif model_name.startswith("segformer"):
            inference_batch_size = 16
        else:
            inference_batch_size = 32
    center_size_value = pseudolabel_cfg.get("center_size") or pseudolabel_cfg.get("sample_size") or job.params.get("center_size") or job.params.get("sample_size")
    center_size = int(center_size_value) if center_size_value is not None else None
    context_value = pseudolabel_cfg.get("context_bounds") or pseudolabel_cfg.get("bounds") or job.params.get("context_bounds") or job.params.get("bounds")
    context_bounds = int(context_value) if context_value is not None else None
    crop_mode = str(pseudolabel_cfg.get("crop_mode") or "").lower()
    if not crop_mode:
        crop_mode = "center" if model_name.startswith("segformer") and center_size and center_size < patch_size else "full"
    if crop_mode == "center" and not center_size:
        center_size = max(1, patch_size - 2 * int(context_bounds or 0))
    if crop_mode == "center" and center_size >= patch_size:
        crop_mode = "full"
    stitch_mode = str(pseudolabel_cfg.get("stitch_mode") or "").lower()
    if not stitch_mode:
        stitch_mode = "weighted_overlap" if model_name.startswith("segformer") else "hard_insert"
    parallel_cfg = ((job.predict.get("inference") or {}).get("parallel") or (job.params.get("inference") or {}).get("parallel") or {})
    parallel_enabled = bool(parallel_cfg.get("enabled", True))
    default_max_workers = 4 if parallel_enabled else 1
    if model_name.startswith("segformer") and patch_size >= 1024:
        # Keep GPU forwards serialized by default, but let several scene
        # feeders read and normalize tiles in parallel. Otherwise one CPU
        # thread becomes the bottleneck and the GPU waits between batches.
        default_max_workers = min(4, max(1, len(matches)))
    max_workers = max(1, int(parallel_cfg.get("max_workers") or default_max_workers))
    torch_threads_per_worker = max(1, int(parallel_cfg.get("torch_threads_per_worker") or max(1, torch.get_num_threads() // max_workers)))
    gpu_forward_concurrency = max(
        1,
        int(
            parallel_cfg.get("gpu_forward_concurrency")
            or parallel_cfg.get("max_concurrent_forwards")
            or (1 if model_name.startswith("segformer") and patch_size >= 1024 else (2 if parallel_enabled and max_workers > 1 else 1))
        ),
    )
    tile_read_workers = max(
        1,
        int(
            parallel_cfg.get("tile_read_workers")
            or parallel_cfg.get("reader_workers")
            or parallel_cfg.get("window_read_workers")
            or (min(8, os.cpu_count() or 1) if model_name.startswith("segformer") and patch_size >= 1024 else 1)
        ),
    )
    tile_prefetch_batches = max(1, int(parallel_cfg.get("tile_prefetch_batches") or parallel_cfg.get("prefetch_batches") or 4))
    return {
        "pseudolabel_cfg": pseudolabel_cfg,
        "post_cfg": post_cfg,
        "thresholds": thresholds,
        "model_name": model_name,
        "full_scene": full_scene,
        "max_windows_per_scene": max_windows_per_scene,
        "debug_mode": bool(pseudolabel_cfg.get("debug", False)),
        "inference_batch_size": inference_batch_size,
        "center_size": center_size,
        "context_bounds": context_bounds,
        "crop_mode": crop_mode,
        "stitch_mode": stitch_mode,
        "parallel_enabled": parallel_enabled,
        "max_workers": max_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
        "gpu_forward_concurrency": gpu_forward_concurrency,
        "tile_read_workers": tile_read_workers,
        "tile_prefetch_batches": tile_prefetch_batches,
    }


def _scene_result_dir(experiment_dir: Path) -> Path:
    return experiment_dir / "pseudolabel_scene_results"


def _save_scene_result(result: Any, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scene_{int(result.scene_index):04d}"
    npz_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.json"
    probability_map = result.probability_map
    transform_values = list(probability_map.transform)[:6] if probability_map.transform is not None else None
    # Keep this uncompressed: compression is CPU-heavy, holds the Airflow GPU
    # slot after inference, and delays downstream queued GPU work.
    np.savez(
        npz_path,
        prob=probability_map.prob.astype(np.float16, copy=False),
        weight_sum=probability_map.weight_sum.astype(np.float16, copy=False),
        coverage_mask=probability_map.coverage_mask.astype(np.uint8, copy=False),
    )
    meta = {
        "scene_index": int(result.scene_index),
        "scene_id": result.scene_id,
        "scene_name": result.scene_name,
        "npz_path": npz_path.name,
        "transform": transform_values,
        "crs": probability_map.crs,
        "coverage_fraction": float(probability_map.coverage_fraction),
        "metadata": probability_map.metadata,
        "debug_row": result.debug_row,
        "windows_preview_features": result.windows_preview_features,
        "tile_insert_features": result.tile_insert_features,
        "tile_insert_debug_rows": result.tile_insert_debug_rows,
        "scene_duration_sec": result.scene_duration_sec,
    }
    write_json(meta_path, meta)
    return {
        "scene_index": int(result.scene_index),
        "scene_id": result.scene_id,
        "scene_name": result.scene_name,
        "npz_path": str(npz_path),
        "meta_path": str(meta_path),
        "coverage_fraction": float(probability_map.coverage_fraction),
        "debug_row": result.debug_row,
        "scene_duration_sec": result.scene_duration_sec,
    }


def _load_scene_result(row: dict[str, Any]) -> Any:
    meta_path = Path(str(row.get("meta_path") or ""))
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    npz_path = Path(str(row.get("npz_path") or meta_path.with_suffix(".npz")))
    payload = np.load(npz_path)
    transform_values = meta.get("transform")
    transform = Affine(*transform_values[:6]) if transform_values else None
    prob = payload["prob"]
    probability_map = ProbabilityMap(
        scene_id=meta["scene_id"],
        prob=prob,
        weight_sum=payload["weight_sum"],
        coverage_mask=payload["coverage_mask"].astype(bool),
        coverage_fraction=float(meta.get("coverage_fraction") or 0.0),
        transform=transform,
        crs=meta.get("crs"),
        metadata=meta.get("metadata") or {},
    )
    return SimpleNamespace(
        scene_index=int(meta["scene_index"]),
        scene_id=meta["scene_id"],
        scene_name=meta["scene_name"],
        probability_map=probability_map,
        windows_preview_features=meta.get("windows_preview_features") or [],
        tile_insert_features=meta.get("tile_insert_features") or [],
        tile_insert_debug_rows=meta.get("tile_insert_debug_rows") or [],
        debug_row=meta.get("debug_row") or {},
        scene_duration_sec=float(meta.get("scene_duration_sec") or 0.0),
    )


def _vectorize_scene_result_row(args: tuple[dict[str, Any], float, float]) -> VectorizationResult:
    row, threshold_value, min_area_prefilter = args
    result = _load_scene_result(row)
    return vectorize_probability_map(
        result.probability_map,
        scene_name=result.scene_name,
        threshold=threshold_value,
        min_area_m2=min_area_prefilter,
    )


def vectorization_result_to_dict(result: VectorizationResult) -> dict[str, Any]:
    return {
        "features_raw": result.features_raw,
        "raw_count": int(result.raw_count),
        "vertices_before": int(result.vertices_before),
        "threshold": float(result.threshold),
        "crs": result.crs,
        "metadata": result.metadata,
    }


def vectorization_result_from_dict(payload: dict[str, Any]) -> VectorizationResult:
    return VectorizationResult(
        features_raw=payload.get("features_raw") or [],
        raw_count=int(payload.get("raw_count") or 0),
        vertices_before=int(payload.get("vertices_before") or 0),
        threshold=float(payload.get("threshold") or 0.0),
        crs=payload.get("crs") or "EPSG:3857",
        metadata=payload.get("metadata") or {},
    )


def _run_vectorize_worker(payload_path: Path, output_path: Path, log_path: Path) -> VectorizationResult:
    env = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = package_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [
        sys.executable,
        "-m",
        "mlsystem.src.pipeline.pseudolabel_vectorize_worker",
        "--input",
        str(payload_path),
        "--output",
        str(output_path),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True, env=env)
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        raise RuntimeError(f"Vectorization worker failed for {payload_path}: {' | '.join(tail)}")
    if output_path.suffix == ".pkl":
        with output_path.open("rb") as handle:
            return vectorization_result_from_dict(pickle.load(handle))
    return vectorization_result_from_dict(json.loads(output_path.read_text(encoding="utf-8-sig")))


def run_pseudolabel_inference_stage(
    config: PipelineConfig,
    job: JobSpec,
    experiment_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    matches: list[Any],
    input_bands: list[int],
    patch_size: int,
    threshold: float,
    seed: int,
) -> tuple[dict[str, Any], list[Path]]:
    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    if not pseudolabel_cfg.get("enabled", False):
        return {"enabled": False}, []
    started = time.time()
    pseudolabel_scenes_path = write_scene_list(experiment_dir / "pseudolabel_scenes.txt", matches)
    options = _pseudolabel_runtime_options(job, matches, patch_size, threshold)
    thresholds = options["thresholds"]
    model.eval()
    previous_torch_threads = torch.get_num_threads()
    torch.set_num_threads(options["torch_threads_per_worker"])
    runner = SceneInferenceRunner(
        config,
        model,
        device,
        SceneInferenceConfig(
            input_bands=input_bands,
            patch_size=patch_size,
            stride=int(job.preprocess.get("stride") or patch_size),
            threshold=float(thresholds[0]),
            crop_mode=options["crop_mode"],
            stitch_mode=options["stitch_mode"],
            center_size=options["center_size"],
            context_bounds=options["context_bounds"],
            full_scene=options["full_scene"],
            max_windows_per_scene=options["max_windows_per_scene"],
            batch_size=options["inference_batch_size"],
            collect_debug_features=options["debug_mode"],
            gpu_forward_concurrency=options["gpu_forward_concurrency"],
            tile_read_workers=options["tile_read_workers"],
            tile_prefetch_batches=options["tile_prefetch_batches"],
        ),
    )
    scene_rows: list[dict[str, Any]] = []
    preview_artifacts: list[Path] = []
    out_dir = _scene_result_dir(experiment_dir)
    try:
        if options["parallel_enabled"] and options["max_workers"] > 1 and len(matches) > 1:
            with ThreadPoolExecutor(max_workers=options["max_workers"]) as executor:
                futures = {executor.submit(runner.run_scene, idx, match): idx for idx, match in enumerate(matches)}
                for future in as_completed(futures):
                    result = future.result()
                    row = _save_scene_result(result, out_dir)
                    scene_rows.append(row)
                    print(
                        "[pseudolabel-inference] saved "
                        f"{len(scene_rows)}/{len(matches)} scene={row['scene_name']} "
                        f"coverage={row['coverage_fraction']:.4f} "
                        f"duration_sec={row['scene_duration_sec']:.1f}",
                        flush=True,
                    )
                    if not preview_artifacts:
                        preview = write_probability_preview(result.probability_map.prob, result.scene_name, experiment_dir)
                        if preview:
                            preview_artifacts.append(preview)
                    del result
                    gc.collect()
        else:
            for idx, match in enumerate(matches):
                result = runner.run_scene(idx, match)
                row = _save_scene_result(result, out_dir)
                scene_rows.append(row)
                print(
                    "[pseudolabel-inference] saved "
                    f"{len(scene_rows)}/{len(matches)} scene={row['scene_name']} "
                    f"coverage={row['coverage_fraction']:.4f} "
                    f"duration_sec={row['scene_duration_sec']:.1f}",
                    flush=True,
                )
                if not preview_artifacts:
                    preview = write_probability_preview(result.probability_map.prob, result.scene_name, experiment_dir)
                    if preview:
                        preview_artifacts.append(preview)
                del result
                gc.collect()
    finally:
        torch.set_num_threads(previous_torch_threads)
    if device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()
    gc.collect()

    scene_rows = sorted(scene_rows, key=lambda row: int(row["scene_index"]))
    manifest_path = experiment_dir / "pseudolabel_scene_results_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": time.time(),
            "scene_count": len(scene_rows),
            "scene_result_dir": str(out_dir),
            "scenes": scene_rows,
            "options": {key: value for key, value in options.items() if key not in {"pseudolabel_cfg", "post_cfg"}},
            "preview_artifacts": [str(path) for path in preview_artifacts],
        },
    )
    tiling_debug_rows = [row.get("debug_row") or {} for row in scene_rows]
    coverage_values = [float(row.get("coverage_fraction") or 0.0) for row in scene_rows]
    coverage_report = {
        "schema_version": 1,
        "job_id": job.job_id,
        "stage": "inference",
        "scenes_processed": len(tiling_debug_rows),
        "scenes_failed": 0,
        "total_expected_windows": int(sum(int(row.get("expected_window_count") or 0) for row in tiling_debug_rows)),
        "total_predicted_windows": int(sum(int(row.get("actual_predicted_window_count") or 0) for row in tiling_debug_rows)),
        "total_skipped_windows": int(sum(int(row.get("skipped_window_count") or 0) for row in tiling_debug_rows)),
        "mean_coverage_fraction": float(np.mean(coverage_values)) if coverage_values else None,
        "min_coverage_fraction": float(np.min(coverage_values)) if coverage_values else None,
        "inference_parallel_enabled": options["parallel_enabled"],
        "inference_max_workers": options["max_workers"],
        "torch_threads_per_worker": options["torch_threads_per_worker"],
        "inference_batch_size": options["inference_batch_size"],
        "gpu_forward_concurrency": options["gpu_forward_concurrency"],
        "tile_read_workers": options["tile_read_workers"],
        "tile_prefetch_batches": options["tile_prefetch_batches"],
        "inference_duration_sec": round(time.time() - started, 3),
        "crop_mode": options["crop_mode"],
        "stitch_mode": options["stitch_mode"],
        "center_size": options["center_size"],
        "context_bounds": options["context_bounds"],
        "scenes": tiling_debug_rows,
    }
    coverage_report_path = experiment_dir / "coverage_report.json"
    write_json(coverage_report_path, coverage_report)
    inference_timing_path = experiment_dir / "inference_timing_report.json"
    write_json(
        inference_timing_path,
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "parallel": {
                "enabled": options["parallel_enabled"],
                "strategy": "scene" if options["parallel_enabled"] else "sequential",
                "max_workers": options["max_workers"],
                "torch_threads_per_worker": options["torch_threads_per_worker"],
                "batch_size": options["inference_batch_size"],
                "gpu_forward_concurrency": options["gpu_forward_concurrency"],
                "tile_read_workers": options["tile_read_workers"],
                "tile_prefetch_batches": options["tile_prefetch_batches"],
            },
            "total_inference_duration_sec": coverage_report["inference_duration_sec"],
            "per_scene": [
                {
                    "scene_id": row.get("scene_id"),
                    "expected_window_count": (row.get("debug_row") or {}).get("expected_window_count"),
                    "actual_predicted_window_count": (row.get("debug_row") or {}).get("actual_predicted_window_count"),
                    "scene_duration_sec": row.get("scene_duration_sec"),
                    "coverage_fraction": row.get("coverage_fraction"),
                }
                for row in scene_rows
            ],
        },
    )
    complete_path = experiment_dir / "pseudolabel_inference_complete.json"
    write_json(
        complete_path,
        {
            "schema_version": 1,
            "scene_count": len(scene_rows),
            "finished_at_sec": round(time.time() - started, 3),
            "vectorization_workers": int((job.postprocess or {}).get("vectorization_workers") or 0),
            "manifest_path": str(manifest_path),
        },
    )
    metrics = {
        "pseudolabel_enabled": True,
        "pseudolabel_inference_done": True,
        "inference_duration_sec": coverage_report["inference_duration_sec"],
        "inference_parallel_enabled": float(options["parallel_enabled"]),
        "inference_max_workers": options["max_workers"],
        "torch_threads_per_worker": options["torch_threads_per_worker"],
        "inference_batch_size": options["inference_batch_size"],
        "gpu_forward_concurrency": options["gpu_forward_concurrency"],
        "tile_read_workers": options["tile_read_workers"],
        "tile_prefetch_batches": options["tile_prefetch_batches"],
        "expected_window_count": coverage_report["total_expected_windows"],
        "actual_predicted_window_count": coverage_report["total_predicted_windows"],
        "coverage_fraction": coverage_report["mean_coverage_fraction"],
        "matched_scene_count": len(scene_rows),
        "scenes_processed": len(scene_rows),
    }
    artifacts = [pseudolabel_scenes_path, manifest_path, coverage_report_path, inference_timing_path, complete_path, *preview_artifacts]
    return metrics, artifacts


def run_pseudolabel_postprocess_stage(
    config: PipelineConfig,
    job: JobSpec,
    experiment_dir: Path,
    threshold: float,
    gt_shapes: list[Any] | None = None,
    object_metrics_prefix: str = "val",
) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = experiment_dir / "pseudolabel_scene_results_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Pseudolabel inference manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    scene_rows = sorted(manifest.get("scenes") or [], key=lambda row: int(row.get("scene_index") or 0))
    if not scene_rows:
        raise RuntimeError("Pseudolabel inference manifest contains no scenes")
    post_cfg = job.postprocess or {}
    thresholds = post_cfg.get("thresholds") or [threshold]
    max_objects_raw = post_cfg.get("max_objects", 500)
    max_objects = None if max_objects_raw is None else int(max_objects_raw)
    min_area_candidates = post_cfg.get("min_object_area_m2_candidates") or [1000]
    simplify_candidates = post_cfg.get("simplify_tolerance_m_candidates") or [5]
    configured_max_geojson_mb = float(post_cfg.get("max_geojson_mb") or 20)
    auto_tune = bool(post_cfg.get("auto_tune", False))
    smooth_polygons = bool(post_cfg.get("smooth_polygons", False))
    vectorization_workers = int(post_cfg.get("vectorization_workers") or max(1, min(8, len(scene_rows), os.cpu_count() or 1)))
    max_raw_features_for_candidate = post_cfg.get("max_raw_features_for_candidate")
    if max_raw_features_for_candidate is None:
        max_raw_features_for_candidate = 200000
    max_raw_features_for_candidate = int(max_raw_features_for_candidate) if max_raw_features_for_candidate else 0
    started = time.time()
    preview_artifacts = [Path(path) for path in manifest.get("preview_artifacts") or [] if Path(path).exists()]
    windows_preview_features: list[dict[str, Any]] = []
    tile_insert_features: list[dict[str, Any]] = []
    tile_insert_debug_rows: list[dict[str, Any]] = []
    tiling_debug_rows: list[dict[str, Any]] = []
    for row in scene_rows:
        meta_path = Path(str(row.get("meta_path")))
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        windows_preview_features.extend(meta.get("windows_preview_features") or [])
        tile_insert_features.extend(meta.get("tile_insert_features") or [])
        tile_insert_debug_rows.extend(meta.get("tile_insert_debug_rows") or [])
        tiling_debug_rows.append(meta.get("debug_row") or {})
    max_geojson_mb, adaptive_geojson_limit = _effective_max_geojson_mb(
        post_cfg,
        tiling_debug_rows,
        configured_max_geojson_mb,
    )

    def build_vectorization(threshold_value: float, min_area_prefilter: float) -> VectorizationResult:
        raw_features: list[dict[str, Any]] = []
        raw_vertices = 0

        print(
            "[pseudolabel-postprocess] vectorize "
            f"threshold={threshold_value} scenes={len(scene_rows)} workers={vectorization_workers} "
            f"min_area_prefilter={min_area_prefilter}",
            flush=True,
        )
        if vectorization_workers > 1 and len(scene_rows) > 1:
            vectorized_rows: list[VectorizationResult] = []
            work_dir = experiment_dir / "vectorization_work" / f"thr_{str(threshold_value).replace('.', '_')}_area_{int(min_area_prefilter)}"
            work_dir.mkdir(parents=True, exist_ok=True)
            tasks: list[tuple[Path, Path, Path]] = []
            for row in scene_rows:
                scene_index = int(row.get("scene_index") or 0)
                payload_path = work_dir / f"scene_{scene_index:04d}.input.json"
                output_path = work_dir / f"scene_{scene_index:04d}.output.pkl"
                log_path = work_dir / f"scene_{scene_index:04d}.log"
                write_json(
                    payload_path,
                    {
                        "row": row,
                        "threshold": threshold_value,
                        "min_area_prefilter": min_area_prefilter,
                    },
                )
                tasks.append((payload_path, output_path, log_path))
            with ThreadPoolExecutor(max_workers=vectorization_workers) as executor:
                futures = {executor.submit(_run_vectorize_worker, *item): item[0] for item in tasks}
                for future in as_completed(futures):
                    vectorized_rows.append(future.result())
            vectorized_iter = vectorized_rows
        else:
            vectorized_iter = [_vectorize_scene_result_row((row, threshold_value, min_area_prefilter)) for row in scene_rows]
        for vectorized in vectorized_iter:
            raw_features.extend(vectorized.features_raw)
            raw_vertices += vectorized.vertices_before
        return VectorizationResult(
            features_raw=raw_features,
            raw_count=len(raw_features),
            vertices_before=raw_vertices,
            threshold=threshold_value,
            crs="EPSG:3857",
            metadata={"scene_count": len(scene_rows), "min_area_m2_prefilter": float(min_area_prefilter or 0)},
        )

    selected: tuple[VectorizationResult, Any, float, float, float, float] | None = None
    selected_score: tuple[float, int, float, float] | None = None
    candidates_checked: list[dict[str, Any]] = []
    tune_with_object_f1 = bool(gt_shapes)
    threshold_values = [float(item) for item in thresholds]
    min_area_values = [float(item) for item in min_area_candidates]
    simplify_values = [float(item) for item in simplify_candidates]
    if not auto_tune:
        threshold_values = threshold_values[:1]
        min_area_values = min_area_values[:1]
        simplify_values = simplify_values[:1]
    run_on = str((job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}).get("run_on") or "").lower()
    min_area_prefilter = min(min_area_values) if min_area_values else 0.0
    if not tune_with_object_f1 and auto_tune and run_on in {"all_available_images", "all_images"} and len(scene_rows) > 20 and min_area_values:
        min_area_prefilter = max(min_area_values)
        min_area_values = [value for value in min_area_values if value >= min_area_prefilter]
    for threshold_index, threshold_candidate in enumerate(threshold_values):
        vectorization_candidate = build_vectorization(threshold_candidate, min_area_prefilter)
        if (
            not tune_with_object_f1
            and max_raw_features_for_candidate
            and vectorization_candidate.raw_count > max_raw_features_for_candidate
            and threshold_index < len(threshold_values) - 1
        ):
            candidates_checked.append(
                {
                    "threshold": threshold_candidate,
                    "raw_count": vectorization_candidate.raw_count,
                    "vertices_before": vectorization_candidate.vertices_before,
                    "skipped": True,
                    "skip_reason": "raw_feature_count_exceeds_limit",
                    "max_raw_features_for_candidate": max_raw_features_for_candidate,
                    "min_area_m2_prefilter": min_area_prefilter,
                }
            )
            continue
        forced_noisy_fallback = (
            not tune_with_object_f1
            and max_raw_features_for_candidate
            and vectorization_candidate.raw_count > max_raw_features_for_candidate
        )
        candidate_min_area_values = sorted(min_area_values, reverse=True) if forced_noisy_fallback and not tune_with_object_f1 else min_area_values
        candidate_simplify_values = sorted(simplify_values, reverse=True) if forced_noisy_fallback and not tune_with_object_f1 else simplify_values
        for min_area_candidate in candidate_min_area_values:
            for simplify_candidate in candidate_simplify_values:
                result_candidate = postprocess_vectorization_result(
                    vectorization_candidate,
                    min_area_m2=min_area_candidate,
                    simplify_tolerance_m=simplify_candidate,
                    max_objects=max_objects,
                )
                candidate_size_mb = _features_geojson_size_mb(result_candidate.features)
                pred_geoms = [shape(item["geometry"]) for item in result_candidate.features]
                object_metrics_candidate = (
                    compute_object_f1(pred_geoms, gt_shapes, iou_threshold=0.5) if tune_with_object_f1 else None
                )
                candidate_row = {
                    "threshold": threshold_candidate,
                    "min_object_area_m2": min_area_candidate,
                    "simplify_tolerance_m": simplify_candidate,
                    "forced_noisy_fallback": forced_noisy_fallback,
                    "object_count": len(result_candidate.features),
                    "vertices_count": int(sum(vertex_count(item["geometry"]) for item in result_candidate.features)),
                    "estimated_geojson_mb": candidate_size_mb,
                    "within_max_geojson_mb": candidate_size_mb <= max_geojson_mb,
                    "min_area_m2_prefilter": min_area_prefilter,
                }
                if object_metrics_candidate:
                    candidate_row.update(
                        {
                            "object_f1": float(object_metrics_candidate["object_f1"]),
                            "object_precision": float(object_metrics_candidate["object_precision"]),
                            "object_recall": float(object_metrics_candidate["object_recall"]),
                            "object_tp": int(object_metrics_candidate["object_tp"]),
                            "object_fp": int(object_metrics_candidate["object_fp"]),
                            "object_fn": int(object_metrics_candidate["object_fn"]),
                        }
                    )
                candidates_checked.append(candidate_row)
                if tune_with_object_f1:
                    score = (
                        1 if candidate_size_mb <= max_geojson_mb else 0,
                        float((object_metrics_candidate or {}).get("object_f1") or 0.0),
                        -candidate_size_mb,
                        -float(len(result_candidate.features)),
                    )
                    if selected_score is None or score > selected_score:
                        selected_score = score
                        selected = (vectorization_candidate, result_candidate, threshold_candidate, min_area_candidate, simplify_candidate, candidate_size_mb)
                    continue
                if selected is None or candidate_size_mb < selected[5]:
                    selected = (vectorization_candidate, result_candidate, threshold_candidate, min_area_candidate, simplify_candidate, candidate_size_mb)
                if candidate_size_mb <= max_geojson_mb:
                    break
            if selected and selected[5] <= max_geojson_mb:
                if tune_with_object_f1:
                    continue
                break
        if selected and selected[5] <= max_geojson_mb:
            if tune_with_object_f1:
                continue
            break
    if selected is None:
        raise RuntimeError("Postprocess did not produce any candidate result")
    vectorization, postprocess_result, threshold_used, min_area, simplify_tolerance, _estimated_geojson_mb = selected
    vertices_before = vectorization.vertices_before
    for result in tiling_debug_rows:
        result["objects_before_filter"] = vectorization.raw_count
        result["objects_after_filter"] = vectorization.raw_count
    vertices_after = int(sum(vertex_count(item["geometry"]) for item in postprocess_result.features))
    options = manifest.get("options") or {}
    tiling_debug = {
        "schema_version": 1,
        "full_scene": options.get("full_scene"),
        "crop_mode": options.get("crop_mode"),
        "stitch_mode": options.get("stitch_mode"),
        "center_size": options.get("center_size"),
        "context_bounds": options.get("context_bounds"),
        "inference_batch_size": options.get("inference_batch_size"),
        "vectorization_workers": vectorization_workers,
        "max_raw_features_for_candidate": max_raw_features_for_candidate,
        "min_area_m2_prefilter": min_area_prefilter,
        "adaptive_geojson_limit": adaptive_geojson_limit,
        "scenes": tiling_debug_rows,
    }
    segformer_debug = {
        "schema_version": 1,
        "model_name": options.get("model_name"),
        "crop_mode": options.get("crop_mode"),
        "stitch_mode": options.get("stitch_mode"),
        "center_size": options.get("center_size"),
        "context_bounds": options.get("context_bounds"),
        "inference_batch_size": options.get("inference_batch_size"),
        "tile_inserts": tile_insert_debug_rows[:10000],
    }
    exported = export_pseudolabel_artifacts(
        experiment_dir=experiment_dir,
        job_id=job.job_id,
        postprocess_result=postprocess_result,
        windows_preview_features=windows_preview_features,
        tile_insert_features=tile_insert_features,
        tiling_debug=tiling_debug,
        segformer_debug=segformer_debug,
        debug_mode=bool((job.predict.get("pseudolabel") or {}).get("debug", False)),
    )
    geojson_mb = _geojson_size_mb(exported["geojson_path"])
    gz_mb = _geojson_size_mb(exported["gz_path"])
    gpkg_mb = _geojson_size_mb(exported["gpkg_path"]) if exported["gpkg_path"].exists() else None
    postprocess_debug = build_postprocess_debug(postprocess_result, geojson_mb=geojson_mb, max_geojson_mb=max_geojson_mb)
    postprocess_debug.update(
        {
            "vertices_before": vertices_before,
            "vertices_after": vertices_after,
            "auto_tune": auto_tune,
            "selection_metric": "object_f1" if tune_with_object_f1 else "geojson_size",
            "smooth_polygons": smooth_polygons,
            "vectorization_workers": vectorization_workers,
            "max_raw_features_for_candidate": max_raw_features_for_candidate,
            "min_area_m2_prefilter": min_area_prefilter,
            "configured_max_geojson_mb": configured_max_geojson_mb,
            "effective_max_geojson_mb": max_geojson_mb,
            "adaptive_geojson_limit": adaptive_geojson_limit,
            "candidates_checked": candidates_checked,
        }
    )
    postprocess_debug_path = experiment_dir / "postprocess_debug.json"
    write_json(postprocess_debug_path, postprocess_debug)
    artifacts: list[Path] = [*exported["artifacts"], experiment_dir / "pseudolabel_scenes.txt", postprocess_debug_path, *preview_artifacts]
    artifacts.extend(write_prediction_examples_report(experiment_dir, job.job_id, preview_artifacts, limit=30))
    coverage_report = json.loads((experiment_dir / "coverage_report.json").read_text(encoding="utf-8-sig"))
    coverage_report.update(
        {
            "stage": "postprocess_complete",
            "accepted_objects_total": len(postprocess_result.features),
            "accepted_geojson_mb": geojson_mb,
            "accepted_geojson_gz_mb": gz_mb,
            "accepted_gpkg_mb": gpkg_mb,
            "top_limit_applied": max_objects is not None and postprocess_result.objects_after_filter > int(max_objects),
            "top500_applied": max_objects == 500 and postprocess_result.objects_after_filter > 500,
            "max_objects": max_objects,
            "max_geojson_mb": max_geojson_mb,
            "configured_max_geojson_mb": configured_max_geojson_mb,
            "adaptive_geojson_limit": adaptive_geojson_limit,
            "smooth_polygons": smooth_polygons,
            "vectorization_workers": vectorization_workers,
            "max_raw_features_for_candidate": max_raw_features_for_candidate,
            "min_area_m2_prefilter": min_area_prefilter,
            "postprocess_duration_sec": round(time.time() - started, 3),
        }
    )
    coverage_report_path = experiment_dir / "coverage_report.json"
    write_json(coverage_report_path, coverage_report)
    artifacts.append(coverage_report_path)
    object_metric_values, object_metric_artifacts = write_object_metrics_artifacts(
        experiment_dir,
        postprocess_result.features,
        gt_shapes,
        prefix=object_metrics_prefix,
        iou_threshold=float(post_cfg.get("object_iou_threshold") or 0.5),
    )
    artifacts.extend(object_metric_artifacts)
    total_area = float(sum(float(item["properties"].get("area_m2") or 0) for item in postprocess_result.features))
    metrics = {
        "pseudolabel_enabled": True,
        "accepted_objects": len(postprocess_result.features),
        "accepted_objects_total": len(postprocess_result.features),
        "accepted_geojson_mb": geojson_mb,
        "accepted_geojson_gz_mb": gz_mb,
        "accepted_gpkg_mb": gpkg_mb,
        "total_area_m2": total_area,
        "total_vertices": vertices_after,
        "vertices_count": vertices_after,
        "threshold_used": threshold_used,
        "min_object_area_m2_used": min_area,
        "simplify_tolerance_m_used": simplify_tolerance,
        "postprocess_sec": round(time.time() - started, 3),
        "vectorization_workers": vectorization_workers,
        "max_raw_features_for_candidate": max_raw_features_for_candidate,
        "min_area_m2_prefilter": min_area_prefilter,
        "top_limit_applied": float(max_objects is not None and postprocess_result.objects_after_filter > int(max_objects)),
        "top500_applied": float(max_objects == 500 and postprocess_result.objects_after_filter > 500),
        "pseudolabel/accepted_geojson_mb": geojson_mb,
        "pseudolabel/object_count": len(postprocess_result.features),
        "pseudolabel/vertices_count": vertices_after,
        "pseudolabel/max_geojson_mb": max_geojson_mb,
        "pseudolabel/configured_max_geojson_mb": configured_max_geojson_mb,
        "pseudolabel/adaptive_geojson_limit_enabled": float(bool(adaptive_geojson_limit.get("enabled"))),
        "pseudolabel/adaptive_area_ratio": adaptive_geojson_limit.get("area_ratio"),
    }
    metrics.update(object_metric_values)
    pseudolabel_summary = {
        "schema_version": 1,
        "job_id": job.job_id,
        "accepted_geojson": exported["geojson_path"].name,
        "accepted_objects": len(postprocess_result.features),
        "accepted_geojson_mb": geojson_mb,
        "metrics": metrics,
        "postprocess_debug": postprocess_debug,
        "selected_postprocess_params": {
            "threshold": threshold_used,
            "min_object_area_m2": min_area,
            "simplify_tolerance_m": simplify_tolerance,
            "max_objects": max_objects,
            "max_geojson_mb": max_geojson_mb,
            "configured_max_geojson_mb": configured_max_geojson_mb,
            "adaptive_geojson_limit": adaptive_geojson_limit,
        },
    }
    pseudolabel_summary_path = experiment_dir / "pseudolabel_summary.json"
    write_json(pseudolabel_summary_path, pseudolabel_summary)
    artifacts.append(pseudolabel_summary_path)
    return metrics, artifacts


def run_pseudolabel_pipeline(
    config: PipelineConfig,
    job: JobSpec,
    experiment_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    matches: list[Any],
    input_bands: list[int],
    patch_size: int,
    threshold: float,
    seed: int,
    gt_shapes: list[Any] | None = None,
    object_metrics_prefix: str = "val",
) -> tuple[dict[str, Any], list[Path]]:
    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    if not pseudolabel_cfg.get("enabled", False):
        return {"enabled": False}, []
    pseudolabel_scenes_path = write_scene_list(experiment_dir / "pseudolabel_scenes.txt", matches)

    post_cfg = job.postprocess or {}
    thresholds = post_cfg.get("thresholds") or [threshold]
    max_objects_raw = post_cfg.get("max_objects", 500)
    max_objects = None if max_objects_raw is None else int(max_objects_raw)
    min_area_candidates = post_cfg.get("min_object_area_m2_candidates") or [1000]
    simplify_candidates = post_cfg.get("simplify_tolerance_m_candidates") or [5]
    configured_max_geojson_mb = float(post_cfg.get("max_geojson_mb") or 20)
    auto_tune = bool(post_cfg.get("auto_tune", False))
    smooth_polygons = bool(post_cfg.get("smooth_polygons", False))
    vectorization_workers = int(
        post_cfg.get("vectorization_workers")
        or pseudolabel_cfg.get("vectorization_workers")
        or max(1, min(4, len(matches), os.cpu_count() or 1))
    )
    max_raw_features_for_candidate = post_cfg.get("max_raw_features_for_candidate")
    if max_raw_features_for_candidate is None:
        max_raw_features_for_candidate = 200000
    max_raw_features_for_candidate = int(max_raw_features_for_candidate) if max_raw_features_for_candidate else 0
    full_scene = bool(pseudolabel_cfg.get("full_scene", True))
    max_windows_per_scene = pseudolabel_cfg.get("max_windows_per_scene")
    batch_size_raw = pseudolabel_cfg.get("batch_size") or pseudolabel_cfg.get("inference_batch_size")
    max_debug_scenes = pseudolabel_cfg.get("max_debug_scenes")
    if max_debug_scenes is not None:
        matches = matches[: max(1, int(max_debug_scenes))]
    debug_mode = bool(pseudolabel_cfg.get("debug", False))
    model_cfg = (job.params.get("model") if isinstance(job.params.get("model"), dict) else {}) or {}
    model_cfg = {**model_cfg, **(job.predict.get("model") or {}), **(job.train.get("model") or {})}
    model_name = str(model_cfg.get("name") or job.train.get("model_name") or job.predict.get("model_name") or "").lower()
    center_size_value = pseudolabel_cfg.get("center_size") or pseudolabel_cfg.get("sample_size") or job.params.get("center_size") or job.params.get("sample_size")
    center_size = int(center_size_value) if center_size_value is not None else None
    context_value = pseudolabel_cfg.get("context_bounds") or pseudolabel_cfg.get("bounds") or job.params.get("context_bounds") or job.params.get("bounds")
    context_bounds = int(context_value) if context_value is not None else None
    crop_mode = str(pseudolabel_cfg.get("crop_mode") or "").lower()
    if not crop_mode:
        crop_mode = "center" if model_name.startswith("segformer") and center_size and center_size < patch_size else "full"
    if crop_mode == "center" and not center_size:
        center_size = max(1, patch_size - 2 * int(context_bounds or 0))
    if crop_mode == "center" and center_size >= patch_size:
        crop_mode = "full"
    stitch_mode = str(pseudolabel_cfg.get("stitch_mode") or "").lower()
    if not stitch_mode:
        stitch_mode = "weighted_overlap" if model_name.startswith("segformer") else "hard_insert"
    if batch_size_raw is not None:
        inference_batch_size = max(1, int(batch_size_raw))
    elif patch_size <= 512:
        inference_batch_size = 128
    elif patch_size <= 768:
        inference_batch_size = 32 if "deeplab" in model_name else 48
    else:
        if model_name.startswith("segformer_b3"):
            inference_batch_size = 4
        elif model_name.startswith("segformer_b2"):
            inference_batch_size = 8
        elif model_name.startswith("segformer"):
            inference_batch_size = 16
        else:
            inference_batch_size = 32

    started = time.time()
    model.eval()
    previous_torch_threads = torch.get_num_threads()
    parallel_cfg = ((job.predict.get("inference") or {}).get("parallel") or (job.params.get("inference") or {}).get("parallel") or {})
    parallel_enabled = bool(parallel_cfg.get("enabled", device.type == "cuda"))
    default_max_workers = 4 if parallel_enabled else 1
    if model_name.startswith("segformer") and patch_size >= 1024:
        default_max_workers = min(4, max(1, len(matches)))
    max_workers = max(1, int(parallel_cfg.get("max_workers") or default_max_workers))
    torch_threads_per_worker = max(1, int(parallel_cfg.get("torch_threads_per_worker") or max(1, torch.get_num_threads() // max_workers)))
    gpu_forward_concurrency = max(
        1,
        int(
            parallel_cfg.get("gpu_forward_concurrency")
            or parallel_cfg.get("max_concurrent_forwards")
            or (1 if model_name.startswith("segformer") and patch_size >= 1024 else (2 if parallel_enabled and device.type == "cuda" and max_workers > 1 else 1))
        ),
    )
    torch.set_num_threads(torch_threads_per_worker)

    runner = SceneInferenceRunner(
        config,
        model,
        device,
        SceneInferenceConfig(
            input_bands=input_bands,
            patch_size=patch_size,
            stride=int(job.preprocess.get("stride") or patch_size),
            threshold=float(thresholds[0]),
            crop_mode=crop_mode,
            stitch_mode=stitch_mode,
            center_size=center_size,
            context_bounds=context_bounds,
            full_scene=full_scene,
            max_windows_per_scene=max_windows_per_scene,
            batch_size=inference_batch_size,
            collect_debug_features=debug_mode,
            gpu_forward_concurrency=gpu_forward_concurrency,
        ),
    )

    scene_results = []
    try:
        if parallel_enabled and max_workers > 1 and len(matches) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(runner.run_scene, idx, match): idx
                    for idx, match in enumerate(matches)
                }
                for future in as_completed(futures):
                    scene_results.append(future.result())
        else:
            scene_results = [runner.run_scene(idx, match) for idx, match in enumerate(matches)]
    finally:
        torch.set_num_threads(previous_torch_threads)
    if device.type == "cuda":
        model.to("cpu")
        torch.cuda.empty_cache()
    gc.collect()
    write_json(
        experiment_dir / "pseudolabel_inference_complete.json",
        {
            "schema_version": 1,
            "scene_count": len(scene_results),
            "finished_at_sec": round(time.time() - started, 3),
            "vectorization_workers": vectorization_workers,
        },
    )

    all_raw_features: list[dict[str, Any]] = []
    windows_preview_features: list[dict[str, Any]] = []
    tile_insert_features: list[dict[str, Any]] = []
    tile_insert_debug_rows: list[dict[str, Any]] = []
    tiling_debug_rows: list[dict[str, Any]] = []
    vertices_before = 0
    preview_artifacts: list[Path] = []
    for result in sorted(scene_results, key=lambda item: item.scene_index):
        windows_preview_features.extend(result.windows_preview_features)
        tile_insert_features.extend(result.tile_insert_features)
        tile_insert_debug_rows.extend(result.tile_insert_debug_rows)
        tiling_debug_rows.append(result.debug_row)
        if not preview_artifacts:
            preview = write_probability_preview(result.probability_map.prob, result.scene_name, experiment_dir)
            if preview:
                preview_artifacts.append(preview)
    max_geojson_mb, adaptive_geojson_limit = _effective_max_geojson_mb(
        post_cfg,
        tiling_debug_rows,
        configured_max_geojson_mb,
    )

    min_area_values_for_prefilter = [float(item) for item in min_area_candidates]
    min_area_prefilter = min(min_area_values_for_prefilter) if min_area_values_for_prefilter else 0.0

    def build_vectorization(threshold_value: float) -> VectorizationResult:
        raw_features: list[dict[str, Any]] = []
        raw_vertices = 0

        def vectorize_scene(result: Any) -> VectorizationResult:
            return vectorize_probability_map(
                result.probability_map,
                scene_name=result.scene_name,
                threshold=threshold_value,
                min_area_m2=min_area_prefilter,
            )

        sorted_results = sorted(scene_results, key=lambda item: item.scene_index)
        if vectorization_workers > 1 and len(sorted_results) > 1:
            vectorized_rows: list[VectorizationResult] = []
            with ThreadPoolExecutor(max_workers=vectorization_workers) as executor:
                futures = {executor.submit(vectorize_scene, result): result.scene_index for result in sorted_results}
                for future in as_completed(futures):
                    vectorized_rows.append(future.result())
            vectorized_iter = vectorized_rows
        else:
            vectorized_iter = [vectorize_scene(result) for result in sorted_results]
        for vectorized in vectorized_iter:
            raw_features.extend(vectorized.features_raw)
            raw_vertices += vectorized.vertices_before
        return VectorizationResult(
            features_raw=raw_features,
            raw_count=len(raw_features),
            vertices_before=raw_vertices,
            threshold=threshold_value,
            crs="EPSG:3857",
            metadata={"scene_count": len(scene_results)},
        )

    selected: tuple[VectorizationResult, Any, float, float, float, float] | None = None
    selected_score: tuple[float, int, float, float] | None = None
    candidates_checked: list[dict[str, Any]] = []
    tune_with_object_f1 = bool(gt_shapes)
    threshold_values = [float(item) for item in thresholds]
    min_area_values = [float(item) for item in min_area_candidates]
    simplify_values = [float(item) for item in simplify_candidates]
    if not auto_tune:
        threshold_values = threshold_values[:1]
        min_area_values = min_area_values[:1]
        simplify_values = simplify_values[:1]
    for threshold_index, threshold_candidate in enumerate(threshold_values):
        vectorization_candidate = build_vectorization(threshold_candidate)
        if (
            not tune_with_object_f1
            and max_raw_features_for_candidate
            and vectorization_candidate.raw_count > max_raw_features_for_candidate
            and threshold_index < len(threshold_values) - 1
        ):
            candidates_checked.append(
                {
                    "threshold": threshold_candidate,
                    "raw_count": vectorization_candidate.raw_count,
                    "vertices_before": vectorization_candidate.vertices_before,
                    "skipped": True,
                    "skip_reason": "raw_feature_count_exceeds_limit",
                    "max_raw_features_for_candidate": max_raw_features_for_candidate,
                }
            )
            continue
        forced_noisy_fallback = (
            not tune_with_object_f1
            and max_raw_features_for_candidate
            and vectorization_candidate.raw_count > max_raw_features_for_candidate
        )
        candidate_min_area_values = min_area_values
        candidate_simplify_values = simplify_values
        if forced_noisy_fallback and not tune_with_object_f1:
            candidate_min_area_values = sorted(min_area_values, reverse=True)
            candidate_simplify_values = sorted(simplify_values, reverse=True)
        for min_area_candidate in candidate_min_area_values:
            for simplify_candidate in candidate_simplify_values:
                result_candidate = postprocess_vectorization_result(
                    vectorization_candidate,
                    min_area_m2=min_area_candidate,
                    simplify_tolerance_m=simplify_candidate,
                    max_objects=max_objects,
                )
                candidate_size_mb = _features_geojson_size_mb(result_candidate.features)
                pred_geoms = [shape(item["geometry"]) for item in result_candidate.features]
                object_metrics_candidate = (
                    compute_object_f1(pred_geoms, gt_shapes, iou_threshold=0.5) if tune_with_object_f1 else None
                )
                candidate_row = {
                    "threshold": threshold_candidate,
                    "min_object_area_m2": min_area_candidate,
                    "simplify_tolerance_m": simplify_candidate,
                    "forced_noisy_fallback": forced_noisy_fallback,
                    "object_count": len(result_candidate.features),
                    "vertices_count": int(sum(vertex_count(item["geometry"]) for item in result_candidate.features)),
                    "estimated_geojson_mb": candidate_size_mb,
                    "within_max_geojson_mb": candidate_size_mb <= max_geojson_mb,
                }
                if object_metrics_candidate:
                    candidate_row.update(
                        {
                            "object_f1": float(object_metrics_candidate["object_f1"]),
                            "object_precision": float(object_metrics_candidate["object_precision"]),
                            "object_recall": float(object_metrics_candidate["object_recall"]),
                            "object_tp": int(object_metrics_candidate["object_tp"]),
                            "object_fp": int(object_metrics_candidate["object_fp"]),
                            "object_fn": int(object_metrics_candidate["object_fn"]),
                        }
                    )
                candidates_checked.append(candidate_row)

                if tune_with_object_f1:
                    # Prefer valid-size candidates with the best CHTZ object F1.
                    # If none fit the size budget yet, still keep the best F1 candidate
                    # but rank valid-size candidates above invalid ones.
                    score = (
                        1 if candidate_size_mb <= max_geojson_mb else 0,
                        float((object_metrics_candidate or {}).get("object_f1") or 0.0),
                        -candidate_size_mb,
                        -float(len(result_candidate.features)),
                    )
                    if selected_score is None or score > selected_score:
                        selected_score = score
                        selected = (
                            vectorization_candidate,
                            result_candidate,
                            threshold_candidate,
                            min_area_candidate,
                            simplify_candidate,
                            candidate_size_mb,
                        )
                    continue

                if selected is None or candidate_size_mb < selected[5]:
                    selected = (
                        vectorization_candidate,
                        result_candidate,
                        threshold_candidate,
                        min_area_candidate,
                        simplify_candidate,
                        candidate_size_mb,
                    )
                if candidate_size_mb <= max_geojson_mb:
                    break
            if selected and selected[5] <= max_geojson_mb:
                if tune_with_object_f1:
                    continue
                break
        if selected and selected[5] <= max_geojson_mb:
            if tune_with_object_f1:
                continue
            break
    if selected is None:
        raise RuntimeError("Postprocess did not produce any candidate result")
    vectorization, postprocess_result, threshold_used, min_area, simplify_tolerance, _estimated_geojson_mb = selected
    all_raw_features = vectorization.features_raw
    vertices_before = vectorization.vertices_before
    for result in tiling_debug_rows:
        result["objects_before_filter"] = vectorization.raw_count
        result["objects_after_filter"] = vectorization.raw_count
    vertices_after = int(sum(vertex_count(item["geometry"]) for item in postprocess_result.features))
    tiling_debug = {
        "schema_version": 1,
        "full_scene": full_scene,
        "crop_mode": crop_mode,
        "stitch_mode": stitch_mode,
        "center_size": center_size,
        "context_bounds": context_bounds,
        "inference_batch_size": inference_batch_size,
        "vectorization_workers": vectorization_workers,
        "max_raw_features_for_candidate": max_raw_features_for_candidate,
        "min_area_m2_prefilter": min_area_prefilter,
        "adaptive_geojson_limit": adaptive_geojson_limit,
        "scenes": tiling_debug_rows,
    }
    segformer_debug = {
        "schema_version": 1,
        "model_name": model_name,
        "crop_mode": crop_mode,
        "stitch_mode": stitch_mode,
        "center_size": center_size,
        "context_bounds": context_bounds,
        "inference_batch_size": inference_batch_size,
        "tile_inserts": tile_insert_debug_rows[:10000],
    }
    exported = export_pseudolabel_artifacts(
        experiment_dir=experiment_dir,
        job_id=job.job_id,
        postprocess_result=postprocess_result,
        windows_preview_features=windows_preview_features,
        tile_insert_features=tile_insert_features,
        tiling_debug=tiling_debug,
        segformer_debug=segformer_debug,
        debug_mode=debug_mode,
    )
    geojson_mb = _geojson_size_mb(exported["geojson_path"])
    gz_mb = _geojson_size_mb(exported["gz_path"])
    gpkg_mb = _geojson_size_mb(exported["gpkg_path"]) if exported["gpkg_path"].exists() else None
    postprocess_debug = build_postprocess_debug(postprocess_result, geojson_mb=geojson_mb, max_geojson_mb=max_geojson_mb)
    postprocess_debug["vertices_before"] = vertices_before
    postprocess_debug["vertices_after"] = vertices_after
    postprocess_debug["auto_tune"] = auto_tune
    postprocess_debug["selection_metric"] = "object_f1" if tune_with_object_f1 else "geojson_size"
    postprocess_debug["smooth_polygons"] = smooth_polygons
    postprocess_debug["inference_batch_size"] = inference_batch_size
    postprocess_debug["vectorization_workers"] = vectorization_workers
    postprocess_debug["max_raw_features_for_candidate"] = max_raw_features_for_candidate
    postprocess_debug["min_area_m2_prefilter"] = min_area_prefilter
    postprocess_debug["configured_max_geojson_mb"] = configured_max_geojson_mb
    postprocess_debug["effective_max_geojson_mb"] = max_geojson_mb
    postprocess_debug["adaptive_geojson_limit"] = adaptive_geojson_limit
    postprocess_debug["candidates_checked"] = candidates_checked
    postprocess_debug_path = experiment_dir / "postprocess_debug.json"
    write_json(postprocess_debug_path, postprocess_debug)

    artifacts: list[Path] = [*exported["artifacts"], pseudolabel_scenes_path, postprocess_debug_path, *preview_artifacts]
    artifacts.extend(write_prediction_examples_report(experiment_dir, job.job_id, preview_artifacts, limit=30))

    coverage_values = [float(row["coverage_fraction"]) for row in tiling_debug_rows]
    coverage_report = {
        "schema_version": 1,
        "job_id": job.job_id,
        "scenes_processed": len(tiling_debug_rows),
        "scenes_failed": 0,
        "total_expected_windows": int(sum(row["expected_window_count"] for row in tiling_debug_rows)),
        "total_predicted_windows": int(sum(row["actual_predicted_window_count"] for row in tiling_debug_rows)),
        "total_skipped_windows": int(sum(row["skipped_window_count"] for row in tiling_debug_rows)),
        "mean_coverage_fraction": float(np.mean(coverage_values)) if coverage_values else None,
        "min_coverage_fraction": float(np.min(coverage_values)) if coverage_values else None,
        "accepted_objects_total": len(postprocess_result.features),
        "accepted_geojson_mb": geojson_mb,
        "accepted_geojson_gz_mb": gz_mb,
        "accepted_gpkg_mb": gpkg_mb,
        "top_limit_applied": max_objects is not None and postprocess_result.objects_after_filter > int(max_objects),
        "top500_applied": max_objects == 500 and postprocess_result.objects_after_filter > 500,
        "max_objects": max_objects,
        "max_geojson_mb": max_geojson_mb,
        "configured_max_geojson_mb": configured_max_geojson_mb,
        "adaptive_geojson_limit": adaptive_geojson_limit,
        "smooth_polygons": smooth_polygons,
        "inference_parallel_enabled": parallel_enabled,
        "inference_max_workers": max_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
        "inference_batch_size": inference_batch_size,
        "gpu_forward_concurrency": gpu_forward_concurrency,
        "vectorization_workers": vectorization_workers,
        "max_raw_features_for_candidate": max_raw_features_for_candidate,
        "min_area_m2_prefilter": min_area_prefilter,
        "inference_duration_sec": round(time.time() - started, 3),
        "crop_mode": crop_mode,
        "stitch_mode": stitch_mode,
        "center_size": center_size,
        "context_bounds": context_bounds,
        "scenes": tiling_debug_rows,
        "warnings": [
            f"low_coverage:{row['scene_id']}:{row['coverage_fraction']:.4f}"
            for row in tiling_debug_rows
            if float(row["coverage_fraction"]) < 0.5
        ],
    }
    coverage_report_path = experiment_dir / "coverage_report.json"
    write_json(coverage_report_path, coverage_report)
    artifacts.append(coverage_report_path)

    inference_timing_report = {
        "schema_version": 1,
        "job_id": job.job_id,
        "parallel": {
            "enabled": parallel_enabled,
            "strategy": "scene" if parallel_enabled else "sequential",
            "max_workers": max_workers,
            "torch_threads_per_worker": torch_threads_per_worker,
            "batch_size": inference_batch_size,
            "gpu_forward_concurrency": gpu_forward_concurrency,
        },
        "total_inference_duration_sec": coverage_report["inference_duration_sec"],
        "per_scene": [
            {
                "scene_id": row["scene_id"],
                "expected_window_count": row["expected_window_count"],
                "actual_predicted_window_count": row["actual_predicted_window_count"],
                "scene_duration_sec": row.get("scene_duration_sec"),
                "coverage_fraction": row["coverage_fraction"],
            }
            for row in tiling_debug_rows
        ],
    }
    inference_timing_path = experiment_dir / "inference_timing_report.json"
    write_json(inference_timing_path, inference_timing_report)
    artifacts.append(inference_timing_path)

    object_metric_values, object_metric_artifacts = write_object_metrics_artifacts(
        experiment_dir,
        postprocess_result.features,
        gt_shapes,
        prefix=object_metrics_prefix,
        iou_threshold=float(post_cfg.get("object_iou_threshold") or 0.5),
    )
    artifacts.extend(object_metric_artifacts)

    total_area = float(sum(float(item["properties"].get("area_m2") or 0) for item in postprocess_result.features))
    metrics = {
        "pseudolabel_enabled": True,
        "accepted_objects": len(postprocess_result.features),
        "accepted_objects_total": len(postprocess_result.features),
        "accepted_geojson_mb": geojson_mb,
        "accepted_geojson_gz_mb": gz_mb,
        "accepted_gpkg_mb": gpkg_mb,
        "total_area_m2": total_area,
        "total_vertices": vertices_after,
        "vertices_count": vertices_after,
        "threshold_used": threshold_used,
        "min_object_area_m2_used": min_area,
        "simplify_tolerance_m_used": simplify_tolerance,
        "postprocess_sec": round(time.time() - started, 3),
        "inference_duration_sec": coverage_report["inference_duration_sec"],
        "inference_parallel_enabled": float(parallel_enabled),
        "inference_max_workers": max_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
        "inference_batch_size": inference_batch_size,
        "gpu_forward_concurrency": gpu_forward_concurrency,
        "vectorization_workers": vectorization_workers,
        "max_raw_features_for_candidate": max_raw_features_for_candidate,
        "min_area_m2_prefilter": min_area_prefilter,
        "top_limit_applied": float(max_objects is not None and postprocess_result.objects_after_filter > int(max_objects)),
        "top500_applied": float(max_objects == 500 and postprocess_result.objects_after_filter > 500),
        "pseudolabel/accepted_geojson_mb": geojson_mb,
        "pseudolabel/object_count": len(postprocess_result.features),
        "pseudolabel/vertices_count": vertices_after,
        "pseudolabel/max_geojson_mb": max_geojson_mb,
        "pseudolabel/configured_max_geojson_mb": configured_max_geojson_mb,
        "pseudolabel/adaptive_geojson_limit_enabled": float(bool(adaptive_geojson_limit.get("enabled"))),
        "pseudolabel/adaptive_area_ratio": adaptive_geojson_limit.get("area_ratio"),
    }
    metrics.update(object_metric_values)
    if tiling_debug_rows:
        metrics["expected_window_count"] = coverage_report["total_expected_windows"]
        metrics["actual_predicted_window_count"] = coverage_report["total_predicted_windows"]
        metrics["coverage_fraction"] = float(np.mean([row["coverage_fraction"] for row in tiling_debug_rows]))
        metrics["matched_scene_count"] = len(tiling_debug_rows)
        metrics["scenes_processed"] = len(tiling_debug_rows)
        metrics["total_expected_windows"] = coverage_report["total_expected_windows"]
        metrics["total_predicted_windows"] = coverage_report["total_predicted_windows"]
        metrics["mean_coverage_fraction"] = coverage_report["mean_coverage_fraction"]
        metrics["min_coverage_fraction"] = coverage_report["min_coverage_fraction"]

    summary_path = experiment_dir / "pseudolabel_summary.json"
    write_json(
        summary_path,
        {
            "metrics": metrics,
            "artifacts": [str(path) for path in artifacts],
            "postprocess_debug": postprocess_debug,
            "coverage_report": coverage_report,
        },
    )
    artifacts.append(summary_path)
    return metrics, artifacts
