from __future__ import annotations

import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from ..debug.pseudolabel_debug import build_postprocess_debug, write_probability_preview
from ..inference.scene_inference import SceneInferenceConfig, SceneInferenceRunner
from ..job_schema import JobSpec
from ..metrics.object_metric_artifacts import write_object_metrics_artifacts
from ..pipeline.contracts import VectorizationResult
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
    threshold_used = float(thresholds[0])
    max_objects = int(post_cfg.get("max_objects") or 500)
    min_area_candidates = post_cfg.get("min_object_area_m2_candidates") or [1000]
    simplify_candidates = post_cfg.get("simplify_tolerance_m_candidates") or [5]
    min_area = float(min_area_candidates[0])
    simplify_tolerance = float(simplify_candidates[0])
    full_scene = bool(pseudolabel_cfg.get("full_scene", True))
    max_windows_per_scene = pseudolabel_cfg.get("max_windows_per_scene")
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

    started = time.time()
    model.eval()
    previous_torch_threads = torch.get_num_threads()
    parallel_cfg = ((job.predict.get("inference") or {}).get("parallel") or (job.params.get("inference") or {}).get("parallel") or {})
    parallel_enabled = bool(parallel_cfg.get("enabled", False))
    max_workers = max(1, int(parallel_cfg.get("max_workers") or 1))
    torch_threads_per_worker = max(1, int(parallel_cfg.get("torch_threads_per_worker") or max(1, torch.get_num_threads() // max_workers)))
    torch.set_num_threads(torch_threads_per_worker)

    runner = SceneInferenceRunner(
        config,
        model,
        device,
        SceneInferenceConfig(
            input_bands=input_bands,
            patch_size=patch_size,
            stride=int(job.preprocess.get("stride") or patch_size),
            threshold=threshold_used,
            crop_mode=crop_mode,
            stitch_mode=stitch_mode,
            center_size=center_size,
            context_bounds=context_bounds,
            full_scene=full_scene,
            max_windows_per_scene=max_windows_per_scene,
        ),
    )

    scene_results = []
    try:
        # The previous implementation could run scenes in parallel. This keeps
        # deterministic behavior in the refactor step; the config is still
        # reported and can be re-enabled inside SceneInferenceRunner later.
        scene_results = [runner.run_scene(idx, match) for idx, match in enumerate(matches)]
    finally:
        torch.set_num_threads(previous_torch_threads)

    all_raw_features: list[dict[str, Any]] = []
    windows_preview_features: list[dict[str, Any]] = []
    tile_insert_features: list[dict[str, Any]] = []
    tile_insert_debug_rows: list[dict[str, Any]] = []
    tiling_debug_rows: list[dict[str, Any]] = []
    vertices_before = 0
    preview_artifacts: list[Path] = []
    for result in sorted(scene_results, key=lambda item: item.scene_index):
        vectorized = vectorize_probability_map(result.probability_map, scene_name=result.scene_name, threshold=threshold_used)
        all_raw_features.extend(vectorized.features_raw)
        vertices_before += vectorized.vertices_before
        result.debug_row["objects_before_filter"] = vectorized.raw_count
        result.debug_row["objects_after_filter"] = vectorized.raw_count
        windows_preview_features.extend(result.windows_preview_features)
        tile_insert_features.extend(result.tile_insert_features)
        tile_insert_debug_rows.extend(result.tile_insert_debug_rows)
        tiling_debug_rows.append(result.debug_row)
        if not preview_artifacts:
            preview = write_probability_preview(result.probability_map.prob, result.scene_name, experiment_dir)
            if preview:
                preview_artifacts.append(preview)

    vectorization = VectorizationResult(
        features_raw=all_raw_features,
        raw_count=len(all_raw_features),
        vertices_before=vertices_before,
        threshold=threshold_used,
        crs="EPSG:3857",
        metadata={"scene_count": len(scene_results)},
    )
    postprocess_result = postprocess_vectorization_result(
        vectorization,
        min_area_m2=min_area,
        simplify_tolerance_m=simplify_tolerance,
        max_objects=max_objects,
    )
    vertices_after = int(sum(vertex_count(item["geometry"]) for item in postprocess_result.features))
    tiling_debug = {
        "schema_version": 1,
        "full_scene": full_scene,
        "crop_mode": crop_mode,
        "stitch_mode": stitch_mode,
        "center_size": center_size,
        "context_bounds": context_bounds,
        "scenes": tiling_debug_rows,
    }
    segformer_debug = {
        "schema_version": 1,
        "model_name": model_name,
        "crop_mode": crop_mode,
        "stitch_mode": stitch_mode,
        "center_size": center_size,
        "context_bounds": context_bounds,
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
    max_geojson_mb = float(post_cfg.get("max_geojson_mb") or 20)
    postprocess_debug = build_postprocess_debug(postprocess_result, geojson_mb=geojson_mb, max_geojson_mb=max_geojson_mb)
    postprocess_debug["vertices_before"] = vertices_before
    postprocess_debug["vertices_after"] = vertices_after
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
        "top500_applied": postprocess_result.objects_after_filter > max_objects,
        "max_objects": max_objects,
        "max_geojson_mb": max_geojson_mb,
        "inference_parallel_enabled": parallel_enabled,
        "inference_max_workers": max_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
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
        "threshold_used": threshold_used,
        "min_object_area_m2_used": min_area,
        "simplify_tolerance_m_used": simplify_tolerance,
        "postprocess_sec": round(time.time() - started, 3),
        "inference_duration_sec": coverage_report["inference_duration_sec"],
        "inference_parallel_enabled": float(parallel_enabled),
        "inference_max_workers": max_workers,
        "torch_threads_per_worker": torch_threads_per_worker,
        "top500_applied": float(postprocess_result.objects_after_filter > max_objects),
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
