from __future__ import annotations

"""Production Airflow pseudolabel stage.

This stage is intentionally orchestration-only: it submits an HTTP job to
InferenceEngine, polls the job, and validates compatibility artifacts.
"""

import os
from pathlib import Path
from typing import Any

from ...storage.local_io import read_json, write_json
from .context import StageContext
from .prepare_inference_scenes import run as prepare_inference_manifest
from .report import StageCheck, StageFailure, StageReport


COMPATIBILITY_ARTIFACTS = [
    "accepted.geojson.gz",
    "coverage_report.json",
    "pseudolabel_summary.json",
    "postprocess_summary.json",
    "vectorization_summary.json",
    "pseudolabel_scene_results_manifest.json",
    "probability_maps_index.json",
    "inference_results.json",
    "inference_timing_report.json",
    "pseudolabel_scenes.txt",
    "prediction_examples.html",
]

AIRFLOW_CONTAINER_RUN_ROOT = Path("/opt/airflow/mlsystem_runs")
DEFAULT_SHARED_RUN_ROOT = Path("/data/mlsystem/airflow/status")


def run(ctx: StageContext) -> StageReport:
    pseudolabel_cfg = dict(ctx.config.pseudolabel or {})
    if "enabled" in pseudolabel_cfg and not bool(pseudolabel_cfg.get("enabled")):
        return StageReport(
            ctx.stage_id,
            "skipped",
            [StageCheck("pseudolabel.enabled", "skipped", "pseudolabel.enabled=false")],
            summary="InferenceEngine pipeline skipped because pseudolabel.enabled=false.",
        )

    warnings: list[str] = []
    if not _is_inference_engine_source(pseudolabel_cfg):
        warnings.append("pseudolabel.source was not inference_engine; production Airflow now routes pseudolabels to InferenceEngine.")

    inference_manifest = _ensure_inference_manifest(ctx)
    payload = _build_payload(ctx, inference_manifest)

    from ...orchestration.inference_engine_client import InferenceEngineClient

    client = InferenceEngineClient()
    created = client.create_job(payload)
    job_id = str(created["job_id"])
    poll_sec = float(os.getenv("INFERENCE_ENGINE_AIRFLOW_POLL_SEC") or "10")
    timeout_sec = float(os.getenv("INFERENCE_ENGINE_AIRFLOW_TIMEOUT_SEC") or str(24 * 3600))
    final_state = client.wait(job_id, poll_sec=poll_sec, timeout_sec=timeout_sec)
    artifacts_response: dict[str, Any] = {}
    try:
        artifacts_response = client.get_artifacts(job_id)
    except Exception as exc:  # noqa: BLE001 - job status already contains enough context.
        warnings.append(f"InferenceEngine artifacts endpoint was not readable after job completion: {type(exc).__name__}: {exc}")

    status_url = f"{client.base_url}/api/v1/jobs/{job_id}"
    artifacts_url = f"{client.base_url}/api/v1/jobs/{job_id}/artifacts"
    if final_state.get("status") != "success":
        return StageReport(
            ctx.stage_id,
            "failed",
            [StageCheck("InferenceEngine HTTP job", "failed", str(final_state.get("error") or final_state.get("status")))],
            errors=[str(final_state.get("error") or f"InferenceEngine job {job_id} ended with {final_state.get('status')}")],
            warnings=warnings,
            details={
                "source": "inference_engine",
                "request_submitted_via_http": True,
                "inference_engine_api_url": client.base_url,
                "inference_engine_job_id": job_id,
                "inference_engine_status_url": status_url,
                "inference_engine_artifacts_url": artifacts_url,
                "inference_engine_job": final_state,
            },
        )

    metrics = final_state.get("metrics") or {}
    coverage = read_json(ctx.store.run_dir / "coverage_report.json", default={}) or {}
    probability_index = read_json(ctx.store.run_dir / "pseudolabel_scene_results_manifest.json", default={}) or {}
    probability_index_path = ctx.store.run_dir / "probability_maps_index.json"
    if probability_index and not probability_index_path.exists():
        write_json(probability_index_path, probability_index)
    inference_results_path = ctx.store.run_dir / "inference_results.json"
    if not inference_results_path.exists():
        write_json(
            inference_results_path,
            {
                "source": "inference_engine",
                "job_id": job_id,
                "coverage": coverage,
                "probability_maps_index": str(probability_index_path),
            },
        )

    required = list(COMPATIBILITY_ARTIFACTS)
    required.append(f"{ctx.config.experiment_id}.accepted.geojson")
    missing = [name for name in required if not (ctx.store.run_dir / name).exists()]
    if missing:
        return StageReport(
            ctx.stage_id,
            "failed",
            [StageCheck("compatibility artifacts", "failed", f"missing={missing}")],
            errors=[f"InferenceEngine job {job_id} succeeded but compatibility artifacts are missing: {missing}"],
            warnings=warnings,
            details={
                "source": "inference_engine",
                "request_submitted_via_http": True,
                "inference_engine_api_url": client.base_url,
                "inference_engine_job_id": job_id,
                "inference_engine_status_url": status_url,
                "inference_engine_artifacts_url": artifacts_url,
                "inference_engine_job": final_state,
                "inference_engine_artifacts_response": artifacts_response,
            },
        )

    ctx.store.update_summary(
        inference_engine={
            "source": "inference_engine",
            "job_id": job_id,
            "status_url": status_url,
            "artifacts_url": artifacts_url,
            "request_submitted_via_http": True,
        }
    )
    artifacts = {
        "inference_results.json": str(inference_results_path),
        "probability_maps_index.json": str(probability_index_path),
        "pseudolabel_scene_results_manifest.json": str(ctx.store.run_dir / "pseudolabel_scene_results_manifest.json"),
        "coverage_report.json": str(ctx.store.run_dir / "coverage_report.json"),
        "accepted_geojson": str(ctx.store.run_dir / f"{ctx.config.experiment_id}.accepted.geojson"),
        "accepted_geojson_gz": str(ctx.store.run_dir / "accepted.geojson.gz"),
        "prediction_examples.html": str(ctx.store.run_dir / "prediction_examples.html"),
    }
    for name, value in (final_state.get("artifacts") or {}).items():
        if isinstance(value, str):
            artifacts[str(name)] = value

    return StageReport(
        ctx.stage_id,
        "success",
        [
            StageCheck("InferenceEngine HTTP job", "ok", f"POST /api/v1/jobs job_id={job_id}"),
            StageCheck("compatibility artifacts", "ok", f"{len(required)} artifacts validated in run_dir"),
        ],
        counters={
            "backend": "inference_engine",
            "source": "inference_engine",
            "request_submitted_via_http": True,
            "inference_engine_http_submitted": 1,
            "inference_engine_job_id": job_id,
            "scene_count": coverage.get("scenes_processed"),
            "pseudolabel_scenes_processed": coverage.get("scenes_processed"),
            "pseudolabel_scenes_failed": coverage.get("scenes_failed") or 0,
            "pseudolabel_scenes_skipped": coverage.get("scenes_skipped") or 0,
            "pseudolabel_prediction_windows": coverage.get("total_predicted_windows"),
            "probability_maps": len((probability_index.get("scenes") or [])) if isinstance(probability_index, dict) else None,
            "tiles_total": metrics.get("tiles_total"),
            "tiles_done": metrics.get("tiles_done"),
            "blocks_total": metrics.get("blocks_total"),
            "blocks_done": metrics.get("blocks_done"),
            "triton_batches": metrics.get("triton_batches"),
        },
        artifacts=artifacts,
        details={
            "source": "inference_engine",
            "request_submitted_via_http": True,
            "inference_engine_api_url": client.base_url,
            "inference_engine_job_id": job_id,
            "inference_engine_status_url": status_url,
            "inference_engine_artifacts_url": artifacts_url,
            "inference_engine_job": final_state,
            "inference_engine_artifacts_response": artifacts_response,
            "compatibility_artifacts_checked": required,
            "payload": {
                "experiment_id": payload.get("experiment_id"),
                "run_id": payload.get("run_id"),
                "scene_count": len(payload.get("scenes") or []),
                "triton_model_name": ((payload.get("model") or {}).get("triton_model_name")),
                "max_scenes": payload.get("max_scenes"),
            },
        },
        warnings=warnings,
        summary="InferenceEngine completed the full pseudolabel pipeline via HTTP API and wrote Airflow-compatible artifacts.",
    )


def _ensure_inference_manifest(ctx: StageContext) -> dict[str, Any]:
    manifest_path = ctx.store.run_dir / "inference_manifest.json"
    manifest = read_json(manifest_path, default={}) or {}
    if manifest.get("scenes"):
        return manifest
    try:
        prepare_report = prepare_inference_manifest(ctx)
    except StageFailure:
        raise
    if prepare_report.status == "failed":
        raise StageFailure("inference_engine_pipeline failed to prepare inference manifest", prepare_report)
    return read_json(manifest_path, default={}) or {}


def _is_inference_engine_source(pseudolabel_cfg: dict[str, Any]) -> bool:
    source = str(pseudolabel_cfg.get("source") or pseudolabel_cfg.get("engine") or "").strip().lower().replace("-", "_")
    return source in {"", "inference_engine"}


def _cfg_dict(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _shared_run_dir_for_inference_engine(run_dir: Path) -> Path:
    shared_root = Path(os.getenv("MLSYSTEM_AIRFLOW_STATUS_ROOT") or os.getenv("MLSYSTEM_AIRFLOW_STATE_DIR") or DEFAULT_SHARED_RUN_ROOT)
    try:
        return shared_root / run_dir.relative_to(AIRFLOW_CONTAINER_RUN_ROOT)
    except ValueError:
        return run_dir


def _build_payload(ctx: StageContext, inference_manifest: dict[str, Any]) -> dict[str, Any]:
    pseudolabel_cfg = _cfg_dict(ctx.config.pseudolabel)
    vector_cfg = _cfg_dict(pseudolabel_cfg.get("vectorization"))
    preprocess_cfg = _cfg_dict(ctx.config.preprocess)
    inference_cfg = _cfg_dict(getattr(ctx.config, "inference", {}))
    model_cfg = _cfg_dict(getattr(ctx.config, "model", {}))
    postprocess_cfg = _cfg_dict(getattr(ctx.config, "postprocess", {}))
    scenes = inference_manifest.get("scenes") or []
    shared_run_dir = _shared_run_dir_for_inference_engine(ctx.store.run_dir)
    return {
        "run_id": ctx.run_id,
        "experiment_id": ctx.config.experiment_id,
        "scenes": scenes,
        "inference_manifest": str(shared_run_dir / "inference_manifest.json"),
        "images_uri": ctx.config.images_uri,
        "layout_uri": ctx.config.layout_uri,
        "run_dir": str(shared_run_dir),
        "storage": {
            "source": "airflow",
            "state_dir": str(ctx.status_dir),
        },
        "model": {
            "mlflow_run_id": model_cfg.get("mlflow_run_id") or pseudolabel_cfg.get("mlflow_run_id") or "a7838f91528a47e1931b685c2ea06686",
            "model_name": model_cfg.get("name") or model_cfg.get("model_name") or "segformer_b2",
            "architecture": model_cfg.get("architecture") or "segformer_b2",
            "triton_model_name": inference_cfg.get("triton_model_name") or pseudolabel_cfg.get("triton_model_name") or "segformer_b2",
        },
        "preprocess": {
            "tile_size": preprocess_cfg.get("tile_size") or preprocess_cfg.get("patch_size") or pseudolabel_cfg.get("tile_size"),
            "patch_size": preprocess_cfg.get("patch_size") or pseudolabel_cfg.get("patch_size") or 1024,
            "stride": preprocess_cfg.get("stride") or pseudolabel_cfg.get("stride") or 768,
            "input_bands": preprocess_cfg.get("input_bands") or pseudolabel_cfg.get("input_bands") or [1, 2, 3, 4],
            "crop_mode": pseudolabel_cfg.get("crop_mode") or "full",
            "center_size": pseudolabel_cfg.get("center_size"),
            "context_bounds": pseudolabel_cfg.get("context_bounds"),
            "stitch_mode": pseudolabel_cfg.get("stitch_mode") or "weighted_overlap",
        },
        "pseudolabel": {
            "threshold": vector_cfg.get("threshold") or postprocess_cfg.get("threshold") or 0.5,
            "core_size_px": vector_cfg.get("core_size_px") or 4096,
            "halo_px": vector_cfg.get("halo_px") or 512,
            "workers": vector_cfg.get("workers") or 4,
            "local_min_area": vector_cfg.get("local_min_area") or 0,
            "final_min_area": vector_cfg.get("final_min_area") or postprocess_cfg.get("min_area_m2") or 0,
            "merge_epsilon": vector_cfg.get("merge_epsilon") if vector_cfg.get("merge_epsilon") is not None else 1.0,
            "simplify_tolerance": postprocess_cfg.get("simplify_tolerance_m") or 0,
            "max_objects": postprocess_cfg.get("max_objects"),
        },
        "resource": {
            "triton_batch_size": inference_cfg.get("triton_batch_size") or pseudolabel_cfg.get("batch_size") or 8,
            "batches_ahead": inference_cfg.get("batches_ahead") or 16,
            "max_preprocess_queue": inference_cfg.get("max_preprocess_queue") or 1024,
            "max_spool_bytes": inference_cfg.get("max_spool_bytes") or 20 * 1024 * 1024 * 1024,
            "max_scenes_inflight": inference_cfg.get("max_scenes_inflight") or 4,
            "max_blocks_inflight": inference_cfg.get("max_blocks_inflight") or 8,
        },
        "max_scenes": pseudolabel_cfg.get("max_scenes") or pseudolabel_cfg.get("max_debug_scenes"),
        "source": "inference_engine",
    }
