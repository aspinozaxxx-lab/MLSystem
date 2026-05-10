from __future__ import annotations

import os
from typing import Any

from ...storage.local_io import read_json, write_json
from .context import StageContext
from .report import StageCheck, StageReport


def run(ctx: StageContext) -> StageReport:
    from ..airflow_tasks import _run_airflow_synthetic_pseudolabel_smoke, _run_pseudolabel_pipeline

    if ctx.config.smoke:
        smoke = _run_airflow_synthetic_pseudolabel_smoke(ctx.store)
        ctx.store.update_summary(pseudolabel_smoke=smoke)
        return StageReport(
            ctx.stage_id,
            "success",
            [StageCheck("synthetic smoke", "ok", "Synthetic pseudolabel smoke completed")],
            counters={"accepted_objects": (smoke.get("smoke_summary") or {}).get("accepted_objects")},
            artifacts={
                "accepted_geojson": str(smoke.get("accepted_geojson")),
                "prediction_examples_html": str(smoke.get("prediction_examples_html")),
            },
            details=smoke,
        )
    if not ctx.config.pseudolabel.get("enabled", False):
        return StageReport(ctx.stage_id, "skipped", [StageCheck("pseudolabel.enabled", "skipped", "pseudolabel.enabled=false")])
    if _is_inference_engine_source(ctx.config.pseudolabel):
        return _run_inference_engine_compat(ctx)

    effective_config = ctx.config
    inference_manifest = read_json(ctx.store.run_dir / "inference_manifest.json", default={}) or {}
    manifest_scenes = inference_manifest.get("scenes") or []
    run_on = str(inference_manifest.get("run_on") or (ctx.config.pseudolabel or {}).get("run_on") or "").lower()
    if manifest_scenes and run_on not in {"all_available_images", "all_images"}:
        pseudolabel_cfg = dict(ctx.config.pseudolabel or {})
        pseudolabel_cfg["scene_entries"] = [str(item.get("entry") or item.get("name")) for item in manifest_scenes if item.get("entry") or item.get("name")]
        if inference_manifest.get("bad_scene_policy"):
            pseudolabel_cfg["bad_scene_policy"] = inference_manifest.get("bad_scene_policy")
        if hasattr(ctx.config, "model_copy"):
            effective_config = ctx.config.model_copy(update={"pseudolabel": pseudolabel_cfg})
        else:
            from types import SimpleNamespace

            effective_config = SimpleNamespace(**{**vars(ctx.config), "pseudolabel": pseudolabel_cfg})

    prediction_result = _run_pseudolabel_pipeline(effective_config, ctx.store, stage_mode="inference")
    coverage = read_json(ctx.store.run_dir / "coverage_report.json", default={}) or {}
    pseudolabel = prediction_result.get("pseudolabel") or {}
    training_result = read_json(ctx.store.run_dir / "training_result.json", default={}) or {}
    manifest_path = ctx.store.run_dir / "pseudolabel_scene_results_manifest.json"
    probability_index_path = ctx.store.run_dir / "probability_maps_index.json"
    inference_results_path = ctx.store.run_dir / "inference_results.json"
    timing_report_path = ctx.store.run_dir / "inference_timing_report.json"
    probability_index = read_json(manifest_path, default={}) or {}
    write_json(probability_index_path, probability_index)
    write_json(
        inference_results_path,
        {
            "coverage": coverage,
            "pseudolabel": pseudolabel,
            "scene_results_manifest": str(manifest_path),
            "probability_maps_index": str(probability_index_path),
            "accepted_objects_status": "not_available_at_this_stage",
            "final_accepted_objects_stage": "vectorize_pseudolabel/postprocess_pseudolabel/export_pseudolabel_artifacts",
        },
    )
    scenes_requested = len(manifest_scenes) if manifest_scenes else coverage.get("scenes_total")
    scenes_processed = coverage.get("scenes_processed")
    scenes_failed = coverage.get("scenes_failed") or coverage.get("failed_scenes") or 0
    scenes_skipped = coverage.get("scenes_skipped") or coverage.get("skipped_scenes") or 0
    matching_report = read_json(ctx.store.run_dir / "scene_matching_report.json", default={}) or {}
    selected_matched_count = matching_report.get("selected_matched_count")
    matched_count = matching_report.get("matched_count")
    explicit_limit = (ctx.config.pseudolabel or {}).get("max_scenes") or (ctx.config.pseudolabel or {}).get("max_debug_scenes")
    explicit_limit_int = _safe_int(explicit_limit)
    limit_source = None
    limit_reason = None
    excluded_scenes = []
    if explicit_limit is not None:
        limit_source = "dag_run.conf.pseudolabel.max_scenes" if (ctx.config.pseudolabel or {}).get("max_scenes") is not None else "dag_run.conf.pseudolabel.max_debug_scenes"
        limit_reason = "explicit pseudolabel inference scene limit"
    requested_names = [str(item.get("entry") or item.get("name")) for item in manifest_scenes if item.get("entry") or item.get("name")]
    if explicit_limit_int is not None and explicit_limit_int >= 0 and len(requested_names) > explicit_limit_int:
        excluded_scenes = requested_names[explicit_limit_int:]
    elif isinstance(matched_count, int) and isinstance(selected_matched_count, int) and matched_count > selected_matched_count:
        selected_names = {str(item.get("entry") or item.get("name")) for item in (matching_report.get("matched") or [])}
        excluded_scenes = [name for name in requested_names if name not in selected_names]
        if limit_source is None:
            limit_source = "scene_matching_report.selected_matched_count"
            limit_reason = "pseudolabel inference selected fewer scenes than requested"
    if excluded_scenes:
        (ctx.store.run_dir / "pseudolabel_skipped_scenes.txt").write_text("\n".join(excluded_scenes) + "\n", encoding="utf-8")
    prediction_windows = coverage.get("total_predicted_windows")
    probability_maps = len(probability_index.get("scene_results") or probability_index.get("scenes") or []) if isinstance(probability_index, dict) else None
    warnings = ["accepted_objects is not available at run_pseudolabel_inference; final accepted object count belongs to vectorize/postprocess/export stages."]
    if limit_source:
        warnings.append(f"Pseudolabel inference was explicitly limited: processed={scenes_processed} of requested={scenes_requested}; source={limit_source}.")
    elif scenes_requested and scenes_processed is not None and int(scenes_processed or 0) < int(scenes_requested or 0):
        warnings.append(f"Pseudolabel inference processed fewer scenes than requested without an explicit limit: processed={scenes_processed}, requested={scenes_requested}.")
    return StageReport(
        ctx.stage_id,
        "success",
        [StageCheck("GPU inference", "ok", "Triton direct pseudolabel inference completed")],
        counters={
            "backend": "direct_triton",
            "scene_count": scenes_requested,
            "scenes_processed": scenes_processed,
            "scenes_failed": scenes_failed,
            "scenes_skipped": scenes_skipped,
            "total_predicted_windows": prediction_windows,
            "mean_coverage_fraction": coverage.get("mean_coverage_fraction"),
            "pseudolabel_scenes_requested": scenes_requested,
            "pseudolabel_scenes_processed": scenes_processed,
            "pseudolabel_scenes_failed": scenes_failed,
            "pseudolabel_scenes_skipped": scenes_skipped,
            "pseudolabel_prediction_windows": prediction_windows,
            "probability_maps": probability_maps,
            "inference_scene_limit": explicit_limit,
            "pseudolabel_scenes_excluded": len(excluded_scenes) if excluded_scenes else (max(0, int(scenes_requested or 0) - int(scenes_processed or 0)) if scenes_requested is not None and scenes_processed is not None else None),
            "limit_source": limit_source,
            "limit_reason": limit_reason,
            "device": training_result.get("device"),
            "cuda_available": training_result.get("cuda_available"),
            "gpu_name": training_result.get("gpu_name"),
        },
        artifacts={
            "inference_results.json": str(inference_results_path),
            "probability_maps_index.json": str(probability_index_path),
            "pseudolabel_scene_results_manifest.json": str(manifest_path),
            "inference_timing_report.json": str(timing_report_path),
            **({"pseudolabel_skipped_scenes.txt": str(ctx.store.run_dir / "pseudolabel_skipped_scenes.txt")} if excluded_scenes else {}),
        },
        warnings=warnings,
        details={
            "inference_input_source": str(ctx.store.run_dir / "inference_manifest.json"),
            "triton_path": "direct",
            "accepted_objects": "not_available_at_this_stage",
            "final_accepted_objects_stage": "vectorize_pseudolabel/postprocess_pseudolabel/export_pseudolabel_artifacts",
            "thresholds": (getattr(ctx.config, "postprocess", {}) or {}),
            "limit": {
                "applied": bool(limit_source),
                "source": limit_source,
                "reason": limit_reason,
                "requested": scenes_requested,
                "processed": scenes_processed,
                "excluded_scenes_artifact": str(ctx.store.run_dir / "pseudolabel_skipped_scenes.txt") if excluded_scenes else None,
            },
        },
        summary="GPU pseudolabel inference completed; CPU vectorization/postprocess are separate compatibility stages.",
    )


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_inference_engine_source(pseudolabel_cfg: dict[str, Any]) -> bool:
    source = str(pseudolabel_cfg.get("source") or pseudolabel_cfg.get("engine") or "").strip().lower().replace("-", "_")
    return source == "inference_engine"


def _run_inference_engine_compat(ctx: StageContext) -> StageReport:
    from ...orchestration.inference_engine_client import InferenceEngineClient

    inference_manifest = read_json(ctx.store.run_dir / "inference_manifest.json", default={}) or {}
    payload = _build_inference_engine_payload(ctx, inference_manifest)
    client = InferenceEngineClient()
    created = client.create_job(payload)
    poll_sec = float(os.getenv("INFERENCE_ENGINE_AIRFLOW_POLL_SEC") or "10")
    timeout_sec = float(os.getenv("INFERENCE_ENGINE_AIRFLOW_TIMEOUT_SEC") or str(24 * 3600))
    final_state = client.wait(str(created["job_id"]), poll_sec=poll_sec, timeout_sec=timeout_sec)
    if final_state.get("status") != "success":
        return StageReport(
            ctx.stage_id,
            "failed",
            [StageCheck("InferenceEngine job", "failed", str(final_state.get("error") or final_state.get("status")))],
            errors=[str(final_state.get("error") or f"InferenceEngine job {created['job_id']} ended with {final_state.get('status')}")],
            details={"job": final_state},
        )
    artifacts = final_state.get("artifacts") or {}
    metrics = final_state.get("metrics") or {}
    coverage = read_json(ctx.store.run_dir / "coverage_report.json", default={}) or {}
    probability_index = read_json(ctx.store.run_dir / "pseudolabel_scene_results_manifest.json", default={}) or {}
    probability_index_path = ctx.store.run_dir / "probability_maps_index.json"
    if probability_index:
        write_json(probability_index_path, probability_index)
    inference_results_path = ctx.store.run_dir / "inference_results.json"
    if not inference_results_path.exists():
        write_json(
            inference_results_path,
            {
                "source": "inference_engine",
                "job_id": created["job_id"],
                "coverage": coverage,
                "probability_maps_index": str(probability_index_path),
            },
        )
    return StageReport(
        ctx.stage_id,
        "success",
        [StageCheck("InferenceEngine job", "ok", f"job_id={created['job_id']}")],
        counters={
            "backend": "inference_engine",
            "scene_count": coverage.get("scenes_processed"),
            "scenes_processed": coverage.get("scenes_processed"),
            "scenes_failed": coverage.get("scenes_failed") or 0,
            "scenes_skipped": 0,
            "total_predicted_windows": coverage.get("total_predicted_windows"),
            "probability_maps": len((probability_index.get("scenes") or [])) if isinstance(probability_index, dict) else None,
            "tiles_total": metrics.get("tiles_total"),
            "tiles_done": metrics.get("tiles_done"),
            "blocks_total": metrics.get("blocks_total"),
            "blocks_done": metrics.get("blocks_done"),
            "triton_batches": metrics.get("triton_batches"),
        },
        artifacts={
            "inference_results.json": str(inference_results_path),
            "probability_maps_index.json": str(probability_index_path),
            "pseudolabel_scene_results_manifest.json": str(ctx.store.run_dir / "pseudolabel_scene_results_manifest.json"),
            "coverage_report.json": str(ctx.store.run_dir / "coverage_report.json"),
            **{str(name): str(path) for name, path in artifacts.items() if isinstance(path, str)},
        },
        details={"inference_engine_job": final_state},
        summary="InferenceEngine pseudolabel pipeline completed; downstream stages are validate-only for source=inference_engine.",
    )


def _build_inference_engine_payload(ctx: StageContext, inference_manifest: dict[str, Any]) -> dict[str, Any]:
    pseudolabel_cfg = dict(ctx.config.pseudolabel or {})
    vector_cfg = dict(pseudolabel_cfg.get("vectorization") or {})
    preprocess_cfg = dict(ctx.config.preprocess or {})
    inference_cfg = dict(ctx.config.inference or {})
    model_cfg = dict(ctx.config.model or {})
    scenes = inference_manifest.get("scenes") or []
    return {
        "run_id": ctx.run_id,
        "experiment_id": ctx.config.experiment_id,
        "scenes": scenes,
        "inference_manifest": str(ctx.store.run_dir / "inference_manifest.json"),
        "images_uri": ctx.config.images_uri,
        "layout_uri": ctx.config.layout_uri,
        "run_dir": str(ctx.store.run_dir),
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
            "threshold": vector_cfg.get("threshold") or (ctx.config.postprocess or {}).get("threshold") or 0.5,
            "core_size_px": vector_cfg.get("core_size_px") or 4096,
            "halo_px": vector_cfg.get("halo_px") or 512,
            "workers": vector_cfg.get("workers") or 4,
            "local_min_area": vector_cfg.get("local_min_area") or 0,
            "final_min_area": vector_cfg.get("final_min_area") or (ctx.config.postprocess or {}).get("min_area_m2") or 0,
            "merge_epsilon": vector_cfg.get("merge_epsilon") if vector_cfg.get("merge_epsilon") is not None else 1.0,
            "simplify_tolerance": (ctx.config.postprocess or {}).get("simplify_tolerance_m") or 0,
            "max_objects": (ctx.config.postprocess or {}).get("max_objects"),
        },
        "resource": {
            "triton_batch_size": inference_cfg.get("triton_batch_size") or pseudolabel_cfg.get("batch_size") or 8,
            "batches_ahead": inference_cfg.get("batches_ahead") or 4,
            "max_preprocess_queue": inference_cfg.get("max_preprocess_queue") or 512,
            "max_spool_bytes": inference_cfg.get("max_spool_bytes") or 20 * 1024 * 1024 * 1024,
            "max_scenes_inflight": inference_cfg.get("max_scenes_inflight") or 1,
            "max_blocks_inflight": inference_cfg.get("max_blocks_inflight") or 8,
        },
        "max_scenes": pseudolabel_cfg.get("max_scenes") or pseudolabel_cfg.get("max_debug_scenes"),
        "source": "inference_engine",
    }
