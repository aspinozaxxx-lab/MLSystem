from __future__ import annotations

from ...storage.local_io import read_json, write_json
from ...vectorization import run_block_parallel_vectorization
from .context import StageContext
from .report import StageCheck, StageReport


def run(ctx: StageContext) -> StageReport:
    from ..airflow_tasks import _run_pseudolabel_pipeline

    if "enabled" in ctx.config.pseudolabel and not bool(ctx.config.pseudolabel.get("enabled")) and not ctx.config.smoke:
        return StageReport(ctx.stage_id, "skipped", [StageCheck("pseudolabel.enabled", "skipped", "pseudolabel.enabled=false")])
    accepted = ctx.store.run_dir / f"{ctx.config.experiment_id}.accepted.geojson"
    if _is_inference_engine_source(ctx.config.pseudolabel):
        summary_path = ctx.store.run_dir / "vectorization_summary.json"
        summary = read_json(summary_path, default={}) or {}
        if not accepted.exists():
            return StageReport(
                ctx.stage_id,
                "failed",
                [StageCheck("InferenceEngine accepted_geojson", "failed", f"missing={accepted}")],
                errors=[f"InferenceEngine accepted GeoJSON is missing: {accepted}"],
            )
        return StageReport(
            ctx.stage_id,
            "success",
            [StageCheck("InferenceEngine vectorization", "ok", "Artifacts already produced by InferenceEngine")],
            counters={
                "vectorization_mode": "inference_engine_validate_only",
                "accepted_objects": ((summary.get("pseudolabel") or {}).get("accepted_objects") if isinstance(summary, dict) else None),
                "blocks_total": ((summary.get("vectorization") or {}).get("blocks_total") if isinstance(summary, dict) else None),
                "blocks_done": ((summary.get("vectorization") or {}).get("blocks_done") if isinstance(summary, dict) else None),
            },
            artifacts={"accepted_geojson": str(accepted), "vectorization_summary.json": str(summary_path)},
            details={"source": "inference_engine", "summary": summary},
            summary="InferenceEngine vectorization artifacts validated; heavy vectorization skipped in mlsystem.",
        )
    vector_cfg = dict((ctx.config.pseudolabel or {}).get("vectorization") or {})
    mode = str(vector_cfg.get("mode") or "legacy").lower()
    if mode == "block_parallel":
        run_on = str((ctx.config.pseudolabel or {}).get("run_on") or "").lower()
        if run_on in {"all_images", "all_available_images"} and not bool(vector_cfg.get("allow_all_images")):
            return StageReport(
                ctx.stage_id,
                "failed",
                [StageCheck("all_images guard", "failed", "block_parallel vectorization for all_images requires pseudolabel.vectorization.allow_all_images=true")],
                errors=["all_images block_parallel run is blocked without explicit allow_all_images=true"],
            )
        manifest_path = ctx.store.run_dir / "pseudolabel_scene_results_manifest.json"
        if not manifest_path.exists():
            return StageReport(
                ctx.stage_id,
                "failed",
                [StageCheck("probability manifest", "failed", "pseudolabel_scene_results_manifest.json is missing")],
                errors=[f"Missing probability manifest: {manifest_path}"],
            )
        threshold = float(vector_cfg.get("threshold") or (ctx.config.postprocess or {}).get("threshold") or 0.5)
        summary = run_block_parallel_vectorization(
            run_id=ctx.run_id,
            manifest_path=manifest_path,
            output_dir=ctx.store.run_dir / "vectorization_block_parallel",
            accepted_geojson=accepted,
            threshold=threshold,
            class_name=ctx.config.class_name or "deforest",
            core_size_px=int(vector_cfg.get("core_size_px") or 4096),
            halo_px=int(vector_cfg.get("halo_px") or 512),
            workers_requested=int(vector_cfg.get("workers") or 30),
            memory_guard_enabled=bool(vector_cfg.get("memory_guard_enabled", True)),
            max_worker_memory_mb=(int(vector_cfg["max_worker_memory_mb"]) if vector_cfg.get("max_worker_memory_mb") is not None else None),
            local_min_area=float(vector_cfg.get("local_min_area") or 0.0),
            final_min_area=float(vector_cfg.get("final_min_area") or (ctx.config.postprocess or {}).get("min_area_m2") or 0.0),
            merge_epsilon=float(vector_cfg.get("merge_epsilon") if vector_cfg.get("merge_epsilon") is not None else 1.0),
            bad_block_policy=str(vector_cfg.get("bad_block_policy") or "fail"),
        )
        accepted_geojson_mb = summary.get("final_geojson_size_mb")
        metrics = {
            "accepted_objects": summary.get("accepted_objects"),
            "threshold_used": threshold,
            "min_object_area_m2_used": float(vector_cfg.get("final_min_area") or (ctx.config.postprocess or {}).get("min_area_m2") or 0.0),
            "simplify_tolerance_m_used": None,
            "accepted_geojson_mb": accepted_geojson_mb,
            "vectorization_mode": "block_parallel",
        }
        root_summary_path = ctx.store.run_dir / "pseudolabel_summary.json"
        root_vectorization_summary_path = ctx.store.run_dir / "vectorization_summary.json"
        write_json(
            root_summary_path,
            {
                "status": "success",
                "mode": "block_parallel",
                "accepted_geojson": str(accepted),
                "metrics": metrics,
                "vectorization": summary,
            },
        )
        write_json(
            root_vectorization_summary_path,
            {
                "accepted_geojson": str(accepted),
                "pseudolabel": {
                    "accepted_objects": summary.get("accepted_objects"),
                    "accepted_geojson_mb": accepted_geojson_mb,
                },
                "vectorization": summary,
            },
        )
        examples_path = ctx.store.run_dir / "prediction_examples.html"
        if not examples_path.exists():
            examples_path.write_text(
                "<!doctype html><html><body><h1>MLSystem block_parallel pseudolabel</h1>"
                f"<p>accepted_objects={summary.get('accepted_objects')}</p>"
                f"<p>accepted_geojson={accepted.name}</p>"
                "</body></html>",
                encoding="utf-8",
            )
        training_result_path = ctx.store.run_dir / "training_result.json"
        training_result = read_json(training_result_path, default={}) or {}
        training_result["postprocess_metrics"] = metrics
        training_result["pseudolabel"] = {
            "accepted_objects": summary.get("accepted_objects"),
            "accepted_geojson_mb": accepted_geojson_mb,
            "accepted_geojson": accepted.name,
            "vectorization_mode": "block_parallel",
        }
        write_json(training_result_path, training_result)
        ctx.store.update_summary(training_result=training_result)
        warnings = []
        if (summary.get("memory_guard") or {}).get("reduced"):
            warnings.append("memory guard reduced workers_effective")
        for ratio_key in ("area_ratio_after_merge_to_before_merge", "area_ratio_final_to_before_merge"):
            ratio = summary.get(ratio_key)
            if ratio is not None and (float(ratio) < 0.5 or float(ratio) > 1.5):
                warnings.append(f"{ratio_key}={ratio} is outside expected QA range [0.5, 1.5]")
        return StageReport(
            ctx.stage_id,
            "success",
            [StageCheck("CPU block vectorization", "ok", "block/core/halo vectorization completed")],
            counters={
                "vectorization_mode": "block_parallel",
                "prediction_tiles": summary.get("prediction_tiles"),
                "prediction_scenes": summary.get("prediction_scenes"),
                "blocks_total": summary.get("blocks_total"),
                "blocks_done": summary.get("blocks_done"),
                "blocks_failed": summary.get("blocks_failed"),
                "workers_requested": summary.get("workers_requested"),
                "workers_effective": summary.get("workers_effective"),
                "boundary_candidates": summary.get("boundary_candidates_count"),
                "polygons_before_merge": summary.get("polygons_before_merge"),
                "polygons_after_merge": summary.get("polygons_after_merge"),
                "area_ratio_after_merge_to_before_merge": summary.get("area_ratio_after_merge_to_before_merge"),
                "area_ratio_final_to_before_merge": summary.get("area_ratio_final_to_before_merge"),
                "accepted_objects": summary.get("accepted_objects"),
                "final_objects": summary.get("final_objects"),
                "final_geojson_size_mb": summary.get("final_geojson_size_mb"),
            },
            warnings=warnings,
            artifacts={
                "accepted_geojson": str(accepted),
                "pseudolabel_summary.json": str(root_summary_path),
                "prediction_examples.html": str(examples_path),
                "vectorization_summary.json": str(root_vectorization_summary_path),
                "block_vectorization_summary.json": str(ctx.store.run_dir / "vectorization_block_parallel" / "vectorization_summary.json"),
                "vectorization_plan.json": str(ctx.store.run_dir / "vectorization_block_parallel" / "vectorization_plan.json"),
                "block_results.json": str(ctx.store.run_dir / "vectorization_block_parallel" / "block_results.json"),
                "processing_blocks.geojson": str(ctx.store.run_dir / "vectorization_block_parallel" / "processing_blocks.geojson"),
            },
            details={
                "metrics": {
                    "vectorization_duration_sec": summary.get("vectorization_duration_sec"),
                    "merge_duration_sec": summary.get("merge_duration_sec"),
                    "area_ratio_after_merge_to_before_merge": summary.get("area_ratio_after_merge_to_before_merge"),
                    "area_ratio_final_to_before_merge": summary.get("area_ratio_final_to_before_merge"),
                },
                "summary": summary,
            },
            summary="CPU block_parallel vectorization completed from saved probability maps.",
        )
    if not accepted.exists():
        prediction_result = _run_pseudolabel_pipeline(ctx.config, ctx.store, stage_mode="postprocess")
        coverage = read_json(ctx.store.run_dir / "coverage_report.json", default={}) or {}
        pseudolabel = prediction_result.get("pseudolabel") or {}
    else:
        coverage = read_json(ctx.store.run_dir / "coverage_report.json", default={}) or {}
        pseudolabel = {}
    summary_path = ctx.store.run_dir / "vectorization_summary.json"
    write_json(summary_path, {"accepted_geojson": str(accepted), "coverage": coverage, "pseudolabel": pseudolabel})
    return StageReport(
        ctx.stage_id,
        "success",
        [StageCheck("CPU vectorization", "ok", "Accepted GeoJSON is available")],
        counters={
            "vectorization_mode": "legacy",
            "scenes_processed": coverage.get("scenes_processed"),
            "accepted_objects": pseudolabel.get("accepted_objects"),
            "accepted_geojson_mb": pseudolabel.get("accepted_geojson_mb"),
        },
        artifacts={"accepted_geojson": str(accepted), "vectorization_summary.json": str(summary_path)},
        summary="CPU vectorization compatibility stage completed from saved probability maps.",
    )


def _is_inference_engine_source(pseudolabel_cfg: dict) -> bool:
    source = str((pseudolabel_cfg or {}).get("source") or (pseudolabel_cfg or {}).get("engine") or "").strip().lower().replace("-", "_")
    return source == "inference_engine"
