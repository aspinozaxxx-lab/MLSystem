from __future__ import annotations

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
    manifest_path = ctx.store.run_dir / "pseudolabel_scene_results_manifest.json"
    probability_index_path = ctx.store.run_dir / "probability_maps_index.json"
    inference_results_path = ctx.store.run_dir / "inference_results.json"
    probability_index = read_json(manifest_path, default={}) or {}
    write_json(probability_index_path, probability_index)
    write_json(
        inference_results_path,
        {
            "coverage": coverage,
            "pseudolabel": pseudolabel,
            "scene_results_manifest": str(manifest_path),
            "probability_maps_index": str(probability_index_path),
        },
    )
    return StageReport(
        ctx.stage_id,
        "success",
        [StageCheck("GPU inference", "ok", "Triton direct pseudolabel inference completed")],
        counters={
            "scenes_processed": coverage.get("scenes_processed"),
            "total_predicted_windows": coverage.get("total_predicted_windows"),
            "mean_coverage_fraction": coverage.get("mean_coverage_fraction"),
            "accepted_objects": pseudolabel.get("accepted_objects"),
        },
        artifacts={
            "inference_results.json": str(inference_results_path),
            "probability_maps_index.json": str(probability_index_path),
            "pseudolabel_scene_results_manifest.json": str(manifest_path),
        },
        summary="GPU pseudolabel inference completed; CPU vectorization/postprocess are separate compatibility stages.",
    )
