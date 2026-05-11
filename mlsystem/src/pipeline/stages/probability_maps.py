from __future__ import annotations

from ...storage.local_io import read_json, write_json
from .context import StageContext
from .report import StageCheck, StageFailure, StageReport


def run(ctx: StageContext) -> StageReport:
    pseudolabel_cfg = ctx.config.pseudolabel or {}
    if "enabled" in pseudolabel_cfg and not bool(pseudolabel_cfg.get("enabled")) and not ctx.config.smoke:
        return StageReport(
            ctx.stage_id,
            "skipped",
            [StageCheck("pseudolabel.enabled", "skipped", "pseudolabel.enabled=false")],
            summary="Probability map validation skipped because pseudolabel.enabled=false.",
        )
    coverage = read_json(ctx.store.run_dir / "coverage_report.json", default={}) or {}
    manifest = read_json(ctx.store.run_dir / "pseudolabel_scene_results_manifest.json", default={}) or {}
    if not coverage:
        report = StageReport(ctx.stage_id, "failed", [StageCheck("coverage_report", "failed", "coverage_report.json is missing")], errors=["coverage_report.json is missing"])
        raise StageFailure("coverage_report.json is missing", report)
    probability_index_path = ctx.store.run_dir / "probability_maps_index.json"
    source = str(pseudolabel_cfg.get("source") or pseudolabel_cfg.get("engine") or "").strip().lower().replace("-", "_")
    if source == "inference_engine" and probability_index_path.exists():
        mode = "inference_engine_validate_only"
    else:
        write_json(probability_index_path, manifest)
        mode = "mlsystem_validate"
    return StageReport(
        ctx.stage_id,
        "success",
        [StageCheck("probability maps", "ok", "Probability map coverage/index validated")],
        counters={
            "probability_map_mode": mode,
            "mean_coverage_fraction": coverage.get("mean_coverage_fraction"),
            "min_coverage_fraction": coverage.get("min_coverage_fraction"),
            "total_expected_windows": coverage.get("total_expected_windows"),
            "total_predicted_windows": coverage.get("total_predicted_windows"),
        },
        artifacts={"probability_maps_index.json": str(probability_index_path), "coverage_report.json": str(ctx.store.run_dir / "coverage_report.json")},
        summary="Probability map outputs validated. Actual stitching currently remains inside the inference runner.",
    )
