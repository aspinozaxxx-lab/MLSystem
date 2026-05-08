from __future__ import annotations

from ...storage.local_io import read_json, write_json
from .context import StageContext
from .report import StageCheck, StageFailure, StageReport


def run(ctx: StageContext) -> StageReport:
    from ..airflow_tasks import _read_training_result

    pseudolabel_cfg = ctx.config.pseudolabel or {}
    if "enabled" in pseudolabel_cfg and not bool(pseudolabel_cfg.get("enabled")) and not ctx.config.smoke:
        return StageReport(
            ctx.stage_id,
            "skipped",
            [StageCheck("pseudolabel.enabled", "skipped", "pseudolabel.enabled=false")],
            summary="Postprocess skipped because pseudolabel.enabled=false.",
        )

    if ctx.config.smoke:
        summary = read_json(ctx.store.run_dir / "pseudolabel_summary.json", default={}) or {}
        metrics = summary.get("metrics") or {
            "accepted_objects": 1,
            "threshold_used": 0.5,
            "min_object_area_m2_used": 1,
            "simplify_tolerance_m_used": 0,
        }
        summary_path = ctx.store.run_dir / "postprocess_summary.json"
        write_json(summary_path, metrics)
        return StageReport(
            ctx.stage_id,
            "success",
            [StageCheck("synthetic postprocess", "ok", "Synthetic smoke postprocess metrics are available")],
            counters={
                "accepted_objects": metrics.get("accepted_objects"),
                "threshold_used": metrics.get("threshold_used"),
            },
            artifacts={"postprocess_summary.json": str(summary_path)},
        )

    training_result = _read_training_result(ctx.store)
    metrics = training_result.get("postprocess_metrics") or {}
    if not metrics:
        report = StageReport(ctx.stage_id, "failed", [StageCheck("postprocess metrics", "failed", "No postprocess metrics found")], errors=["No postprocess metrics found"])
        raise StageFailure("No postprocess metrics found", report)
    summary_path = ctx.store.run_dir / "postprocess_summary.json"
    write_json(summary_path, metrics)
    return StageReport(
        ctx.stage_id,
        "success",
        [StageCheck("postprocess metrics", "ok", "Postprocess metrics found")],
        counters={
            "accepted_objects": metrics.get("accepted_objects"),
            "threshold_used": metrics.get("threshold_used"),
            "min_object_area_m2_used": metrics.get("min_object_area_m2_used"),
            "simplify_tolerance_m_used": metrics.get("simplify_tolerance_m_used"),
        },
        artifacts={"postprocess_summary.json": str(summary_path)},
    )
