from __future__ import annotations

from ...storage.local_io import read_json, write_json
from .context import StageContext
from .report import StageCheck, StageReport


def run(ctx: StageContext) -> StageReport:
    from ..airflow_tasks import _run_pseudolabel_pipeline

    accepted = ctx.store.run_dir / f"{ctx.config.experiment_id}.accepted.geojson"
    if not accepted.exists():
        if not ctx.config.pseudolabel.get("enabled", False):
            return StageReport(ctx.stage_id, "skipped", [StageCheck("pseudolabel.enabled", "skipped", "pseudolabel.enabled=false")])
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
            "scenes_processed": coverage.get("scenes_processed"),
            "accepted_objects": pseudolabel.get("accepted_objects"),
            "accepted_geojson_mb": pseudolabel.get("accepted_geojson_mb"),
        },
        artifacts={"accepted_geojson": str(accepted), "vectorization_summary.json": str(summary_path)},
        summary="CPU vectorization compatibility stage completed from saved probability maps.",
    )
