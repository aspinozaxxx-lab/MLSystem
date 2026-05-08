from __future__ import annotations

from .context import StageContext
from .report import StageCheck, StageFailure, StageReport


def run(ctx: StageContext) -> StageReport:
    from ..airflow_tasks import _existing_artifacts, _forbidden_logged_artifacts

    pseudolabel_cfg = ctx.config.pseudolabel or {}
    if "enabled" in pseudolabel_cfg and not bool(pseudolabel_cfg.get("enabled")) and not ctx.config.smoke:
        return StageReport(
            ctx.stage_id,
            "skipped",
            [StageCheck("pseudolabel.enabled", "skipped", "pseudolabel.enabled=false")],
            summary="Pseudolabel export skipped because pseudolabel.enabled=false.",
        )

    artifacts = _existing_artifacts(ctx.store)
    required = ["pseudolabel_scenes.txt", f"{ctx.config.experiment_id}.accepted.geojson", "coverage_report.json", "pseudolabel_summary.json"]
    missing_required = [name for name in required if name not in artifacts]
    if missing_required:
        report = StageReport(
            ctx.stage_id,
            "failed",
            [StageCheck("required artifacts", "failed", f"missing={missing_required}")],
            errors=[f"Missing pseudolabel artifacts: {missing_required}"],
            artifacts=artifacts,
        )
        raise StageFailure("Missing pseudolabel artifacts", report)
    return StageReport(
        ctx.stage_id,
        "success",
        [StageCheck("required artifacts", "ok", "Required pseudolabel artifacts are present")],
        counters={"artifact_count": len(required)},
        artifacts={name: artifacts[name] for name in required},
        details={"excluded_local_artifacts_not_logged": _forbidden_logged_artifacts(ctx.store)},
    )
