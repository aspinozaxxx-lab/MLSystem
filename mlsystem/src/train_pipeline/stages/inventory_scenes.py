from __future__ import annotations

from ...dataset_preparing.api import DatasetInspectionRequest, inspect_dataset
from .context import StageContext
from .report import StageCheck, StageFailure, StageReport


def run(ctx: StageContext) -> StageReport:
    result = inspect_dataset(
        DatasetInspectionRequest(
            experiment_id=ctx.config.experiment_id,
            images_uri=ctx.config.images_uri,
            layout_uri=ctx.config.layout_uri,
            scenes_file=ctx.config.scenes_file,
            annotation_file=ctx.config.annotation_file,
            annotations=getattr(ctx.config, "annotations", None) or {},
            preprocess=ctx.config.preprocess or {},
            output_dir=ctx.store.run_dir,
        )
    )
    ctx.store.update_summary(scene_inventory=result.inventory, scene_matching=result.matching)
    report = StageReport(
        ctx.stage_id,
        result.status,
        [_stage_check(item) for item in result.checks],
        result.counters,
        result.warnings,
        result.errors,
        result.artifacts,
    )
    if result.errors:
        raise StageFailure("inventory_scenes failed", report)
    return report


def _stage_check(item: dict[str, str]) -> StageCheck:
    return StageCheck(item.get("name", "check"), item.get("status", "ok"), item.get("message", ""))
