from __future__ import annotations

from ...dataset_preparing.api import TrainingDatasetPrepareRequest, prepare_training_dataset
from ...storage.api import read_json
from .context import StageContext
from .report import StageCheck, StageFailure, StageReport


def run(ctx: StageContext) -> StageReport:
    inventory = read_json(ctx.store.run_dir / "inventory_scenes.json", default=None) or (ctx.store.read_summary().get("scene_inventory") or {})
    matching = read_json(ctx.store.run_dir / "scene_matching_report.json", default=None) or (ctx.store.read_summary().get("scene_matching") or {})
    result = prepare_training_dataset(
        TrainingDatasetPrepareRequest(
            run_id=ctx.run_id,
            experiment_id=ctx.config.experiment_id,
            output_dir=ctx.store.run_dir,
            class_name=getattr(ctx.config, "class_name", None),
            class_slug=_class_slug(getattr(ctx.config, "class_name", None)),
            inventory=inventory,
            matching=matching,
            preprocess=ctx.config.preprocess or {},
            raw_preprocess=dict((ctx.raw_conf or {}).get("preprocess") or {}),
            annotations=getattr(ctx.config, "annotations", None) or {},
            schema_version=getattr(ctx.config, "schema_version", None),
        )
    )
    ctx.store.update_summary(
        dataset_manifest=result.manifest,
        split_summary=result.split_summary,
        dataset_identity=result.dataset_identity.__dict__ if result.dataset_identity else None,
        dataset_mlflow_params=result.mlflow_params,
    )
    report = StageReport(
        ctx.stage_id,
        result.status,
        [_stage_check(item) for item in result.checks],
        result.counters,
        result.warnings,
        result.errors,
        result.artifacts,
        result.details,
    )
    if result.errors:
        raise StageFailure("prepare_dataset failed", report)
    return report


def _stage_check(item: dict[str, str]) -> StageCheck:
    return StageCheck(item.get("name", "check"), item.get("status", "ok"), item.get("message", ""))


def _class_slug(value: str | None) -> str | None:
    if not value:
        return None
    import re

    normalized = str(value).strip().casefold()
    known = {
        "абразия": "abrasion",
        "abrasion": "abrasion",
        "ветровая эрозия": "wind_erosion",
        "wind_erosion": "wind_erosion",
        "водная эрозия": "water_erosion",
        "water_erosion": "water_erosion",
        "вырубки": "deforest",
        "deforest": "deforest",
        "deforestation": "deforest",
        "cuttings": "deforest",
        "clearcuts": "deforest",
        "clear_cuts": "deforest",
        "гари": "burnt_forests",
        "burnt_forests": "burnt_forests",
        "границы леса": "forest_boundaries",
        "forest": "forest_boundaries",
        "forest_boundaries": "forest_boundaries",
        "засоления": "salty",
        "salty": "salty",
        "карьеры": "quarries",
        "careers": "quarries",
        "quarries": "quarries",
        "обвально-оползневые и осыпные": "obval_opolz_osyp",
        "landslide": "obval_opolz_osyp",
        "landslides": "obval_opolz_osyp",
        "озера": "lakes",
        "озёра": "lakes",
        "lakes": "lakes",
        "опустынивание": "desertification",
        "desertification": "desertification",
        "пашни": "arable_land",
        "areas_of_used_arable_land": "arable_land",
        "arable_land": "arable_land",
    }
    if normalized in known:
        return known[normalized]
    if normalized in {"deforest", "cuttings", "clearcuts", "clear_cuts", "вырубки", "РІС‹СЂСѓР±РєРё".casefold()}:
        return "deforest"
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or None
