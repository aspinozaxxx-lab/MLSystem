from __future__ import annotations

# Deprecated compatibility helper. Source of truth: InferenceEngine. Not used by the production pipeline runner.

from typing import Any

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
            summary="Pseudolabel scene preparation skipped because pseudolabel.enabled=false.",
        )
    run_on = str(pseudolabel_cfg.get("run_on") or "dataset_scenes")
    bad_scene_policy = str(pseudolabel_cfg.get("bad_scene_policy") or "skip")
    inventory = read_json(ctx.store.run_dir / "inventory_scenes.json", default={}) or {}
    manifest = read_json(ctx.store.run_dir / "dataset_manifest.json", default={}) or {}
    matched = inventory.get("matched") or (manifest.get("scene_matching") or {}).get("matched") or []
    available_by_entry = {str(item.get("entry") or item.get("name")): item for item in matched if item.get("entry") or item.get("name")}
    available_by_name = {str(item.get("name") or item.get("entry")): item for item in matched if item.get("entry") or item.get("name")}

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    if run_on in {"synthetic", "smoke_synthetic"} or ctx.config.smoke:
        selected = [{"entry": "synthetic", "name": "synthetic", "key": None, "score": 1.0, "source": "synthetic_smoke"}]
    elif run_on in {"dataset_scenes", "dataset", "matched_scenes"}:
        selected = list(matched)
    elif run_on in {"all_images", "all_available_images"}:
        selected = _images_as_matches(inventory.get("available_images") or [])
    elif run_on in {"validation_scenes", "val_scenes", "validation"}:
        selected, missing = _select_manifest_scenes(manifest.get("val_scenes") or [], available_by_entry, available_by_name)
    elif run_on in {"train_scenes", "training_scenes", "train"}:
        selected, missing = _select_manifest_scenes(manifest.get("train_scenes") or [], available_by_entry, available_by_name)
    elif run_on == "explicit_scene_list":
        explicit = pseudolabel_cfg.get("scene_list") or pseudolabel_cfg.get("explicit_scenes") or []
        if isinstance(explicit, str):
            explicit = [item.strip() for item in explicit.split(",") if item.strip()]
        for scene in explicit:
            match = available_by_entry.get(str(scene)) or available_by_name.get(str(scene))
            if match:
                selected.append(match)
            else:
                missing.append(str(scene))
    else:
        return _fail(ctx, [f"Unsupported pseudolabel.run_on: {run_on}"])

    errors: list[str] = []
    if missing and run_on in {"dataset_scenes", "dataset", "matched_scenes", "explicit_scene_list", "validation_scenes", "val_scenes", "validation", "train_scenes", "training_scenes", "train"}:
        errors.append(f"{len(missing)} inference scenes are missing: {missing[:10]}")
    if not selected:
        errors.append("No inference scenes selected")

    inference_manifest = {
        "schema_version": 1,
        "stage": ctx.stage_id,
        "experiment_id": ctx.config.experiment_id,
        "run_on": run_on,
        "bad_scene_policy": bad_scene_policy,
        "scene_count": len(selected),
        "missing_count": len(missing),
        "scenes": selected,
        "missing": missing,
    }
    manifest_path = ctx.store.run_dir / "inference_manifest.json"
    scenes_path = ctx.store.run_dir / "inference_scenes.txt"
    report_path = ctx.store.run_dir / "inference_inventory_report.json"
    write_json(manifest_path, inference_manifest)
    write_json(report_path, inference_manifest)
    scenes_path.write_text("\n".join(str(item.get("entry") or item.get("name")) for item in selected) + ("\n" if selected else ""), encoding="utf-8")
    ctx.store.update_summary(inference_manifest=inference_manifest)

    checks = [StageCheck("scene selection", "failed" if errors else "ok", f"run_on={run_on}, scenes={len(selected)}, missing={len(missing)}")]
    counters = {"run_on": run_on, "inference_scenes": len(selected), "missing": len(missing), "bad_scene_policy": bad_scene_policy}
    details = {
        "source_inventory": str(ctx.store.run_dir / "inventory_scenes.json"),
        "source_dataset_manifest": str(ctx.store.run_dir / "dataset_manifest.json") if manifest else None,
    }
    artifacts = {
        "inference_manifest.json": str(manifest_path),
        "inference_scenes.txt": str(scenes_path),
        "inference_inventory_report.json": str(report_path),
    }
    report = StageReport(ctx.stage_id, "failed" if errors else "success", checks, counters, errors=errors, artifacts=artifacts, details=details)
    if errors:
        raise StageFailure("prepare_inference_scenes failed", report)
    return report


def _select_manifest_scenes(items: list[dict[str, Any]], by_entry: dict[str, dict[str, Any]], by_name: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for item in items:
        scene = str(item.get("entry") or item.get("name") or "")
        match = by_entry.get(scene) or by_name.get(scene)
        if match:
            selected.append(match)
        else:
            missing.append(scene)
    return selected, missing


def _images_as_matches(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"entry": item.get("name"), "name": item.get("name"), "key": item.get("key"), "score": 1.0} for item in images]


def _fail(ctx: StageContext, errors: list[str]) -> StageReport:
    report = StageReport(ctx.stage_id, "failed", errors=errors)
    raise StageFailure(errors[0], report)
