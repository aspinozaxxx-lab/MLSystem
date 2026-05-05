from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...data.dataset_split import (
    SceneObjectCount,
    TrainValSplit,
    count_objects_per_scene,
    split_manifest_scene_rows,
    split_train_val_by_object_counts,
    write_scene_object_counts_report,
    write_split_outputs,
    write_train_val_split_report,
)
from ...pipeline_config import load_config
from ...storage.local_io import read_json, write_json
from ...storage.s3 import raster_path_for_s3_key, read_s3_text
from .context import StageContext
from .report import StageCheck, StageFailure, StageReport


def run(ctx: StageContext) -> StageReport:
    inventory = read_json(ctx.store.run_dir / "inventory_scenes.json", default=None) or (ctx.store.read_summary().get("scene_inventory") or {})
    matching = read_json(ctx.store.run_dir / "scene_matching_report.json", default=None) or (ctx.store.read_summary().get("scene_matching") or {})
    matched = inventory.get("matched") or matching.get("matched") or []
    if not matched:
        report = StageReport(ctx.stage_id, "failed", errors=["inventory_scenes output has no matched scenes"])
        raise StageFailure("prepare_dataset cannot run without matched scenes", report)

    max_scenes = ctx.config.preprocess.get("max_scenes")
    if max_scenes is not None:
        matched = matched[: max(1, int(max_scenes))]

    pipeline_config = load_config()
    annotation_uri = inventory.get("annotation_uri") or matching.get("annotation_uri")
    if not annotation_uri:
        report = StageReport(ctx.stage_id, "failed", errors=["annotation_uri is missing in inventory"])
        raise StageFailure("annotation_uri is missing", report)

    annotation_path = ctx.store.run_dir / "dataset_annotation.geojson"
    try:
        annotation_path.write_text(read_s3_text(pipeline_config, annotation_uri), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        report = StageReport(ctx.stage_id, "failed", errors=[f"annotation_file cannot be read: {exc}"])
        raise StageFailure("annotation_file cannot be read", report) from exc

    scene_names = [str(item.get("entry") or item.get("name") or "") for item in matched if item.get("entry") or item.get("name")]
    scene_to_match = {str(item.get("entry") or item.get("name")): item for item in matched if item.get("entry") or item.get("name")}
    scene_to_image = {
        scene: Path(raster_path_for_s3_key(pipeline_config, str(scene_to_match[scene].get("key"))))
        for scene in scene_names
    }

    preprocess = ctx.config.preprocess or {}
    count_mode = str(preprocess.get("count_mode") or "auto")
    annotation_crs = preprocess.get("annotation_crs")
    allow_inferred_crs = bool(preprocess.get("allow_inferred_annotation_crs", True))
    try:
        rows = count_objects_per_scene(
            scene_names,
            scene_to_image,
            annotation_path,
            count_mode=count_mode,
            annotation_crs=annotation_crs,
            allow_inferred_crs=allow_inferred_crs,
        )
    except Exception as exc:  # noqa: BLE001
        report = StageReport(ctx.stage_id, "failed", errors=[f"Object counting failed: {exc}"])
        raise StageFailure("object counting failed", report) from exc

    split_strategy = _resolve_split_strategy(ctx.config, preprocess)
    target_val_fraction = float(preprocess.get("target_val_fraction", 0.2))
    split_seed = int(preprocess.get("split_seed", 42))
    if split_strategy == "legacy_75_25":
        train_matches, val_matches = split_manifest_scene_rows(matched, train_fraction=0.75)
        by_name = {row.scene_name: row for row in rows}
        train_rows = [by_name[str(item.get("entry") or item.get("name"))] for item in train_matches]
        val_rows = [by_name[str(item.get("entry") or item.get("name"))] for item in val_matches]
        split_summary = _split_summary(train_rows, val_rows, split_seed, target_val_fraction)
    elif split_strategy == "object_balanced":
        split = split_train_val_by_object_counts(rows, target_val_fraction=target_val_fraction, seed=split_seed)
        train_rows = split.train
        val_rows = split.val
        train_names = {row.scene_name for row in train_rows}
        val_names = {row.scene_name for row in val_rows}
        train_matches = [scene_to_match[row.scene_name] for row in train_rows]
        val_matches = [scene_to_match[row.scene_name] for row in val_rows]
        split_summary = split.summary
        if train_names & val_names:
            return _fail(ctx, ["train/val split has overlapping scenes"])
    else:
        return _fail(ctx, [f"Unsupported preprocess.split_strategy: {split_strategy}"])

    errors: list[str] = []
    warnings = _collect_count_warnings(rows)
    if not train_rows:
        errors.append("train split is empty")
    if not val_rows:
        errors.append("validation split is empty")
    split_names = {row.scene_name for row in train_rows + val_rows}
    lost = [scene for scene in scene_names if scene not in split_names]
    if lost:
        errors.append(f"{len(lost)} scenes were lost during split: {lost[:10]}")
    if any(row.object_count == 0 for row in rows):
        warnings.append(f"{sum(1 for row in rows if row.object_count == 0)} scenes have zero objects")

    manifest = {
        "experiment_id": ctx.config.experiment_id,
        "created_by": "prepare_dataset",
        "source": "airflow",
        "split_strategy": split_strategy,
        "object_count_mode": rows[0].matched_by if rows else count_mode,
        "scene_matching": matching,
        "selected_scene_count": len(matched),
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "train_scenes": train_matches,
        "val_scenes": val_matches,
        "scene_object_counts": [row.__dict__ | {"image_path": str(row.image_path) if row.image_path else None} for row in rows],
        "split_summary": split_summary,
        "limits": {
            "max_scenes": max_scenes,
            "max_train_tiles": preprocess.get("max_train_tiles"),
            "max_val_tiles": preprocess.get("max_val_tiles"),
        },
    }
    manifest_path = ctx.store.run_dir / "dataset_manifest.json"
    scene_counts_path = ctx.store.run_dir / "scene_object_counts.txt"
    split_report_path = ctx.store.run_dir / "train_val_split.txt"
    train_path = ctx.store.run_dir / "train_scenes.txt"
    val_path = ctx.store.run_dir / "val_scenes.txt"
    summary_path = ctx.store.run_dir / "split_summary.json"
    validation_path = ctx.store.run_dir / "dataset_validation_report.json"
    write_json(manifest_path, manifest)
    write_scene_object_counts_report(rows, scene_counts_path)
    train_path.write_text("\n".join(row.scene_name for row in train_rows) + ("\n" if train_rows else ""), encoding="utf-8")
    val_path.write_text("\n".join(row.scene_name for row in val_rows) + ("\n" if val_rows else ""), encoding="utf-8")
    output_split = TrainValSplit(train=train_rows, val=val_rows, summary=split_summary)
    write_train_val_split_report(output_split, split_report_path)
    write_split_outputs(rows, output_split, ctx.store.run_dir)
    write_json(summary_path, split_summary)
    write_json(validation_path, {"status": "failed" if errors else "success", "errors": errors, "warnings": warnings, "split_summary": split_summary})
    ctx.store.update_summary(dataset_manifest=manifest, split_summary=split_summary)

    counters = {
        "total_scenes": len(rows),
        "total_objects": sum(row.object_count for row in rows),
        "scenes_without_objects": sum(1 for row in rows if row.object_count == 0),
        "train_scenes": len(train_rows),
        "train_objects": split_summary["train_objects"],
        "val_scenes": len(val_rows),
        "val_objects": split_summary["val_objects"],
        "split_strategy": split_strategy,
        "count_mode": rows[0].matched_by if rows else count_mode,
    }
    details = {"scene_object_counts": [{"scene_name": row.scene_name, "object_count": row.object_count} for row in rows]}
    artifacts = {
        "dataset_manifest.json": str(manifest_path),
        "train_scenes.txt": str(train_path),
        "val_scenes.txt": str(val_path),
        "scene_object_counts.txt": str(scene_counts_path),
        "train_val_split.txt": str(split_report_path),
        "split_summary.json": str(summary_path),
        "dataset_validation_report.json": str(validation_path),
    }
    checks = [
        StageCheck("inventory", "ok", f"{len(matched)} matched scenes loaded"),
        StageCheck("object counts", "ok", f"{sum(row.object_count for row in rows)} objects counted"),
        StageCheck("split", "failed" if errors else "ok", f"train={len(train_rows)}, val={len(val_rows)}"),
    ]
    report = StageReport(ctx.stage_id, "failed" if errors else "success", checks, counters, warnings, errors, artifacts, details)
    if errors:
        raise StageFailure("prepare_dataset failed", report)
    return report


def _collect_count_warnings(rows: list[SceneObjectCount]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        warnings.extend(row.warnings)
    return sorted(set(warnings))


def _split_summary(train_rows: list[SceneObjectCount], val_rows: list[SceneObjectCount], seed: int, target_val_fraction: float) -> dict[str, Any]:
    train_objects = sum(row.object_count for row in train_rows)
    val_objects = sum(row.object_count for row in val_rows)
    total_files = len(train_rows) + len(val_rows)
    total_objects = train_objects + val_objects
    rows = train_rows + val_rows
    warnings = _collect_count_warnings(rows)
    return {
        "train_files": len(train_rows),
        "train_objects": train_objects,
        "val_files": len(val_rows),
        "val_objects": val_objects,
        "total_files": total_files,
        "total_objects": total_objects,
        "scenes_with_objects": sum(1 for row in rows if row.object_count > 0),
        "scenes_without_objects": sum(1 for row in rows if row.object_count <= 0),
        "warnings_count": len(warnings),
        "count_modes": sorted(set(row.matched_by for row in rows)),
        "crs_source": _crs_source_from_warnings(warnings),
        "val_fraction_by_files": (len(val_rows) / total_files) if total_files else 0.0,
        "val_fraction_by_objects": (val_objects / total_objects) if total_objects else 0.0,
        "seed": seed,
        "target_val_fraction": target_val_fraction,
    }


def _crs_source_from_warnings(warnings: list[str]) -> str:
    import re

    for warning in warnings:
        match = re.search(r"inferred annotation CRS ([A-Za-z0-9:]+)", warning)
        if match:
            return f"inferred:{match.group(1)}"
    for warning in warnings:
        match = re.search(r"annotation CRS ([A-Za-z0-9:]+)", warning)
        if match:
            return f"explicit_or_geojson:{match.group(1)}"
    return "not_used"


def _resolve_split_strategy(config: Any, preprocess: dict[str, Any]) -> str:
    configured = preprocess.get("split_strategy")
    if configured:
        return str(configured)
    schema_version = getattr(config, "schema_version", None)
    if schema_version is not None and int(schema_version) >= 2:
        return "object_balanced"
    return "legacy_75_25"


def _fail(ctx: StageContext, errors: list[str]) -> StageReport:
    report = StageReport(ctx.stage_id, "failed", errors=errors)
    raise StageFailure(errors[0], report)
