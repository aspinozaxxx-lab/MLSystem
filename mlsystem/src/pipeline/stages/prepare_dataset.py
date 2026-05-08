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
    inventory_path = ctx.store.run_dir / "inventory_scenes.json"
    matching_path = ctx.store.run_dir / "scene_matching_report.json"
    matched_path = ctx.store.run_dir / "matched_scenes.txt"
    manifest_path = ctx.store.run_dir / "dataset_manifest.json"
    audit_json_path = ctx.store.run_dir / "prepare_dataset_input_audit.json"
    audit_txt_path = ctx.store.run_dir / "prepare_dataset_input_audit.txt"
    excluded_path = ctx.store.run_dir / "excluded_dataset_scenes.txt"

    inventory = read_json(inventory_path, default=None) or (ctx.store.read_summary().get("scene_inventory") or {})
    matching = read_json(matching_path, default=None) or (ctx.store.read_summary().get("scene_matching") or {})
    inventory_matched = list(inventory.get("matched") or matching.get("matched") or [])
    upstream_inventory_matched_count = _inventory_matched_count(inventory, matching, inventory_matched)
    limit_info = _resolve_dataset_input_limit(ctx)

    matched = _select_dataset_matches(inventory_matched, limit_info)
    excluded_matches = inventory_matched[len(matched) :]
    audit = _build_input_audit(
        ctx,
        inventory_report_path=inventory_path,
        matched_scenes_file=matched_path,
        inventory_matched_count=upstream_inventory_matched_count,
        selected=matched,
        excluded=excluded_matches,
        limit_info=limit_info,
        invariant_status=_input_invariant_status(upstream_inventory_matched_count, len(inventory_matched), len(matched), limit_info),
    )
    _write_input_audit_payload(audit, audit_json_path, audit_txt_path, excluded_path)
    input_warnings = _input_lineage_warnings(audit)
    input_errors = _input_invariant_errors(audit, manifest_path)
    if input_errors:
        report = StageReport(
            ctx.stage_id,
            "failed",
            [
                StageCheck(
                    "input lineage",
                    "failed",
                    f"inventory matched={audit['inventory_matched_count']}, selected={audit['selected_count']}, limit={audit['dataset_input_limit']}",
                )
            ],
            counters=_input_lineage_counters(audit),
            warnings=input_warnings,
            errors=input_errors,
            artifacts={
                "inventory_scenes.json": str(inventory_path),
                "matched_scenes.txt": str(matched_path),
                "prepare_dataset_input_audit.json": str(audit_json_path),
                "prepare_dataset_input_audit.txt": str(audit_txt_path),
                "excluded_dataset_scenes.txt": str(excluded_path),
            },
            details={"input_lineage": audit},
        )
        raise StageFailure("prepare_dataset input mismatch", report)

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
    warnings = input_warnings + _collect_count_warnings(rows)
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
        "annotation_source": inventory.get("annotation_source") or "layout_uri",
        "annotations": inventory.get("annotations") or getattr(ctx.config, "annotations", None) or {},
        "split_strategy": split_strategy,
        "object_count_mode": rows[0].matched_by if rows else count_mode,
        "scene_matching": matching,
        "input_lineage": audit,
        "selected_scene_count": len(matched),
        "train_scene_count": len(train_matches),
        "val_scene_count": len(val_matches),
        "train_scenes": train_matches,
        "val_scenes": val_matches,
        "scene_object_counts": [row.__dict__ | {"image_path": str(row.image_path) if row.image_path else None} for row in rows],
        "split_summary": split_summary,
        "limits": {
            "max_scenes": limit_info.get("legacy_max_scenes"),
            "dataset_input_limit": limit_info.get("value"),
            "dataset_input_limit_source": limit_info.get("source"),
            "dataset_input_limit_reason": limit_info.get("reason"),
            "max_train_tiles": preprocess.get("max_train_tiles"),
            "max_val_tiles": preprocess.get("max_val_tiles"),
        },
    }
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
        **_input_lineage_counters(audit),
        "total_objects": sum(row.object_count for row in rows),
        "scenes_without_objects": sum(1 for row in rows if row.object_count == 0),
        "train_scenes": len(train_rows),
        "train_objects": split_summary["train_objects"],
        "val_scenes": len(val_rows),
        "val_objects": split_summary["val_objects"],
        "split_strategy": split_strategy,
        "count_mode": rows[0].matched_by if rows else count_mode,
    }
    details = {
        "input_lineage": audit,
        "scene_object_counts": [{"scene_name": row.scene_name, "object_count": row.object_count} for row in rows],
    }
    artifacts = {
        "prepare_dataset_input_audit.json": str(audit_json_path),
        "prepare_dataset_input_audit.txt": str(audit_txt_path),
        "excluded_dataset_scenes.txt": str(excluded_path),
        "dataset_manifest.json": str(manifest_path),
        "train_scenes.txt": str(train_path),
        "val_scenes.txt": str(val_path),
        "scene_object_counts.txt": str(scene_counts_path),
        "train_val_split.txt": str(split_report_path),
        "split_summary.json": str(summary_path),
        "dataset_validation_report.json": str(validation_path),
    }
    checks = [
        StageCheck(
            "input lineage",
            "warning" if audit["limit_applied"] else "ok",
            f"source=inventory_scenes, inventory={audit['inventory_matched_count']}, selected={audit['selected_count']}, invariant={audit['invariant_status']}",
        ),
        StageCheck("inventory", "ok", f"{len(matched)} matched scenes loaded"),
        StageCheck("object counts", "ok", f"{sum(row.object_count for row in rows)} objects counted"),
        StageCheck("split", "failed" if errors else "ok", f"train={len(train_rows)}, val={len(val_rows)}"),
    ]
    report = StageReport(ctx.stage_id, "failed" if errors else "success", checks, counters, warnings, errors, artifacts, details)
    if errors:
        raise StageFailure("prepare_dataset failed", report)
    return report


def _inventory_matched_count(inventory: dict[str, Any], matching: dict[str, Any], matched: list[dict[str, Any]]) -> int:
    for key in ("matched_count", "selected_matched_count"):
        value = inventory.get(key)
        if value is not None:
            return int(value)
    for key in ("matched_count", "selected_matched_count"):
        value = matching.get(key)
        if value is not None:
            return int(value)
    return len(matched)


def _scene_name(item: dict[str, Any]) -> str:
    return str(item.get("entry") or item.get("name") or item.get("key") or "")


def _resolve_dataset_input_limit(ctx: StageContext) -> dict[str, Any]:
    preprocess = dict(getattr(ctx.config, "preprocess", None) or {})
    raw_preprocess = dict((ctx.raw_conf or {}).get("preprocess") or {})
    candidates = [
        ("dag_run.conf.preprocess.max_dataset_scenes", raw_preprocess.get("max_dataset_scenes"), preprocess.get("max_dataset_scenes")),
        ("dag_run.conf.preprocess.dataset_limit", raw_preprocess.get("dataset_limit"), preprocess.get("dataset_limit")),
        ("dag_run.conf.preprocess.scene_limit", raw_preprocess.get("scene_limit"), preprocess.get("scene_limit")),
        ("dag_run.conf.preprocess.sample_size", raw_preprocess.get("sample_size"), preprocess.get("sample_size")),
        ("dag_run.conf.preprocess.max_scenes", raw_preprocess.get("max_scenes"), preprocess.get("max_scenes")),
    ]
    for source, raw_value, config_value in candidates:
        value = raw_value if raw_value is not None else config_value
        if value in (None, ""):
            continue
        limit = max(1, int(value))
        reason = (
            raw_preprocess.get("dataset_limit_reason")
            or raw_preprocess.get("limit_reason")
            or preprocess.get("dataset_limit_reason")
            or preprocess.get("limit_reason")
            or ("preprocess.max_scenes compatibility limit" if source.endswith(".max_scenes") else "explicit dataset input limit configured")
        )
        return _dataset_limit_info(limit, source, str(reason), legacy_max_scenes=limit if source.endswith(".max_scenes") else preprocess.get("max_scenes"))
    return _dataset_limit_info(None, None, None, legacy_max_scenes=preprocess.get("max_scenes"))


def _dataset_limit_info(value: int | None, source: str | None, reason: str | None, *, legacy_max_scenes: Any = None) -> dict[str, Any]:
    return {
        "value": value,
        "source": source,
        "reason": reason,
        "configured": value is not None,
        "legacy_max_scenes": legacy_max_scenes,
    }


def _select_dataset_matches(matched: list[dict[str, Any]], limit_info: dict[str, Any]) -> list[dict[str, Any]]:
    limit = limit_info.get("value")
    if limit is None:
        return list(matched)
    return list(matched[: int(limit)])


def _build_input_audit(
    ctx: StageContext,
    *,
    inventory_report_path: Path,
    matched_scenes_file: Path,
    inventory_matched_count: int,
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    limit_info: dict[str, Any],
    invariant_status: str,
) -> dict[str, Any]:
    return {
        "run_id": ctx.run_id,
        "source": "inventory_scenes",
        "inventory_report_path": str(inventory_report_path),
        "inventory_matched_count": inventory_matched_count,
        "matched_scenes_file": str(matched_scenes_file),
        "selected_count": len(selected),
        "selected_scenes": [_scene_name(item) for item in selected],
        "excluded_count": len(excluded),
        "excluded_scenes": [_scene_name(item) for item in excluded],
        "limit_applied": bool(limit_info.get("value") is not None and len(selected) < inventory_matched_count),
        "dataset_input_limit": limit_info.get("value"),
        "limit_source": limit_info.get("source"),
        "limit_reason": limit_info.get("reason"),
        "invariant_status": invariant_status,
    }


def _input_invariant_status(inventory_matched_count: int, inventory_list_count: int, selected_count: int, limit_info: dict[str, Any]) -> str:
    if inventory_matched_count != inventory_list_count:
        return "FAILED: inventory matched_count does not match matched scene rows"
    if limit_info.get("value") is not None:
        return "OK with explicit limit"
    if selected_count == inventory_matched_count:
        return "OK"
    return "FAILED: selected scenes differ from inventory without explicit dataset limit"


def _input_invariant_errors(audit: dict[str, Any], manifest_path: Path) -> list[str]:
    errors: list[str] = []
    if int(audit.get("selected_count") or 0) <= 0:
        errors.append(
            "inventory_scenes output has no matched scenes. "
            f"inventory_scenes.json path={audit.get('inventory_report_path')}; "
            f"matched_scenes.txt path={audit.get('matched_scenes_file')}"
        )
    if audit.get("invariant_status") not in {"OK", "OK with explicit limit"}:
        errors.append(
            "prepare_dataset input mismatch: "
            f"inventory matched {audit.get('inventory_matched_count')} scenes, "
            f"selected {audit.get('selected_count')} scenes, no valid explicit dataset limit configured. "
            f"inventory_scenes.json path={audit.get('inventory_report_path')}; "
            f"matched_scenes.txt path={audit.get('matched_scenes_file')}; "
            f"dataset_manifest.json path={manifest_path if manifest_path.exists() else str(manifest_path) + ' (not created)'}; "
            f"excluded_scenes={audit.get('excluded_scenes')[:10] if isinstance(audit.get('excluded_scenes'), list) else []}"
        )
    return errors


def _input_lineage_warnings(audit: dict[str, Any]) -> list[str]:
    if not audit.get("limit_applied"):
        return []
    return [
        "Dataset input was explicitly limited: "
        f"selected={audit.get('selected_count')} of matched={audit.get('inventory_matched_count')} "
        f"source={audit.get('limit_source')} reason={audit.get('limit_reason')}"
    ]


def _input_lineage_counters(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "upstream_inventory_matched_scenes": audit.get("inventory_matched_count"),
        "selected_dataset_scenes": audit.get("selected_count"),
        "excluded_dataset_scenes": audit.get("excluded_count"),
        "dataset_input_limit": audit.get("dataset_input_limit"),
        "limit_applied": bool(audit.get("limit_applied")),
    }


def _write_input_audit_payload(audit: dict[str, Any], audit_json_path: Path, audit_txt_path: Path, excluded_path: Path) -> None:
    write_json(audit_json_path, audit)
    excluded = list(audit.get("excluded_scenes") or [])
    excluded_path.write_text("\n".join(excluded) + ("\n" if excluded else ""), encoding="utf-8")
    audit_txt_path.write_text(_input_audit_text(audit), encoding="utf-8")


def _input_audit_text(audit: dict[str, Any]) -> str:
    lines = [
        "prepare_dataset input audit",
        f"run_id={audit.get('run_id')}",
        f"source={audit.get('source')}",
        f"inventory_report_path={audit.get('inventory_report_path')}",
        f"inventory_matched_count={audit.get('inventory_matched_count')}",
        f"matched_scenes_file={audit.get('matched_scenes_file')}",
        f"selected_count={audit.get('selected_count')}",
        f"excluded_count={audit.get('excluded_count')}",
        f"limit_applied={audit.get('limit_applied')}",
        f"dataset_input_limit={audit.get('dataset_input_limit')}",
        f"limit_source={audit.get('limit_source')}",
        f"limit_reason={audit.get('limit_reason')}",
        f"invariant_status={audit.get('invariant_status')}",
        "",
        "[selected_scenes]",
        *(audit.get("selected_scenes") or []),
        "",
        "[excluded_scenes]",
        *(audit.get("excluded_scenes") or []),
    ]
    return "\n".join(str(line) for line in lines) + "\n"


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
