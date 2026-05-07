from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def build_annotation_report(
    run_id: str,
    status_root: Path,
    jobs: list[dict[str, Any]] | None = None,
    frontend_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frontend_status = frontend_status or {}
    run_dir = (status_root / run_id).resolve()
    status_root_resolved = status_root.resolve()
    if status_root_resolved not in [run_dir, *run_dir.parents]:
        raise ValueError("run_id resolves outside status root")

    inventory = read_json(run_dir / "inventory_scenes.json")
    manifest = read_json(run_dir / "dataset_manifest.json")
    split_summary = read_json(run_dir / "split_summary.json")
    validation = read_json(run_dir / "dataset_validation_report.json")
    inv_stage = read_json(run_dir / "stages" / "inventory_scenes.json")
    prep_stage = read_json(run_dir / "stages" / "prepare_dataset.json")

    train_names = {_scene_name(item) for item in manifest.get("train_scenes") or []}
    val_names = {_scene_name(item) for item in manifest.get("val_scenes") or []}
    object_counts = {
        str(item.get("scene_name")): item
        for item in (manifest.get("scene_object_counts") or [])
        if item.get("scene_name")
    }
    rows = []
    for item in inventory.get("matched") or []:
        scene = _scene_name(item)
        rows.append(
            {
                "scene": scene,
                "objects": int((object_counts.get(scene) or {}).get("object_count") or 0),
                "storage_status": "found",
                "split": "train" if scene in train_names else ("val" if scene in val_names else "none"),
            }
        )
    for scene in inventory.get("missing") or []:
        rows.append({"scene": str(scene), "objects": None, "storage_status": "missing", "split": "none"})

    frontend_stage_statuses = {
        str(item.get("name")): item
        for item in (frontend_status.get("stage_statuses") or [])
        if item.get("name")
    }
    stages = [
        _stage_view("inventory_scenes", inv_stage, frontend_stage_statuses.get("inventory_scenes")),
        _stage_view("prepare_dataset", prep_stage, frontend_stage_statuses.get("prepare_dataset")),
    ]
    status = "running"
    if frontend_status.get("status") == "failed":
        status = "failed"
    elif any(stage.get("status") == "failed" for stage in stages):
        status = "failed"
    elif prep_stage.get("status") in {"success", "success_with_warning"}:
        status = "succeeded"
    elif frontend_status.get("status") in {"queued", "running"}:
        status = str(frontend_status["status"])

    has_inventory_artifact = bool(inventory)
    has_manifest_artifact = bool(manifest)
    uploaded = frontend_status.get("uploaded_files") or {}
    summary = {
        "total_scenes_requested": (
            inventory.get("expanded_scene_count") or inventory.get("scene_count")
            if has_inventory_artifact
            else frontend_status.get("scene_count") or uploaded.get("scene_count")
        ),
        "requested_entries_count": inventory.get("requested_entries_count") if has_inventory_artifact else frontend_status.get("scene_count") or uploaded.get("scene_count"),
        "matched_scenes": inventory.get("matched_count") if has_inventory_artifact else None,
        "missing_scenes": inventory.get("missing_count") if has_inventory_artifact else None,
        "total_objects": (prep_stage.get("counters") or {}).get("total_objects") or split_summary.get("total_objects"),
        "train_scenes": (prep_stage.get("counters") or {}).get("train_scenes") or (manifest.get("train_scene_count") if has_manifest_artifact else None),
        "val_scenes": (prep_stage.get("counters") or {}).get("val_scenes") or (manifest.get("val_scene_count") if has_manifest_artifact else None),
        "train_objects": (prep_stage.get("counters") or {}).get("train_objects") or split_summary.get("train_objects"),
        "val_objects": (prep_stage.get("counters") or {}).get("val_objects") or split_summary.get("val_objects"),
        "split_strategy": (prep_stage.get("counters") or {}).get("split_strategy") or manifest.get("split_strategy"),
    }
    artifacts = [
        name
        for name in [
            "inventory_scenes.json",
            "scene_matching_report.json",
            "missing_scenes.txt",
            "matched_scenes.txt",
            "dataset_manifest.json",
            "scene_object_counts.txt",
            "split_summary.json",
            "dataset_validation_report.json",
        ]
        if (run_dir / name).exists()
    ]
    return {
        "run_id": run_id,
        "status": status,
        "jobs": jobs or [],
        "stages": stages,
        "summary": summary,
        "scene_rows": rows,
        "artifacts": artifacts,
        "validation": validation,
        "error": frontend_status.get("error"),
        "failed_step": frontend_status.get("failed_step"),
        "uploaded_files": uploaded,
    }


def _scene_name(item: dict[str, Any]) -> str:
    return str(item.get("entry") or item.get("name") or item.get("scene_name") or "")


def _stage_view(name: str, data: dict[str, Any], frontend_stage: dict[str, Any] | None = None) -> dict[str, Any]:
    frontend_stage = frontend_stage or {}
    return {
        "name": name,
        "status": data.get("status") or frontend_stage.get("status") or "pending",
        "summary": data.get("summary") or frontend_stage.get("summary") or "",
        "checks": data.get("checks") or [],
        "counters": data.get("counters") or {},
        "warnings": data.get("warnings") or [],
        "errors": data.get("errors") or frontend_stage.get("errors") or [],
        "artifacts": data.get("artifacts") or {},
    }
