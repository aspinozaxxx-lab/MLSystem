from __future__ import annotations

import hashlib
import json
from typing import Any

from ._dataset_split import (
    SceneObjectCount,
    TrainValSplit,
    clean_scene_list_file,
    count_objects_per_scene,
    filter_existing_scenes,
    index_image_files,
    read_scene_list,
    split_manifest_scene_rows,
    split_train_val_by_object_counts,
    write_scene_object_counts_report,
    write_split_outputs,
    write_train_val_split_report,
)
from ._scene_matching import build_scene_matching_report, match_scenes, norm_scene_name, parse_scene_list_text
from .contracts import DatasetIdentity, DatasetPreparingError


def compute_dataset_identity(
    *,
    dataset_manifest: dict[str, Any] | None = None,
    inventory: dict[str, Any] | None = None,
    scene_matching: dict[str, Any] | None = None,
    class_name: str | None = None,
    class_slug: str | None = None,
    images_uri: str | None = None,
    layout_uri: str | None = None,
    scenes_uri: str | None = None,
    annotation_uri: str | None = None,
) -> DatasetIdentity:
    manifest = dataset_manifest or {}
    inv = inventory or {}
    matching = scene_matching or manifest.get("scene_matching") or {}
    annotations = _first_dict(manifest.get("annotations"), inv.get("annotations"), matching.get("annotations"))

    train_scenes = _scene_names(manifest.get("train_scenes"))
    val_scenes = _scene_names(manifest.get("val_scenes"))
    selected_scenes = _first_list(
        _scene_names((manifest.get("input_lineage") or {}).get("selected_scenes")),
        _scene_names(inv.get("matched")),
        _scene_names(matching.get("matched")),
        train_scenes + val_scenes,
    )
    scenes_count = _first_int(
        manifest.get("selected_scene_count"),
        inv.get("scene_count"),
        inv.get("expanded_scene_count"),
        inv.get("matched_count"),
        matching.get("expanded_scene_count"),
        len(selected_scenes),
    )
    objects_count = _objects_count(manifest)
    split_summary = _first_dict(manifest.get("split_summary"), {})
    git_commit = _first_text(
        annotations.get("commit"),
        annotations.get("git_commit"),
        annotations.get("mlmarkup_commit"),
        manifest.get("git_commit"),
        inv.get("git_commit"),
    )
    git_commit_date = _first_text(
        annotations.get("commit_date"),
        annotations.get("git_commit_date"),
        annotations.get("mlmarkup_commit_date"),
        manifest.get("git_commit_date"),
        inv.get("git_commit_date"),
    )
    payload = {
        "annotation_uri": annotation_uri or inv.get("annotation_uri") or matching.get("annotation_uri"),
        "annotations": annotations,
        "class_name": class_name,
        "class_slug": class_slug,
        "images_uri": images_uri or inv.get("images_uri") or matching.get("images_uri"),
        "layout_uri": layout_uri or inv.get("layout_uri") or matching.get("layout_uri"),
        "objects_count": objects_count,
        "scenes_count": scenes_count,
        "scenes_uri": scenes_uri or inv.get("scenes_uri") or matching.get("scenes_uri"),
        "selected_scenes": selected_scenes,
        "split_strategy": manifest.get("split_strategy"),
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
    }
    fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    version = git_commit or fingerprint
    return DatasetIdentity(
        fingerprint=fingerprint,
        version=version,
        version_source="mlmarkup_git_commit" if git_commit else "fingerprint",
        objects_count=objects_count,
        scenes_count=scenes_count,
        selected_scenes=selected_scenes,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        git_commit=git_commit,
        git_commit_date=git_commit_date,
        class_name=class_name,
        class_slug=class_slug,
        split_strategy=_first_text(manifest.get("split_strategy"), split_summary.get("split_strategy")),
        annotation_uri=payload["annotation_uri"],
        scenes_uri=payload["scenes_uri"],
        images_uri=payload["images_uri"],
        layout_uri=payload["layout_uri"],
    )


def dataset_identity_mlflow_payload(identity: DatasetIdentity) -> dict[str, object]:
    payload = {
        "dataset.version": identity.version,
        "dataset.version_source": identity.version_source,
        "dataset.fingerprint": identity.fingerprint,
        "dataset.git_commit": identity.git_commit,
        "dataset.git_commit_date": identity.git_commit_date,
        "dataset.class_name": identity.class_name,
        "dataset.class_slug": identity.class_slug,
        "dataset.objects": identity.objects_count,
        "dataset.scenes": identity.scenes_count,
        "dataset.selected_scenes": len(identity.selected_scenes),
        "dataset.train_scenes": len(identity.train_scenes),
        "dataset.val_scenes": len(identity.val_scenes),
        "dataset.split_strategy": identity.split_strategy,
        "dataset.annotation_uri": identity.annotation_uri,
        "dataset.scenes_uri": identity.scenes_uri,
        "dataset.images_uri": identity.images_uri,
        "dataset.layout_uri": identity.layout_uri,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _objects_count(manifest: dict[str, Any]) -> int:
    split_summary = manifest.get("split_summary") if isinstance(manifest.get("split_summary"), dict) else {}
    total = _int_or_none(split_summary.get("total_objects"))
    if total is not None:
        return total
    total = 0
    for row in manifest.get("scene_object_counts") or []:
        if isinstance(row, dict):
            total += int(row.get("object_count") or 0)
    return total


def _scene_names(rows: Any) -> list[str]:
    if not isinstance(rows, list):
        return []
    result: list[str] = []
    for row in rows:
        if isinstance(row, str):
            name = row
        elif isinstance(row, dict):
            name = str(row.get("scene_name") or row.get("entry") or row.get("name") or row.get("key") or "")
        else:
            name = str(getattr(row, "scene_name", "") or "")
        if name:
            result.append(name)
    return result


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return dict(value)
    return {}


def _first_list(*values: list[str]) -> list[str]:
    for value in values:
        if value:
            return list(value)
    return []


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _int_or_none(value)
        if parsed is not None:
            return parsed
    return 0


__all__ = [
    "DatasetIdentity",
    "DatasetPreparingError",
    "SceneObjectCount",
    "TrainValSplit",
    "build_scene_matching_report",
    "clean_scene_list_file",
    "compute_dataset_identity",
    "count_objects_per_scene",
    "dataset_identity_mlflow_payload",
    "filter_existing_scenes",
    "index_image_files",
    "match_scenes",
    "norm_scene_name",
    "parse_scene_list_text",
    "read_scene_list",
    "split_manifest_scene_rows",
    "split_train_val_by_object_counts",
    "write_scene_object_counts_report",
    "write_split_outputs",
    "write_train_val_split_report",
]
