from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetInspectionRequest:
    experiment_id: str
    images_uri: str
    layout_uri: str
    scenes_file: str = "scenes.txt"
    annotation_file: str = "auto"
    annotations: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)
    output_dir: Path | None = None
    runtime_config: Any | None = None


@dataclass(frozen=True)
class DatasetInspectionResult:
    status: str
    inventory: dict[str, Any] = field(default_factory=dict)
    matching: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingDatasetPrepareRequest:
    run_id: str
    experiment_id: str
    output_dir: Path
    class_name: str | None = None
    class_slug: str | None = None
    inventory: dict[str, Any] = field(default_factory=dict)
    matching: dict[str, Any] = field(default_factory=dict)
    preprocess: dict[str, Any] = field(default_factory=dict)
    raw_preprocess: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)
    schema_version: int | None = None
    runtime_config: Any | None = None


@dataclass(frozen=True)
class TrainingDatasetPrepareResult:
    status: str
    manifest: dict[str, Any] = field(default_factory=dict)
    split_summary: dict[str, Any] = field(default_factory=dict)
    dataset_identity: "DatasetIdentity | None" = None
    counters: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    mlflow_params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetIdentity:
    fingerprint: str
    version: str
    version_source: str
    objects_count: int
    scenes_count: int
    selected_scenes: list[str] = field(default_factory=list)
    train_scenes: list[str] = field(default_factory=list)
    val_scenes: list[str] = field(default_factory=list)
    git_commit: str | None = None
    git_commit_date: str | None = None
    class_name: str | None = None
    class_slug: str | None = None
    split_strategy: str | None = None
    annotation_uri: str | None = None
    scenes_uri: str | None = None
    images_uri: str | None = None
    layout_uri: str | None = None


class DatasetPreparingError(Exception):
    pass
