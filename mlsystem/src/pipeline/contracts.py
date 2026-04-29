from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SceneRef:
    scene_id: str
    name: str
    uri: str | None = None
    bucket: str | None = None
    key: str | None = None
    width: int | None = None
    height: int | None = None
    crs: str | None = None
    transform: Any | None = None
    bounds: tuple[float, float, float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneMatchReport:
    matched: list[SceneRef]
    missing: list[str]
    ambiguous: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    annotation_uri: str | None = None
    scenes_uri: str | None = None
    images_uri: str | None = None
    layout_uri: str | None = None
    thresholds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TileWindow:
    scene_id: str
    x: int
    y: int
    width: int
    height: int
    tile_size: int
    stride: int
    is_edge: bool = False
    bounds: tuple[float, float, float, float] | None = None
    index: int | None = None


@dataclass
class TileBatch:
    windows: list[TileWindow]
    images: Any
    masks: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionTile:
    scene_id: str
    window: TileWindow
    prob: np.ndarray
    logits_shape: list[int] | None = None
    insert_slices: dict[str, int] = field(default_factory=dict)
    weight: np.ndarray | None = None
    skipped_reason: str | None = None


@dataclass
class ProbabilityMap:
    scene_id: str
    prob: np.ndarray
    weight_sum: np.ndarray
    coverage_mask: np.ndarray
    coverage_fraction: float
    transform: Any | None = None
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorizationResult:
    features_raw: list[dict[str, Any]]
    raw_count: int
    vertices_before: int
    threshold: float
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostprocessResult:
    features: list[dict[str, Any]]
    objects_before_filter: int
    objects_after_filter: int
    objects_after_top: int
    params: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class TrainingResult:
    model_name: str
    checkpoint_path: Path | None
    history: list[dict[str, float]]
    best_epoch: int
    best_val_iou: float
    dataset_report: dict[str, Any]
    timing: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Path] = field(default_factory=list)


@dataclass
class PseudolabelResult:
    enabled: bool
    scenes_processed: int
    postprocess_result: PostprocessResult | None = None
    coverage_report: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[Path] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectMetricsResult:
    object_tp: int
    object_fp: int
    object_fn: int
    object_precision: float
    object_recall: float
    object_f1: float
    object_iou_threshold: float
    matching_method: str
    matches: list[dict[str, Any]] = field(default_factory=list)
    unmatched_pred_indices: list[int] = field(default_factory=list)
    unmatched_gt_indices: list[int] = field(default_factory=list)


@dataclass
class JobRunSummary:
    job_id: str
    claim_id: str | None
    task: str
    status: str
    timing: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    mlflow: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
