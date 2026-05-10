from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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
