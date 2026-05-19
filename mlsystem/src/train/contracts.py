from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class TrainError(RuntimeError):
    """Public training module error."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    input_channels: int | None = None
    output_channels: int = 1
    requires_optional_dependency: str | None = None


@dataclass
class TrainConfig:
    model_name: str = "tiny_unet_4ch"
    input_channels: int = 4
    output_channels: int = 1
    base_channels: int = 8
    epochs: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    scheduler: dict[str, Any] | str | None = None
    loss: dict[str, Any] = field(default_factory=dict)
    metric_threshold: float = 0.5
    metric_thresholds: list[float] | None = None
    device: str | None = None
    require_gpu: bool = False
    batch_size: int | None = None
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    early_stopping_patience: int | None = None
    initial_checkpoint_path: str | None = None
    initial_checkpoint_strict: bool = True


@dataclass
class CheckpointArtifact:
    path: str
    model_name: str
    epoch: int
    metric_name: str
    metric_value: float


@dataclass
class EpochMetrics:
    epoch: int
    metrics: dict[str, float | int]


@dataclass
class TrainProgressEvent:
    stage: str
    epoch: int | None = None
    metrics: dict[str, float | int] = field(default_factory=dict)
    message: str | None = None


class TrainProgressSink(Protocol):
    def emit(self, event: TrainProgressEvent) -> None:
        ...


@dataclass
class TrainRequest:
    config: TrainConfig
    train_dataloader: Any | None = None
    val_dataloader: Any | None = None
    output_dir: Path | str | None = None
    experiment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    status: str
    model_name: str
    device: str
    cuda_available: bool
    epochs_completed: int
    best_epoch: int
    best_val_pixel_f1: float
    best_val_iou: float
    last_val_pixel_f1: float
    checkpoint: CheckpointArtifact | None
    history: list[EpochMetrics]
    mlflow_params: dict[str, Any] = field(default_factory=dict)
    mlflow_metrics: dict[str, float | int] = field(default_factory=dict)
    mlflow_artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
