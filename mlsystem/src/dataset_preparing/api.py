from __future__ import annotations

from ._inspection import inspect_dataset as _inspect_dataset
from ._preparation import prepare_training_dataset as _prepare_training_dataset
from .contracts import (
    DatasetIdentity,
    DatasetInspectionRequest,
    DatasetInspectionResult,
    DatasetPreparingError,
    TrainingDatasetPrepareRequest,
    TrainingDatasetPrepareResult,
)


def inspect_dataset(request: DatasetInspectionRequest) -> DatasetInspectionResult:
    if not request.images_uri:
        raise DatasetPreparingError("DatasetInspectionRequest.images_uri is required.")
    if not request.layout_uri:
        raise DatasetPreparingError("DatasetInspectionRequest.layout_uri is required.")
    return _inspect_dataset(request)


def prepare_training_dataset(request: TrainingDatasetPrepareRequest) -> TrainingDatasetPrepareResult:
    if not request.experiment_id:
        raise DatasetPreparingError("TrainingDatasetPrepareRequest.experiment_id is required.")
    if not request.run_id:
        raise DatasetPreparingError("TrainingDatasetPrepareRequest.run_id is required.")
    return _prepare_training_dataset(request)


__all__ = [
    "DatasetIdentity",
    "DatasetInspectionRequest",
    "DatasetInspectionResult",
    "DatasetPreparingError",
    "TrainingDatasetPrepareRequest",
    "TrainingDatasetPrepareResult",
    "inspect_dataset",
    "prepare_training_dataset",
]
