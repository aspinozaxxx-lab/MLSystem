from __future__ import annotations

from typing import Any

from .contracts import MetricsError
from .evaluation import numeric_metrics
from .pixel_metrics import dice_iou_precision_recall
from .segmentation import (
    PixelCounts,
    PixelMetricAccumulator,
    WeightedLossAccumulator,
    binary_mask_from_probabilities,
    binary_segmentation_metrics_from_logits,
    metrics_from_counts,
    pixel_counts_from_logits,
    pixel_counts_from_masks,
    probabilities_from_logits,
    safe_divide,
)


def clean_geometries(*args: Any, **kwargs: Any) -> Any:
    from .object_metrics import clean_geometries as impl

    return impl(*args, **kwargs)


def compute_object_f1(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .object_metrics import compute_object_f1 as impl

    return impl(*args, **kwargs)


def compute_object_metrics_by_scene(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .object_metrics import compute_object_metrics_by_scene as impl

    return impl(*args, **kwargs)


def compute_pairwise_iou(*args: Any, **kwargs: Any) -> list[list[float]]:
    from .object_metrics import compute_pairwise_iou as impl

    return impl(*args, **kwargs)


def match_objects_by_iou(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .object_metrics import match_objects_by_iou as impl

    return impl(*args, **kwargs)


def prepare_geometries(*args: Any, **kwargs: Any) -> Any:
    from .object_metrics import prepare_geometries as impl

    return impl(*args, **kwargs)


def reproject_geometries(*args: Any, **kwargs: Any) -> Any:
    from .object_metrics import reproject_geometries as impl

    return impl(*args, **kwargs)


def write_object_metrics_artifacts(*args: Any, **kwargs: Any) -> Any:
    from .object_metric_artifacts import write_object_metrics_artifacts as impl

    return impl(*args, **kwargs)


__all__ = [
    "MetricsError",
    "PixelCounts",
    "PixelMetricAccumulator",
    "WeightedLossAccumulator",
    "binary_mask_from_probabilities",
    "binary_segmentation_metrics_from_logits",
    "clean_geometries",
    "compute_object_f1",
    "compute_object_metrics_by_scene",
    "compute_pairwise_iou",
    "dice_iou_precision_recall",
    "match_objects_by_iou",
    "metrics_from_counts",
    "numeric_metrics",
    "pixel_counts_from_logits",
    "pixel_counts_from_masks",
    "prepare_geometries",
    "probabilities_from_logits",
    "reproject_geometries",
    "safe_divide",
    "write_object_metrics_artifacts",
]
