from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class PixelCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, other: "PixelCounts") -> None:
        self.tp += int(other.tp)
        self.fp += int(other.fp)
        self.fn += int(other.fn)
        self.tn += int(other.tn)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def safe_divide(numerator: float, denominator: float, *, empty_value: float = 0.0) -> float:
    if denominator == 0:
        return float(empty_value)
    return float(numerator / denominator)


def metrics_from_counts(counts: PixelCounts) -> dict[str, float | int]:
    tp = float(counts.tp)
    fp = float(counts.fp)
    fn = float(counts.fn)
    tn = float(counts.tn)
    no_positive_support = (tp + fp + fn) == 0
    precision = safe_divide(tp, tp + fp, empty_value=1.0 if no_positive_support else 0.0)
    recall = safe_divide(tp, tp + fn, empty_value=1.0 if no_positive_support else 0.0)
    f1 = safe_divide(2.0 * tp, 2.0 * tp + fp + fn, empty_value=1.0 if no_positive_support else 0.0)
    iou = safe_divide(tp, tp + fp + fn, empty_value=1.0 if no_positive_support else 0.0)
    accuracy = safe_divide(tp + tn, tp + fp + fn + tn, empty_value=1.0)
    return {
        "pixel_tp": counts.tp,
        "pixel_fp": counts.fp,
        "pixel_fn": counts.fn,
        "pixel_tn": counts.tn,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "pixel_iou": iou,
        "pixel_accuracy": accuracy,
        "pixel_dice": f1,
        "dice": f1,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def probabilities_from_logits(logits: torch.Tensor, *, class_index: int = 1) -> torch.Tensor:
    if logits.shape[1:2] == (1,):
        return torch.sigmoid(logits)
    if class_index < 0 or class_index >= int(logits.shape[1]):
        raise ValueError(f"class_index={class_index} is outside logits channels={int(logits.shape[1])}")
    return torch.softmax(logits, dim=1)[:, class_index : class_index + 1]


def binary_mask_from_probabilities(probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return probs >= float(threshold)


def pixel_counts_from_masks(
    pred_mask: torch.Tensor | np.ndarray,
    target_mask: torch.Tensor | np.ndarray,
    *,
    ignore_index: int | float | None = None,
) -> PixelCounts:
    pred = _to_numpy_bool(pred_mask)
    target_raw = _to_numpy_array(target_mask)
    target_raw = _squeeze_single_channel(target_raw)
    if target_raw.shape != pred.shape:
        raise ValueError(f"pred/target shape mismatch: pred={pred.shape}, target={target_raw.shape}")
    valid = np.ones(target_raw.shape, dtype=bool)
    if ignore_index is not None:
        valid = target_raw != ignore_index
    target = target_raw.astype(np.float32) > 0.5
    pred = pred & valid
    target = target & valid
    inv_pred = ~pred & valid
    inv_target = ~target & valid
    return PixelCounts(
        tp=int(np.count_nonzero(pred & target)),
        fp=int(np.count_nonzero(pred & inv_target)),
        fn=int(np.count_nonzero(inv_pred & target)),
        tn=int(np.count_nonzero(inv_pred & inv_target)),
    )


def pixel_counts_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    class_index: int = 1,
    ignore_index: int | float | None = None,
) -> PixelCounts:
    probs = probabilities_from_logits(logits.detach(), class_index=class_index)
    pred = binary_mask_from_probabilities(probs, threshold=threshold)
    return pixel_counts_from_masks(pred, target.detach(), ignore_index=ignore_index)


def binary_segmentation_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    class_index: int = 1,
    ignore_index: int | float | None = None,
) -> dict[str, float | int]:
    return metrics_from_counts(
        pixel_counts_from_logits(logits, target, threshold=threshold, class_index=class_index, ignore_index=ignore_index)
    )


class PixelMetricAccumulator:
    def __init__(self, *, threshold: float = 0.5, ignore_index: int | float | None = None) -> None:
        self.threshold = float(threshold)
        self.ignore_index = ignore_index
        self.counts = PixelCounts()

    def update_from_logits(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        self.counts.update(pixel_counts_from_logits(logits, target, threshold=self.threshold, ignore_index=self.ignore_index))

    def metrics(self) -> dict[str, float | int]:
        payload = metrics_from_counts(self.counts)
        payload["threshold"] = self.threshold
        return payload


def _to_numpy_array(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_numpy_bool(value: torch.Tensor | np.ndarray) -> np.ndarray:
    arr = _to_numpy_array(value)
    arr = _squeeze_single_channel(arr)
    return arr.astype(np.float32) > 0.5


def _squeeze_single_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 3 and arr.shape[-3] == 1:
        return np.squeeze(arr, axis=-3)
    return arr
