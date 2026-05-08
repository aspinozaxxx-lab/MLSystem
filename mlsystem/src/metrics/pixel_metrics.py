from __future__ import annotations

import torch

from .segmentation import binary_segmentation_metrics_from_logits


def dice_iou_precision_recall(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    metrics = binary_segmentation_metrics_from_logits(logits, target, threshold=threshold)
    return {
        "dice": float(metrics["pixel_f1"]),
        "iou": float(metrics["pixel_iou"]),
        "precision": float(metrics["pixel_precision"]),
        "recall": float(metrics["pixel_recall"]),
        "pixel_f1": float(metrics["pixel_f1"]),
        "f1": float(metrics["pixel_f1"]),
    }
