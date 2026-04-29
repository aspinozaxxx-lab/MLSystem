from __future__ import annotations

import torch


def dice_iou_precision_recall(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).float()
    target = target.float()
    eps = 1e-7
    tp = float((pred * target).sum().item())
    fp = float((pred * (1 - target)).sum().item())
    fn = float(((1 - pred) * target).sum().item())
    intersection = tp
    union = float(((pred + target) > 0).float().sum().item())
    dice = (2 * intersection + eps) / (float(pred.sum().item() + target.sum().item()) + eps)
    iou = (intersection + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    pixel_f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "pixel_f1": pixel_f1, "f1": pixel_f1}
