from __future__ import annotations

import torch


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    eps = 1e-7
    dice = 1 - ((2 * (probs * target).sum() + eps) / (probs.sum() + target.sum() + eps))
    return bce + dice
