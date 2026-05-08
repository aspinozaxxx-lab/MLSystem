from __future__ import annotations

from typing import Any

import torch


def _float_config(config: dict[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    if value is None:
        return float(default)
    return float(value)


def _positive_weight(config: dict[str, Any], logits: torch.Tensor) -> torch.Tensor | None:
    value = config.get("pos_weight", config.get("positive_weight"))
    if value is None:
        return None
    return torch.as_tensor(float(value), dtype=logits.dtype, device=logits.device)


def _dice_loss(probs: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    return 1 - ((2 * (probs * target).sum() + eps) / (probs.sum() + target.sum() + eps))


def _focal_loss(logits: torch.Tensor, target: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    gamma = _float_config(config, "focal_gamma", 2.0)
    alpha = config.get("focal_alpha", config.get("alpha"))
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=_positive_weight(config, logits),
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, probs, 1.0 - probs)
    focal = ((1.0 - pt).clamp_min(0.0) ** gamma) * bce
    if alpha is not None:
        alpha_value = float(alpha)
        alpha_factor = torch.where(target > 0.5, torch.as_tensor(alpha_value, dtype=logits.dtype, device=logits.device), torch.as_tensor(1.0 - alpha_value, dtype=logits.dtype, device=logits.device))
        focal = alpha_factor * focal
    return focal.mean()


def _tversky_loss(probs: torch.Tensor, target: torch.Tensor, config: dict[str, Any], eps: float) -> torch.Tensor:
    alpha = _float_config(config, "tversky_alpha", 0.5)
    beta = _float_config(config, "tversky_beta", 0.5)
    tp = (probs * target).sum()
    fp = (probs * (1.0 - target)).sum()
    fn = ((1.0 - probs) * target).sum()
    return 1.0 - ((tp + eps) / (tp + alpha * fp + beta * fn + eps))


def segmentation_loss_components(logits: torch.Tensor, target: torch.Tensor, config: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    cfg = dict(config or {})
    loss_name = str(cfg.get("name") or cfg.get("type") or "bce_dice").strip().lower()
    eps = _float_config(cfg, "eps", 1e-7)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=_positive_weight(cfg, logits))
    probs = torch.sigmoid(logits)
    dice = _dice_loss(probs, target, eps)
    focal = _focal_loss(logits, target, cfg)
    tversky = _tversky_loss(probs, target, cfg, eps)

    bce_weight = _float_config(cfg, "bce_weight", 1.0)
    dice_weight = _float_config(cfg, "dice_weight", 1.0)
    focal_weight = _float_config(cfg, "focal_weight", 1.0)
    tversky_weight = _float_config(cfg, "tversky_weight", 1.0)

    if loss_name in {"bce", "binary_cross_entropy"}:
        total = bce_weight * bce
    elif loss_name in {"dice"}:
        total = dice_weight * dice
    elif loss_name in {"focal"}:
        total = focal_weight * focal
    elif loss_name in {"tversky"}:
        total = tversky_weight * tversky
    elif loss_name in {"focal_dice", "dice_focal"}:
        total = focal_weight * focal + dice_weight * dice
    elif loss_name in {"bce_tversky", "tversky_bce"}:
        total = bce_weight * bce + tversky_weight * tversky
    elif loss_name in {"focal_tversky", "tversky_focal"}:
        total = focal_weight * focal + tversky_weight * tversky
    elif loss_name in {"bce_focal_dice", "bce_dice_focal"}:
        total = bce_weight * bce + focal_weight * focal + dice_weight * dice
    else:
        total = bce_weight * bce + dice_weight * dice

    return {
        "loss_total": total,
        "loss_bce": bce,
        "loss_dice": dice,
        "loss_focal": focal,
        "loss_tversky": tversky,
    }


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, config: dict[str, Any] | None = None) -> torch.Tensor:
    return segmentation_loss_components(logits, target, config=config)["loss_total"]
