from __future__ import annotations

from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("optimizer") or config.get("optimizer_name") or "adamw").strip().lower()
    lr = float(config.get("learning_rate", 5e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name in {"adamw", "adam_w"}:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name in {"sgd", "momentum_sgd"}:
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(config.get("momentum", 0.9)),
            nesterov=bool(config.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer={name}; supported: adamw, adam, sgd")
