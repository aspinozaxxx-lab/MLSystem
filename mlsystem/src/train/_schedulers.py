from __future__ import annotations

from typing import Any

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    cfg = config.get("scheduler")
    if cfg is None:
        cfg = {"name": config.get("scheduler_name") or "none"}
    elif isinstance(cfg, str):
        cfg = {"name": cfg}
    elif not isinstance(cfg, dict):
        cfg = {"name": "none"}
    name = str(cfg.get("name") or cfg.get("type") or "none").strip().lower()
    if name in {"", "none", "off", "disabled"}:
        return None
    if name in {"cosine", "cosine_annealing"}:
        t_max = int(cfg.get("t_max") or config.get("scheduler_t_max") or epochs)
        eta_min = float(cfg.get("eta_min") or config.get("scheduler_eta_min") or 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, t_max), eta_min=eta_min)
    if name in {"step", "step_lr"}:
        step_size = int(cfg.get("step_size") or config.get("scheduler_step_size") or max(1, epochs // 3))
        gamma = float(cfg.get("gamma") or config.get("scheduler_gamma") or 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, step_size), gamma=gamma)
    raise ValueError(f"Unsupported scheduler={name}; supported: none, cosine, step")
