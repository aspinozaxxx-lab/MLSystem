from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, model: torch.nn.Module, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **payload}, path)
    return path


def load_model_state(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location=device)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint payload at {path}")
    return state
