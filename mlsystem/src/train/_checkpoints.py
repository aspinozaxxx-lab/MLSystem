from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        state = payload.get("model_state_dict") or payload.get("state_dict") or payload
        if isinstance(state, dict):
            return state
    raise RuntimeError("Unsupported checkpoint payload: expected model_state_dict/state_dict mapping")


def load_initial_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, *, device: torch.device, strict: bool) -> dict[str, Any]:
    path = Path(str(checkpoint_path))
    if not path.exists():
        raise FileNotFoundError(f"Initial checkpoint is missing: {path}")
    payload = torch.load(str(path), map_location=device)
    state = checkpoint_state_dict(payload)
    result = model.load_state_dict(state, strict=bool(strict))
    return {
        "path": str(path),
        "strict": bool(strict),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def save_checkpoint(path: Path, model: torch.nn.Module, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **payload}, path)
    return path
