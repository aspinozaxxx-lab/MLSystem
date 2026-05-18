from __future__ import annotations

import torch

from .contracts import TrainError


def resolve_device(device_name: str | None, *, require_gpu: bool = False) -> torch.device:
    cuda_available = torch.cuda.is_available()
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if cuda_available else "cpu")
    if require_gpu and device.type != "cuda":
        raise TrainError("GPU training was requested, but CUDA is not available.")
    return device
