from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    time_limit_sec: int
    freeze_batchnorm: bool = False
