from __future__ import annotations

import math
from typing import TypeVar

T = TypeVar("T")


def train_val_split(items: list[T], train_fraction: float = 0.75) -> tuple[list[T], list[T]]:
    if not items:
        return [], []
    split_idx = max(1, int(math.ceil(len(items) * train_fraction)))
    train_items = items[:split_idx]
    val_items = items[split_idx:] or items[-1:]
    return train_items, val_items
