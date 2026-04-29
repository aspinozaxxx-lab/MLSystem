from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


def make_batches(samples: list[T], batch_size: int) -> list[list[T]]:
    return [samples[i : i + batch_size] for i in range(0, len(samples), max(1, batch_size))]
