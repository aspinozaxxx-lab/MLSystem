from __future__ import annotations

import os
import platform
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .records import ReadyTileSample


def resolve_worker_count(workers: int | None) -> int:
    if workers is not None:
        return max(0, int(workers))
    if platform.system().lower().startswith("win"):
        return 0
    cpu_count = os.cpu_count() or 1
    return max(0, min(8, int(cpu_count) // 2))


def resolve_prefetch_factor(workers: int, prefetch_factor: int | None) -> int | None:
    if workers <= 0:
        return None
    if prefetch_factor is None:
        return 2
    return max(1, int(prefetch_factor))


def tile_collate_fn(samples: list[ReadyTileSample]) -> tuple[list[int], torch.Tensor, torch.Tensor]:
    indices: list[int] = []
    for fallback_index, sample in enumerate(samples):
        raw_index = sample.metadata.get("sample_index", fallback_index)
        indices.append(int(raw_index))
    x = torch.from_numpy(np.stack([sample.image for sample in samples]))
    y = torch.from_numpy(np.stack([sample.mask for sample in samples]))
    return indices, x, y


def tile_collate_with_metadata_fn(samples: list[ReadyTileSample]) -> tuple[list[int], torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    indices, x, y = tile_collate_fn(samples)
    metadata = [dict(sample.metadata) for sample in samples]
    return indices, x, y, metadata


def tile_worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + int(worker_id)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "close"):
        worker_info.dataset.close()


def make_tile_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    workers: int | None = None,
    prefetch_factor: int | None = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    collate_fn: Callable[[list[ReadyTileSample]], Any] = tile_collate_fn,
) -> torch.utils.data.DataLoader:
    resolved_workers = resolve_worker_count(workers)
    resolved_prefetch = resolve_prefetch_factor(resolved_workers, prefetch_factor)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "shuffle": bool(shuffle),
        "num_workers": resolved_workers,
        "collate_fn": collate_fn,
        "pin_memory": bool(pin_memory),
        "generator": generator,
    }
    if resolved_workers > 0:
        kwargs["worker_init_fn"] = tile_worker_init_fn
        kwargs["prefetch_factor"] = resolved_prefetch
        kwargs["persistent_workers"] = bool(persistent_workers)
    return torch.utils.data.DataLoader(dataset, **kwargs)
