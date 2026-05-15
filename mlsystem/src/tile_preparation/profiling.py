from __future__ import annotations

import os
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


PROFILE_ENV = "MLSYSTEM_TILE_PREP_PROFILE"


def profiling_enabled() -> bool:
    return str(os.getenv(PROFILE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TilePrepProfiler:
    enabled: bool = field(default_factory=profiling_enabled)
    values: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, name: str, seconds: float) -> None:
        if self.enabled:
            self.values[str(name)].append(float(seconds))

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    @contextmanager
    def time_sample(self, name: str, sample_values: dict[str, float]) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            self.add(name, seconds)
            sample_values[str(name)] = float(sample_values.get(str(name), 0.0)) + float(seconds)

    def summary(self, *, reset: bool = False) -> dict[str, float]:
        if not self.enabled:
            return {}
        result: dict[str, float] = {}
        for name, samples in self.values.items():
            if not samples:
                continue
            ordered = sorted(samples)
            result[f"tile_prep/{name}_count"] = float(len(ordered))
            result[f"tile_prep/{name}_total"] = float(sum(ordered))
            result[f"tile_prep/{name}_mean"] = float(sum(ordered) / len(ordered))
            result[f"tile_prep/{name}_p50"] = float(statistics.median(ordered))
            result[f"tile_prep/{name}_p95"] = float(_percentile(ordered, 0.95))
        if reset:
            self.values.clear()
        return result


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
