from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RuntimeMetrics:
    counters: dict[str, float] = field(default_factory=dict)

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = float(self.counters.get(name, 0.0)) + float(value)

    def set(self, name: str, value: float | int | None) -> None:
        if value is not None:
            self.counters[name] = float(value)

    def snapshot(self) -> dict[str, Any]:
        return dict(self.counters)


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    total = 0
    for item in root.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def gpu_util_snapshot() -> list[dict[str, Any]]:
    if str(os.getenv("INFERENCE_ENGINE_DISABLE_NVIDIA_SMI") or "").lower() in {"1", "true", "yes", "on"}:
        return []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "gpu_util_pct": _float(parts[2]),
                    "memory_util_pct": _float(parts[3]),
                    "memory_used_mb": _float(parts[4]),
                    "memory_total_mb": _float(parts[5]),
                }
            )
    return rows


def _float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
