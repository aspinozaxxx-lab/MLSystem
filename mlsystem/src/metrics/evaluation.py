from __future__ import annotations

from typing import Any


def numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in payload.items() if isinstance(value, (int, float, bool))}
