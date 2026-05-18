from __future__ import annotations

from pathlib import Path
from typing import Any

from ..storage.api import write_json


def write_run_summary(path: Path, payload: dict[str, Any]) -> Path:
    write_json(path, payload)
    return path


def build_timeline_payload(**kwargs: Any) -> dict[str, Any]:
    return {"schema_version": 1, **kwargs}
