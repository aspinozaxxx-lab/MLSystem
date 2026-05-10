from __future__ import annotations

from pathlib import Path
from typing import Any


def write_key_value_report(path: str | Path, title: str, payload: dict[str, Any]) -> None:
    lines = [title]
    for key in sorted(payload):
        lines.append(f"{key}={payload[key]}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
