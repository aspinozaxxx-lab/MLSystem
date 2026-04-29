from __future__ import annotations

from pathlib import Path
from typing import Any

from .local_io import read_json, write_json


def load_manifest(path: Path, default: Any = None) -> Any:
    return read_json(path, default)


def write_manifest(path: Path, payload: Any) -> None:
    write_json(path, payload)
