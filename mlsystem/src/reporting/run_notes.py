from __future__ import annotations

from typing import Any

from ..mlflow_adapter.api import build_run_note, compact_run_label


def build_job_run_note(**kwargs: Any) -> str:
    return build_run_note(**kwargs)


def build_compact_run_label(job_id: str, model_name: str | None = None, tile_size: int | str | None = None) -> str:
    return compact_run_label(job_id, model_name, tile_size)
