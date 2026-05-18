from __future__ import annotations

from pathlib import Path
from typing import Any

from ._run_store import PseudolabelRunStore
from ._runner import InferencePipelineRunner
from .contracts import InferencePipelineError, PseudolabelRun, PseudolabelRunRequest


def create_inference_pipeline(run_root: str | Path | None = None) -> InferencePipelineRunner:
    return InferencePipelineRunner(PseudolabelRunStore(run_root))


def start_pseudolabel_run(request: PseudolabelRunRequest | dict[str, Any]) -> PseudolabelRun:
    return create_inference_pipeline().start_run(request)


def get_pseudolabel_run(run_id: str) -> PseudolabelRun:
    return create_inference_pipeline().get_run(run_id)


def cancel_pseudolabel_run(run_id: str) -> PseudolabelRun:
    return create_inference_pipeline().cancel_run(run_id)


def tail_pseudolabel_log(run_id: str, max_chars: int = 20000) -> str:
    return create_inference_pipeline().tail_log(run_id, max_chars=max_chars)


__all__ = [
    "InferencePipelineError",
    "InferencePipelineRunner",
    "PseudolabelRun",
    "PseudolabelRunRequest",
    "cancel_pseudolabel_run",
    "create_inference_pipeline",
    "get_pseudolabel_run",
    "start_pseudolabel_run",
    "tail_pseudolabel_log",
]
