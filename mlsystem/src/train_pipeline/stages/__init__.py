from __future__ import annotations

from typing import Any

from .. import experiment_stages
from ..config import DEFAULT_PIPELINE_STAGES, canonical_stage_name, parse_pipeline_run_config
from ..contracts import PipelineRunConfig
from ..run_store import TrainPipelineRunStore
from .context import StageContext
from .registry import get_stage_entrypoint, known_stages, register_stage
from .report import StageCheck, StageFailure, StageReport

DISPATCHER_STAGE_NAMES = experiment_stages.DISPATCHER_STAGE_NAMES
STAGE_POOLS = experiment_stages.STAGE_POOLS


def all_stage_names() -> list[str]:
    return sorted(set(DEFAULT_PIPELINE_STAGES) | set(DISPATCHER_STAGE_NAMES) | set(known_stages()))


def validate_stage_name(stage_name: str) -> str:
    stage = canonical_stage_name(stage_name)
    if stage not in all_stage_names():
        known = ", ".join(all_stage_names())
        raise ValueError(f"Unknown MLSystem pipeline stage: {stage}. Known stages: {known}")
    return stage


def run_stage(stage_name: str, config: PipelineRunConfig | dict[str, Any], store: TrainPipelineRunStore, run_id: str) -> dict[str, Any]:
    stage = validate_stage_name(stage_name)
    parsed = parse_pipeline_run_config(config)
    return experiment_stages.run_stage(stage, parsed, store.bind(run_id, parsed))


def get_registry_stage(stage_name: str) -> Any:
    return get_stage_entrypoint(validate_stage_name(stage_name))


__all__ = [
    "DISPATCHER_STAGE_NAMES",
    "STAGE_POOLS",
    "StageCheck",
    "StageContext",
    "StageFailure",
    "StageReport",
    "all_stage_names",
    "get_stage_entrypoint",
    "get_registry_stage",
    "register_stage",
    "run_stage",
    "validate_stage_name",
]
