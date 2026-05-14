from __future__ import annotations

from typing import Any

from ..pipeline import experiment_stages
from ..pipeline.stages.registry import get_stage_entrypoint, known_stages
from .config import DEFAULT_PIPELINE_STAGES, PipelineRunConfig, canonical_stage_name
from .progress import STAGE_WEIGHTS
from .run_store import PipelineRunStore

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


def run_stage(stage_name: str, config: PipelineRunConfig | dict[str, Any], store: PipelineRunStore, run_id: str) -> dict[str, Any]:
    stage = validate_stage_name(stage_name)
    parsed = config if isinstance(config, PipelineRunConfig) else PipelineRunConfig.model_validate(config)
    return experiment_stages.run_stage(stage, parsed.model_dump(mode="json"), run_id, store.root)


def get_registry_stage(stage_name: str) -> Any:
    return get_stage_entrypoint(validate_stage_name(stage_name))
