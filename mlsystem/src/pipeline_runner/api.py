from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_PIPELINE_STAGES,
    STAGE_ALIASES,
    dump_trace_yaml,
    load_trace_file,
    load_trace_payload,
    parse_pipeline_run_config,
)
from .contracts import (
    ApiJob,
    JobError,
    JobState,
    JobStatusResponse,
    PipelineRun,
    PipelineRunConfig,
    PipelineRunnerError,
    PipelineSpec,
    StageStartRequest,
    StageStartResponse,
)
from .run_store import PipelineRunStore
from .runner import PipelineRunner
from .stage_jobs import (
    StageJobRunner,
    StageJobStore,
    debug_run_stage_sync,
    job_status,
    run_summary,
    stages_payload,
    start_stage,
    validate_stage_name,
)


def create_run_store(run_root: str | Path | None = None) -> PipelineRunStore:
    return PipelineRunStore(run_root)


def create_pipeline_runner(run_root: str | Path | None = None) -> PipelineRunner:
    return PipelineRunner(create_run_store(run_root))


def create_stage_job_store(job_root: str | Path | None = None) -> StageJobStore:
    return StageJobStore(Path(job_root) if job_root is not None else None)


def create_stage_job_runner(job_root: str | Path | None = None) -> StageJobRunner:
    return StageJobRunner(create_stage_job_store(job_root))


def start_pipeline_run(config: PipelineRunConfig | dict[str, Any], *, source: str = "api", run_root: str | Path | None = None) -> PipelineRun:
    runner = create_pipeline_runner(run_root)
    return runner.start_run(parse_pipeline_run_config(config), source=source)


__all__ = [
    "ApiJob",
    "DEFAULT_PIPELINE_STAGES",
    "JobError",
    "JobState",
    "JobStatusResponse",
    "PipelineRun",
    "PipelineRunConfig",
    "PipelineRunStore",
    "PipelineRunner",
    "PipelineRunnerError",
    "PipelineSpec",
    "STAGE_ALIASES",
    "StageJobRunner",
    "StageJobStore",
    "StageStartRequest",
    "StageStartResponse",
    "create_pipeline_runner",
    "create_run_store",
    "create_stage_job_runner",
    "create_stage_job_store",
    "debug_run_stage_sync",
    "dump_trace_yaml",
    "job_status",
    "load_trace_file",
    "load_trace_payload",
    "parse_pipeline_run_config",
    "run_summary",
    "stages_payload",
    "start_pipeline_run",
    "start_stage",
    "validate_stage_name",
]
