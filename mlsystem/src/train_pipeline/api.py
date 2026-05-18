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
    JobMlflowConfig,
    JobResourceConfig,
    JobSpec,
    JobState,
    JobTask,
    JobStatusResponse,
    PipelineRun,
    PipelineRunConfig,
    TrainPipelineError,
    PipelineSpec,
    StageStartRequest,
    StageStartResponse,
)
from .run_store import TrainPipelineRunStore
from .runner import TrainPipelineRunner
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


def create_run_store(run_root: str | Path | None = None) -> TrainPipelineRunStore:
    return TrainPipelineRunStore(run_root)


def create_train_pipeline(run_root: str | Path | None = None) -> TrainPipelineRunner:
    return TrainPipelineRunner(create_run_store(run_root))


def create_stage_job_store(job_root: str | Path | None = None) -> StageJobStore:
    return StageJobStore(Path(job_root) if job_root is not None else None)


def create_stage_job_runner(job_root: str | Path | None = None) -> StageJobRunner:
    return StageJobRunner(create_stage_job_store(job_root))


def start_train_pipeline_run(config: PipelineRunConfig | dict[str, Any], *, source: str = "api", run_root: str | Path | None = None) -> PipelineRun:
    runner = create_train_pipeline(run_root)
    return runner.start_run(parse_pipeline_run_config(config), source=source)


def get_train_pipeline_run(run_id: str, *, run_root: str | Path | None = None) -> PipelineRun:
    return create_train_pipeline(run_root).refresh_run(run_id)


def cancel_train_pipeline_run(run_id: str, *, run_root: str | Path | None = None) -> PipelineRun:
    return create_train_pipeline(run_root).cancel_run(run_id)


def tail_train_pipeline_log(run_id: str, max_chars: int = 20000, *, run_root: str | Path | None = None) -> str:
    return create_run_store(run_root).tail_log(run_id, max_chars=max_chars)


__all__ = [
    "ApiJob",
    "DEFAULT_PIPELINE_STAGES",
    "JobError",
    "JobMlflowConfig",
    "JobResourceConfig",
    "JobSpec",
    "JobState",
    "JobTask",
    "JobStatusResponse",
    "PipelineRun",
    "PipelineRunConfig",
    "TrainPipelineRunStore",
    "TrainPipelineRunner",
    "TrainPipelineError",
    "PipelineSpec",
    "STAGE_ALIASES",
    "StageJobRunner",
    "StageJobStore",
    "StageStartRequest",
    "StageStartResponse",
    "create_train_pipeline",
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
    "start_train_pipeline_run",
    "get_train_pipeline_run",
    "cancel_train_pipeline_run",
    "tail_train_pipeline_log",
    "start_stage",
    "validate_stage_name",
]
