from __future__ import annotations

from pathlib import Path
from typing import Any

from ..pipeline.stages.registry import known_stages
from ..pipeline_runner.config import STAGE_ALIASES
from ..pipeline_runner.stages import DEFAULT_PIPELINE_STAGES, DISPATCHER_STAGE_NAMES, STAGE_POOLS
from ..storage.local_io import read_json
from .job_runner import JobRunner
from .job_store import JobStore
from .models import JobStatusResponse, StageStartRequest, StageStartResponse


def stages_payload() -> dict[str, Any]:
    registry_stages = known_stages()
    return {
        "pipeline_stages": DEFAULT_PIPELINE_STAGES,
        "registry_stages": registry_stages,
        "dispatcher_stages": sorted(DISPATCHER_STAGE_NAMES),
        "stage_pools": STAGE_POOLS,
        "aliases": STAGE_ALIASES,
    }


def validate_stage_name(stage_name: str) -> None:
    payload = stages_payload()
    known = set(payload["pipeline_stages"]) | set(payload["registry_stages"]) | set(payload["dispatcher_stages"])
    if stage_name not in known:
        raise ValueError(f"Unknown stage: {stage_name}")


def start_stage(run_id: str, stage_name: str, request: StageStartRequest, runner: JobRunner | None = None) -> StageStartResponse:
    validate_stage_name(stage_name)
    runner = runner or JobRunner()
    job = runner.start_stage(run_id, stage_name, request)
    return StageStartResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        stage=job.stage,
        state=job.state,
        status_url=f"/api/v1/jobs/{job.job_id}",
        created_at=job.created_at,
    )


def job_status(job_id: str, runner: JobRunner | None = None) -> JobStatusResponse:
    runner = runner or JobRunner()
    return runner.refresh_job(job_id).to_status_response()


def run_summary(run_id: str, status_root: str = "/data/mlsystem/runs") -> dict[str, Any]:
    path = Path(status_root) / run_id / "summary.json"
    return read_json(path, default={}) or {}


def debug_run_stage_sync(run_id: str, stage_name: str, request: StageStartRequest, store: JobStore | None = None) -> JobStatusResponse:
    from .stage_job_worker import run_job

    store = store or JobStore()
    validate_stage_name(stage_name)
    job = store.create_job(run_id, stage_name, request)
    run_job(job.job_id, store.root)
    return store.read_job(job.job_id).to_status_response()
