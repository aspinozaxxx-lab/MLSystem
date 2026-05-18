from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from ..api.security import mask_secrets, mask_text
from ..storage.local_io import read_json
from .config import DEFAULT_PIPELINE_STAGES, STAGE_ALIASES
from .contracts import ApiJob, JobError, JobState, JobStatusResponse, StageStartRequest, StageStartResponse
from .run_store import utc_now
from .stages import DISPATCHER_STAGE_NAMES, STAGE_POOLS, validate_stage_name as _validate_stage_name


def default_stage_job_root() -> Path:
    return Path(os.getenv("MLSYSTEM_API_JOB_ROOT", "/data/mlsystem/api/jobs"))


class StageJobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else default_stage_job_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(self, run_id: str, stage: str, request: StageStartRequest) -> ApiJob:
        canonical_stage = validate_stage_name(stage)
        job_id = f"{run_id}_{canonical_stage}_{uuid.uuid4().hex[:12]}"
        job = ApiJob(
            job_id=job_id,
            run_id=run_id,
            pipeline_run_id=request.pipeline_run_id or run_id,
            stage=canonical_stage,
            state="queued",
            created_at=utc_now(),
            request_id=request.request_id,
            source=request.source,
            status_root=request.status_root,
        )
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self.write_json(job_dir / "request.json", mask_secrets(request.model_dump()))
        self.write_job(job)
        return job

    def job_dir(self, job_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in job_id)
        return self.root / safe

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "request.json"

    def read_job(self, job_id: str) -> ApiJob:
        return ApiJob.model_validate(self.read_json(self.job_path(job_id)))

    def write_job(self, job: ApiJob) -> None:
        self.write_json(self.job_path(job.job_id), mask_secrets(job.model_dump()))

    def update_job(self, job_id: str, **updates: Any) -> ApiJob:
        job = self.read_job(job_id)
        payload = job.model_dump()
        payload.update(updates)
        updated = ApiJob.model_validate(payload)
        self.write_job(updated)
        return updated

    def write_report(self, job_id: str, report: dict[str, Any]) -> None:
        self.write_json(self.job_dir(job_id) / "report.json", mask_secrets(report))

    def write_error(self, job_id: str, error: JobError | dict[str, Any]) -> None:
        payload = error.model_dump() if isinstance(error, JobError) else error
        self.write_json(self.job_dir(job_id) / "error.json", mask_secrets(payload))

    def read_request(self, job_id: str) -> dict[str, Any]:
        return self.read_json(self.request_path(job_id))

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def finish_stage_job(
    store: StageJobStore,
    job: ApiJob,
    *,
    state: str,
    report: dict[str, Any] | None = None,
    error: JobError | None = None,
) -> ApiJob:
    finished_at = utc_now()
    started = job.started_at or job.created_at
    duration = None
    try:
        duration = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started)).total_seconds()
    except Exception:
        duration = None
    artifacts = (report or {}).get("artifacts") or job.artifacts
    updated = store.update_job(
        job.job_id,
        state=state,
        finished_at=finished_at,
        duration_sec=duration,
        report=report,
        error=error,
        artifacts=artifacts,
    )
    if report is not None:
        store.write_report(job.job_id, report)
    if error is not None:
        store.write_error(job.job_id, error)
    return updated


def stage_job_worker_module_name() -> str:
    configured = os.getenv("MLSYSTEM_API_WORKER_MODULE")
    if configured:
        return configured
    if Path("/opt/mlsystem/src").exists():
        return "src.train_pipeline.stage_job_worker"
    return "mlsystem.src.train_pipeline.stage_job_worker"


class StageJobRunner:
    def __init__(self, store: StageJobStore | None = None) -> None:
        self.store = store or StageJobStore()

    def start_stage(self, run_id: str, stage: str, request: StageStartRequest) -> ApiJob:
        job = self.store.create_job(run_id, stage, request)
        if request.dry_run:
            return finish_stage_job(
                self.store,
                job,
                state="succeeded",
                report={"status": "success", "summary": "dry_run=true; stage was not executed", "stage": job.stage},
            )
        cmd = [
            sys.executable,
            "-m",
            stage_job_worker_module_name(),
            "--job-id",
            job.job_id,
            "--job-root",
            str(self.store.root),
        ]
        try:
            process = subprocess.Popen(cmd, cwd=os.getenv("MLSYSTEM_API_WORKDIR") or None)  # noqa: S603
        except Exception as exc:  # noqa: BLE001
            error = JobError(type=type(exc).__name__, message=mask_text(f"Failed to start worker subprocess: {exc}"))
            return finish_stage_job(self.store, job, state="failed", error=error, report={"status": "failed", "error": error.model_dump()})
        return self.store.update_job(job.job_id, pid=process.pid)

    def refresh_job(self, job_id: str) -> ApiJob:
        job = self.store.read_job(job_id)
        if job.state not in {"queued", "running"} or not job.pid:
            return job
        if _pid_running(job.pid):
            return job
        return self.store.read_job(job_id)


def stages_payload() -> dict[str, Any]:
    from .stages.registry import known_stages

    registry_stages = known_stages()
    return {
        "pipeline_stages": DEFAULT_PIPELINE_STAGES,
        "registry_stages": registry_stages,
        "dispatcher_stages": sorted(DISPATCHER_STAGE_NAMES),
        "stage_pools": STAGE_POOLS,
        "aliases": STAGE_ALIASES,
    }


def validate_stage_name(stage_name: str) -> str:
    try:
        return _validate_stage_name(stage_name)
    except ValueError as exc:
        raise ValueError(f"Unknown stage: {stage_name}") from exc


def start_stage(run_id: str, stage_name: str, request: StageStartRequest, runner: StageJobRunner | None = None) -> StageStartResponse:
    canonical_stage = validate_stage_name(stage_name)
    runner = runner or StageJobRunner()
    job = runner.start_stage(run_id, canonical_stage, request)
    return StageStartResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        stage=job.stage,
        state=job.state,
        status_url=f"/api/v1/jobs/{job.job_id}",
        created_at=job.created_at,
    )


def job_status(job_id: str, runner: StageJobRunner | None = None) -> JobStatusResponse:
    runner = runner or StageJobRunner()
    return job_to_status_response(runner.refresh_job(job_id))


def run_summary(run_id: str, status_root: str = "/data/mlsystem/runs") -> dict[str, Any]:
    path = Path(status_root) / run_id / "summary.json"
    return read_json(path, default={}) or {}


def debug_run_stage_sync(run_id: str, stage_name: str, request: StageStartRequest, store: StageJobStore | None = None) -> JobStatusResponse:
    from .stage_job_worker import run_job

    store = store or StageJobStore()
    canonical_stage = validate_stage_name(stage_name)
    job = store.create_job(run_id, canonical_stage, request)
    run_job(job.job_id, store.root)
    return job_to_status_response(store.read_job(job.job_id))


def job_to_status_response(job: ApiJob) -> JobStatusResponse:
    return JobStatusResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        stage=job.stage,
        state=job.state,
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_sec=job.duration_sec,
        report=job.report,
        error=job.error,
        artifacts=job.artifacts,
    )


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


__all__ = [
    "ApiJob",
    "JobError",
    "JobState",
    "JobStatusResponse",
    "StageJobRunner",
    "StageJobStore",
    "StageStartRequest",
    "StageStartResponse",
    "debug_run_stage_sync",
    "default_stage_job_root",
    "finish_stage_job",
    "job_status",
    "run_summary",
    "stage_job_worker_module_name",
    "stages_payload",
    "start_stage",
    "validate_stage_name",
]
