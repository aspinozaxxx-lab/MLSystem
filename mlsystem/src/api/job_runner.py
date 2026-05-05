from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .job_store import JobStore, finish_job
from .models import ApiJob, JobError, StageStartRequest
from .security import mask_text


def worker_module_name() -> str:
    configured = os.getenv("MLSYSTEM_API_WORKER_MODULE")
    if configured:
        return configured
    if Path("/opt/mlsystem/src").exists():
        return "src.api.stage_job_worker"
    return "mlsystem.src.api.stage_job_worker"


class JobRunner:
    def __init__(self, store: JobStore | None = None) -> None:
        self.store = store or JobStore()

    def start_stage(self, run_id: str, stage: str, request: StageStartRequest) -> ApiJob:
        job = self.store.create_job(run_id, stage, request)
        if request.dry_run:
            return finish_job(
                self.store,
                job,
                state="succeeded",
                report={"status": "success", "summary": "dry_run=true; stage was not executed", "stage": stage},
            )
        cmd = [
            sys.executable,
            "-m",
            worker_module_name(),
            "--job-id",
            job.job_id,
            "--job-root",
            str(self.store.root),
        ]
        try:
            process = subprocess.Popen(cmd, cwd=os.getenv("MLSYSTEM_API_WORKDIR") or None)  # noqa: S603 - controlled module invocation.
        except Exception as exc:  # noqa: BLE001
            error = JobError(type=type(exc).__name__, message=mask_text(f"Failed to start worker subprocess: {exc}"))
            return finish_job(self.store, job, state="failed", error=error, report={"status": "failed", "error": error.model_dump()})
        return self.store.update_job(job.job_id, pid=process.pid)

    def refresh_job(self, job_id: str) -> ApiJob:
        job = self.store.read_job(job_id)
        if job.state not in {"queued", "running"} or not job.pid:
            return job
        if _pid_running(job.pid):
            return job
        return self.store.read_job(job_id)


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
