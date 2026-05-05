from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from .models import ApiJob, JobError, StageStartRequest, utc_now
from .security import mask_secrets


def default_job_root() -> Path:
    return Path(os.getenv("MLSYSTEM_API_JOB_ROOT", "/data/mlsystem/api/jobs"))


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else default_job_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_job(self, run_id: str, stage: str, request: StageStartRequest) -> ApiJob:
        job_id = f"{run_id}_{stage}_{uuid.uuid4().hex[:12]}"
        job = ApiJob(
            job_id=job_id,
            run_id=run_id,
            airflow_run_id=request.airflow_run_id or run_id,
            stage=stage,
            state="queued",
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


def finish_job(store: JobStore, job: ApiJob, *, state: str, report: dict[str, Any] | None = None, error: JobError | None = None) -> ApiJob:
    finished_at = utc_now()
    started = job.started_at or job.created_at
    duration = None
    try:
        from datetime import datetime

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
