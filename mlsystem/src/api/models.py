from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled", "timed_out"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageStartRequest(BaseModel):
    experiment_config: dict[str, Any] = Field(default_factory=dict)
    pipeline_run_id: str | None = None
    status_root: str = "/data/mlsystem/runs"
    force: bool = False
    dry_run: bool = False
    request_id: str | None = None
    source: str = "api"


class StageStartResponse(BaseModel):
    job_id: str
    run_id: str
    stage: str
    state: JobState
    status_url: str
    created_at: str


class JobError(BaseModel):
    type: str
    message: str
    traceback_tail: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    run_id: str
    stage: str
    state: JobState
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    report: dict[str, Any] | None = None
    error: JobError | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ApiJob(BaseModel):
    job_id: str
    run_id: str
    pipeline_run_id: str
    stage: str
    state: JobState = "queued"
    pid: int | None = None
    created_at: str = Field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    request_id: str | None = None
    source: str = "api"
    status_root: str = "/data/mlsystem/runs"
    report: dict[str, Any] | None = None
    error: JobError | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)

    def to_status_response(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            run_id=self.run_id,
            stage=self.stage,
            state=self.state,
            started_at=self.started_at,
            finished_at=self.finished_at,
            duration_sec=self.duration_sec,
            report=self.report,
            error=self.error,
            artifacts=self.artifacts,
        )
