from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


PseudolabelRunState = Literal["queued", "running", "succeeded", "failed", "cancelled", "timed_out"]


class PseudolabelRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    model_ref: str
    images_uri: str
    class_name: str | None = None
    run_id: str | None = None
    scenes: list[str] | None = None
    layout_uri: str | None = None
    dry_run: bool = False


class PseudolabelRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    experiment_id: str
    state: PseudolabelRunState
    progress_percent: int = 0
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    pid: int | None = None
    inference_engine_job_id: str | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    mlflow: dict[str, Any] = Field(default_factory=dict)
    error: Any = None


class InferencePipelineError(Exception):
    pass
