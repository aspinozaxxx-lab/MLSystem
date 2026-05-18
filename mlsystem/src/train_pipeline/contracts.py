from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled", "timed_out"]


class PipelineSpec(BaseModel):
    stages: list[str] = Field(default_factory=list)
    stop_on_failure: bool = True
    dry_run: bool = False
    log_mlflow: bool = True


class PipelineRunConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int | None = 1
    run_id: str | None = None
    experiment_id: str
    class_name: str | None = None
    task: str = "train_predict_pseudolabel"
    smoke: bool = False
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)
    images_uri: str = "s3://mlsystems/images/"
    layout_uri: str = "s3://mlsystems/layouts/deforest/"
    scenes_file: str = "scenes.txt"
    annotation_file: str = "auto"
    model: dict[str, Any] = Field(default_factory=dict)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)
    evaluate: dict[str, Any] = Field(default_factory=dict)
    pseudolabel: dict[str, Any] = Field(default_factory=dict)
    postprocess: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    predict: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    mlflow: dict[str, Any] = Field(default_factory=dict)


class PipelineRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    state: str
    progress_percent: int = 0
    current_stage: str | None = None
    current_stage_index: int = 0
    total_stages: int = 0
    created_at: str
    started_at: str | None = None
    updated_at: str
    finished_at: str | None = None
    duration_sec: float | None = None
    pid: int | None = None
    mlflow: dict[str, Any] = Field(default_factory=dict)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error: Any = None


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
    created_at: str
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    request_id: str | None = None
    source: str = "api"
    status_root: str = "/data/mlsystem/runs"
    report: dict[str, Any] | None = None
    error: JobError | None = None
    artifacts: dict[str, Any] = Field(default_factory=dict)


class TrainPipelineError(Exception):
    pass
