from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator

JobTask = Literal["noop", "status_check", "inventory", "prepare", "train", "predict", "postprocess", "train_predict_pseudolabel"]

class JobMlflowConfig(BaseModel):
    experiment: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

class JobResourceConfig(BaseModel):
    requires_gpu: bool = False
    cpu_train_slots: int = 0
    cpu_preprocess_slots: int = 0
    workers: int | None = None

class JobSpec(BaseModel):
    schema_version: int = 1
    job_id: str
    task: JobTask
    class_name: str | None = None
    priority: int = 100
    description: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)
    predict: dict[str, Any] = Field(default_factory=dict)
    postprocess: dict[str, Any] = Field(default_factory=dict)
    resources: JobResourceConfig = Field(default_factory=JobResourceConfig)
    mlflow: JobMlflowConfig = Field(default_factory=JobMlflowConfig)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        if not value or any(ch not in allowed for ch in value):
            raise ValueError("job_id may contain only letters, digits, dot, underscore, and dash")
        return value
