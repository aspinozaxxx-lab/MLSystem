from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceEngineSettings:
    api_token: str | None = None
    job_root: Path = Path("/data/mlsystem/inference-engine/jobs")
    spool_root: Path = Path("/data/mlsystem/inference-engine/spool")
    artifact_root: Path = Path("/data/mlsystem/inference-engine/artifacts")
    logs_root: Path = Path("/data/mlsystem/inference-engine/logs")
    rabbitmq_url: str = "amqp://guest:guest@rabbitmq:5672/"
    triton_url: str = "http://triton:8000"
    use_rabbitmq: bool = False
    max_wait_ms: int = 50
    default_triton_batch_size: int = 8
    default_worker_concurrency: int = 1

    @classmethod
    def from_env(cls) -> "InferenceEngineSettings":
        return cls(
            api_token=os.getenv("INFERENCE_ENGINE_API_TOKEN"),
            job_root=Path(os.getenv("INFERENCE_ENGINE_JOB_ROOT", "/data/mlsystem/inference-engine/jobs")),
            spool_root=Path(os.getenv("INFERENCE_ENGINE_SPOOL_ROOT", "/data/mlsystem/inference-engine/spool")),
            artifact_root=Path(os.getenv("INFERENCE_ENGINE_ARTIFACT_ROOT", "/data/mlsystem/inference-engine/artifacts")),
            logs_root=Path(os.getenv("INFERENCE_ENGINE_LOGS_ROOT", "/data/mlsystem/inference-engine/logs")),
            rabbitmq_url=os.getenv("INFERENCE_ENGINE_RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/"),
            triton_url=os.getenv("INFERENCE_ENGINE_TRITON_URL", "http://triton:8000"),
            use_rabbitmq=str(os.getenv("INFERENCE_ENGINE_USE_RABBITMQ", "")).lower() in {"1", "true", "yes", "on"},
            max_wait_ms=int(os.getenv("INFERENCE_ENGINE_MAX_WAIT_MS", "50")),
            default_triton_batch_size=int(os.getenv("INFERENCE_ENGINE_DEFAULT_TRITON_BATCH_SIZE", "8")),
            default_worker_concurrency=int(os.getenv("INFERENCE_ENGINE_WORKER_CONCURRENCY", "1")),
        )

    def ensure_dirs(self) -> None:
        for path in (self.job_root, self.spool_root, self.artifact_root, self.logs_root):
            path.mkdir(parents=True, exist_ok=True)
