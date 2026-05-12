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
    triton_grpc_url: str | None = None
    triton_transport: str = "http"
    triton_shared_memory: str = "off"
    use_rabbitmq: bool = False
    publish_rabbitmq_events: bool = False
    max_wait_ms: int = 50
    default_triton_batch_size: int = 8
    default_worker_concurrency: int = 1
    hotpath_mode: str = "disk_spool"
    fused_max_in_memory_tiles: int | None = None
    fused_max_in_memory_bytes: int = 16 * 1024 * 1024 * 1024
    fused_read_workers: int = 8
    fused_batch_workers: int = 1
    cleanup_intermediates_enabled: bool = True
    cleanup_events_max_bytes: int = 5 * 1024 * 1024

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
            triton_grpc_url=os.getenv("INFERENCE_ENGINE_TRITON_GRPC_URL"),
            triton_transport=os.getenv("INFERENCE_ENGINE_TRITON_TRANSPORT", "http").lower(),
            triton_shared_memory=os.getenv("INFERENCE_ENGINE_TRITON_SHARED_MEMORY", "off").lower(),
            use_rabbitmq=str(os.getenv("INFERENCE_ENGINE_USE_RABBITMQ", "")).lower() in {"1", "true", "yes", "on"},
            publish_rabbitmq_events=str(os.getenv("INFERENCE_ENGINE_PUBLISH_RABBITMQ_EVENTS", "")).lower() in {"1", "true", "yes", "on"},
            max_wait_ms=int(os.getenv("INFERENCE_ENGINE_MAX_WAIT_MS", "500")),
            default_triton_batch_size=int(os.getenv("INFERENCE_ENGINE_DEFAULT_TRITON_BATCH_SIZE", "8")),
            default_worker_concurrency=int(os.getenv("INFERENCE_ENGINE_WORKER_CONCURRENCY", "1")),
            hotpath_mode=os.getenv("INFERENCE_ENGINE_HOTPATH_MODE", "disk_spool").lower(),
            fused_max_in_memory_tiles=_optional_int(os.getenv("INFERENCE_ENGINE_FUSED_MAX_IN_MEMORY_TILES")),
            fused_max_in_memory_bytes=int(os.getenv("INFERENCE_ENGINE_FUSED_MAX_IN_MEMORY_BYTES", str(16 * 1024 * 1024 * 1024))),
            fused_read_workers=int(os.getenv("INFERENCE_ENGINE_FUSED_READ_WORKERS", "8")),
            fused_batch_workers=int(os.getenv("INFERENCE_ENGINE_FUSED_BATCH_WORKERS", "1")),
            cleanup_intermediates_enabled=str(
                os.getenv("INFERENCE_ENGINE_CLEANUP_INTERMEDIATES", os.getenv("INFERENCE_ENGINE_CLEANUP_INTERMEDIATES_ON_TERMINAL", "true"))
            ).lower()
            in {"1", "true", "yes", "on"},
            cleanup_events_max_bytes=int(os.getenv("INFERENCE_ENGINE_CLEANUP_EVENTS_MAX_BYTES", str(5 * 1024 * 1024))),
        )

    def ensure_dirs(self) -> None:
        for path in (self.job_root, self.spool_root, self.artifact_root, self.logs_root):
            path.mkdir(parents=True, exist_ok=True)


def _optional_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    return int(str(value))
