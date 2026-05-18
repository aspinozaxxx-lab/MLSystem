from __future__ import annotations

from pathlib import Path

from ._loader import load_pipeline_config
from .contracts import MLflowConfig, PipelineConfig, S3PathsConfig, SettingsError, StorageConfig, WebConfig


def load_config(path: str | Path | None = None) -> PipelineConfig:
    return load_pipeline_config(path)


def ensure_storage_layout(config: PipelineConfig) -> None:
    for directory in [
        config.storage_root,
        config.storage_root / "images",
        config.storage_root / "layouts",
        config.jobs_root / "pending",
        config.jobs_root / "running",
        config.jobs_root / "done",
        config.jobs_root / "failed",
        config.storage_root / "cache",
        config.experiments_root,
        config.system_root,
        config.logs_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


__all__ = [
    "MLflowConfig",
    "PipelineConfig",
    "S3PathsConfig",
    "SettingsError",
    "StorageConfig",
    "WebConfig",
    "ensure_storage_layout",
    "load_config",
]
