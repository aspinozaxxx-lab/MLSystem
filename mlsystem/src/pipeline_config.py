from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "pipeline.server.yaml"

class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8010

class PipelineConfig(BaseModel):
    schema_version: int = 1
    project_root: Path
    storage_root: Path
    logs_root: Path
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "mlsystem-mvp"
    s3_endpoint_url: str = "http://127.0.0.1:9000"
    s3_alias: str | None = "mlplatform"
    artifact_bucket: str = "mlflow-artifacts"
    dataset_cache_bucket: str | None = "ml-datasets"
    s3_write_test_enabled: bool = False
    s3_test_bucket: str | None = None
    cpu_only: bool = True
    max_gpu_train_jobs: int = 0
    max_cpu_train_jobs: int = 1
    max_cpu_preprocess_jobs: int = 1
    default_workers: int = 0
    known_data_roots: list[Path] = Field(default_factory=list)
    max_inventory_files: int = 2000
    max_inventory_depth: int = 5
    disk_warning_free_gb: float = 20.0
    recent_jobs_limit: int = 20
    web: WebConfig = Field(default_factory=WebConfig)

    @property
    def jobs_root(self) -> Path:
        return self.storage_root / "jobs"
    @property
    def system_root(self) -> Path:
        return self.storage_root / "system"
    @property
    def experiments_root(self) -> Path:
        return self.storage_root / "experiments"

def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value

def load_config(path: str | Path | None = None) -> PipelineConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    payload = _expand_env(payload)
    if os.getenv("MLFLOW_TRACKING_URI"):
        payload["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        payload["s3_endpoint_url"] = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    return PipelineConfig.model_validate(payload)

def ensure_storage_layout(config: PipelineConfig) -> None:
    for directory in [
        config.storage_root, config.storage_root / "images", config.storage_root / "layouts",
        config.jobs_root / "pending", config.jobs_root / "running", config.jobs_root / "done", config.jobs_root / "failed",
        config.storage_root / "cache", config.experiments_root, config.system_root, config.logs_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
