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

class MLflowConfig(BaseModel):
    tracking_uri_internal: str = "http://127.0.0.1:5000"
    tracking_uri_external: str = "http://172.26.12.169:5000"
    default_experiment: str = "mlsystem"

class StorageConfig(BaseModel):
    local_root: Path = Path("/data/mlsystem/storage")
    heavy_backend: str = "s3"
    s3_bucket: str = "mlsystems"
    s3_prefix: str = ""

class S3PathsConfig(BaseModel):
    images: str = "s3://mlsystems/images/"
    layouts: str = "s3://mlsystems/layouts/"
    datasets: str = "s3://mlsystems/datasets/"
    cache: str = "s3://mlsystems/cache/"
    models: str = "s3://mlsystems/models/"
    predictions: str = "s3://mlsystems/predictions/"
    pseudolabels: str = "s3://mlsystems/pseudolabels/"
    reports: str = "s3://mlsystems/reports/"
    experiments: str = "s3://mlsystems/experiments/"
    system: str = "s3://mlsystems/system/"

class PipelineConfig(BaseModel):
    schema_version: int = 1
    project_root: Path
    storage_root: Path
    logs_root: Path
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    s3_paths: S3PathsConfig = Field(default_factory=S3PathsConfig)
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str | None = None
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
    preprocess_interval_sec: int = 300
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
    @property
    def mlflow_tracking_uri_internal(self) -> str:
        return self.mlflow.tracking_uri_internal or self.mlflow_tracking_uri or "http://127.0.0.1:5000"
    @property
    def mlflow_tracking_uri_external(self) -> str:
        return self.mlflow.tracking_uri_external or self.mlflow_tracking_uri_internal
    @property
    def mlflow_default_experiment(self) -> str:
        return self.mlflow.default_experiment or self.mlflow_experiment or "mlsystem"

def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    return value

def load_config(path: str | Path | None = None) -> PipelineConfig:
    config_path = Path(path or os.getenv("MLSYSTEM_PIPELINE_CONFIG") or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise FileNotFoundError(
            f"MLSystem pipeline config not found: {config_path}. "
            "Set MLSYSTEM_PIPELINE_CONFIG or deploy mlsystem/configs/pipeline.server.yaml."
        )
    with config_path.open("r", encoding="utf-8") as fp:
        payload = yaml.safe_load(fp) or {}
    payload = _expand_env(payload)
    if os.getenv("MLFLOW_TRACKING_URI"):
        payload.setdefault("mlflow", {})["tracking_uri_internal"] = os.environ["MLFLOW_TRACKING_URI"]
    if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
        payload["s3_endpoint_url"] = os.environ["MLFLOW_S3_ENDPOINT_URL"]
    if "mlflow_tracking_uri" in payload and "mlflow" not in payload:
        payload["mlflow"] = {"tracking_uri_internal": payload["mlflow_tracking_uri"]}
    if "mlflow_experiment" in payload:
        payload.setdefault("mlflow", {}).setdefault("default_experiment", payload["mlflow_experiment"])
    return PipelineConfig.model_validate(payload)

def ensure_storage_layout(config: PipelineConfig) -> None:
    for directory in [
        config.storage_root, config.storage_root / "images", config.storage_root / "layouts",
        config.jobs_root / "pending", config.jobs_root / "running", config.jobs_root / "done", config.jobs_root / "failed",
        config.storage_root / "cache", config.experiments_root, config.system_root, config.logs_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
