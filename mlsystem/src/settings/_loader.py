from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .contracts import PipelineConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pipeline.server.yaml"


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
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


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value
