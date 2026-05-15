from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class FrontendConfig:
    host: str = os.getenv("MLSYSTEM_FRONTEND_HOST", "0.0.0.0")
    port: int = int(os.getenv("MLSYSTEM_FRONTEND_PORT", "8090"))
    username: str = os.getenv("MLSYSTEM_FRONTEND_USER", "mluser")
    password: str = os.getenv("MLSYSTEM_FRONTEND_PASSWORD", "qazwsxedc")
    session_secret: str = os.getenv("MLSYSTEM_FRONTEND_SESSION_SECRET", "dev-only-change-me")
    session_cookie_name: str = os.getenv("MLSYSTEM_FRONTEND_SESSION_COOKIE_NAME", "mlsystem_session")
    session_ttl_seconds: int = int(os.getenv("MLSYSTEM_FRONTEND_SESSION_TTL_SECONDS", "28800"))
    secure_cookies: bool = _bool_env(
        "MLSYSTEM_FRONTEND_COOKIE_SECURE",
        _bool_env("MLSYSTEM_FRONTEND_SECURE_COOKIES", False),
    )
    public_base_url: str = os.getenv("MLSYSTEM_FRONTEND_PUBLIC_BASE_URL", "").rstrip("/")
    allowed_hosts: str = os.getenv("MLSYSTEM_FRONTEND_ALLOWED_HOSTS", "")
    api_base_url: str = os.getenv("MLSYSTEM_API_BASE_URL", os.getenv("MLSYSTEM_API_URL", "http://mlsystem-api:8088")).rstrip("/")
    api_token: str | None = os.getenv("MLSYSTEM_API_TOKEN")
    inference_engine_api_url: str = os.getenv("INFERENCE_ENGINE_API_URL", "http://inference-engine-api:8095").rstrip("/")
    inference_engine_api_token: str | None = os.getenv("INFERENCE_ENGINE_API_TOKEN")
    mlflow_ui_url: str = os.getenv("FRONTEND_MLFLOW_UI_URL", "/mlflow/")
    minio_ui_url: str = os.getenv("FRONTEND_MINIO_UI_URL", "/minio-browser/")
    rabbitmq_management_url: str = os.getenv(
        "FRONTEND_RABBITMQ_MANAGEMENT_URL",
        os.getenv("RABBITMQ_MANAGEMENT_PUBLIC_URL", "/rabbitmq/"),
    )
    grafana_url: str = os.getenv("FRONTEND_GRAFANA_URL", "/grafana/")
    prometheus_url: str = os.getenv("FRONTEND_PROMETHEUS_URL", "/prometheus/")
    jupyter_url: str = os.getenv("FRONTEND_JUPYTER_URL", "/jupyter/")
    grafana_main_dashboard_url: str = os.getenv(
        "FRONTEND_GRAFANA_MAIN_DASHBOARD_URL",
        "/grafana/d/mlsystem-overview/mlsystem-overview?orgId=1&kiosk",
    )
    upload_root: Path = Path(os.getenv("MLSYSTEM_FRONTEND_UPLOAD_ROOT", "/data/mlsystem/frontend/uploads"))
    status_root: Path = Path(os.getenv("MLSYSTEM_RUN_ROOT", "/data/mlsystem/runs"))
    training_report_root: Path = Path(os.getenv("MLSYSTEM_FRONTEND_TRAINING_REPORT_ROOT", "/data/mlsystem/frontend/training_report"))
    training_report_refresh_seconds: int = int(os.getenv("MLSYSTEM_FRONTEND_TRAINING_REPORT_REFRESH_SECONDS", "120"))
    training_report_stale_minutes: int = int(os.getenv("MLSYSTEM_FRONTEND_TRAINING_REPORT_STALE_MINUTES", "15"))
    training_report_background_enabled: bool = _bool_env("MLSYSTEM_FRONTEND_TRAINING_REPORT_BACKGROUND_ENABLED", True)
    training_tuning_root: Path = Path(os.getenv("MLSYSTEM_TUNING_ROOT", "/data/mlsystem/tuning"))
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", os.getenv("MLSYSTEM_MLFLOW_TRACKING_URI", "http://mlflow:5000/mlflow")).rstrip("/")
    mlmarkup_path: Path = Path(os.getenv("MLSYSTEM_MLMARKUP_REPO_PATH", os.getenv("MLMARKUP_REPO_PATH", "/data/MLMarkup")))
    default_images_uri: str = os.getenv("MLSYSTEM_FRONTEND_DEFAULT_IMAGES_URI", "s3://mlsystems/images/")
    default_layout_uri: str = os.getenv("MLSYSTEM_FRONTEND_DEFAULT_LAYOUT_URI", "s3://mlsystems/frontend-checks/")
    default_split_strategy: str = os.getenv("MLSYSTEM_FRONTEND_DEFAULT_SPLIT_STRATEGY", "object_balanced")
    max_upload_mb: int = int(os.getenv("MLSYSTEM_FRONTEND_MAX_UPLOAD_MB", "100"))
    s3_bucket: str = os.getenv("MLSYSTEM_BUCKET", "mlsystems")
    s3_prefix: str = os.getenv("MLSYSTEM_FRONTEND_S3_PREFIX", "frontend-checks")
    s3_endpoint_url: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLSYSTEM_S3_ENDPOINT_URL", "http://minio:9000"))
    aws_access_key_id: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_config() -> FrontendConfig:
    return FrontendConfig()
