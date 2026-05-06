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
    session_ttl_seconds: int = int(os.getenv("MLSYSTEM_FRONTEND_SESSION_TTL_SECONDS", "28800"))
    secure_cookies: bool = _bool_env(
        "MLSYSTEM_FRONTEND_COOKIE_SECURE",
        _bool_env("MLSYSTEM_FRONTEND_SECURE_COOKIES", False),
    )
    public_base_url: str = os.getenv("MLSYSTEM_FRONTEND_PUBLIC_BASE_URL", "").rstrip("/")
    allowed_hosts: str = os.getenv("MLSYSTEM_FRONTEND_ALLOWED_HOSTS", "")
    api_base_url: str = os.getenv("MLSYSTEM_API_BASE_URL", os.getenv("MLSYSTEM_API_URL", "http://mlsystem-api:8088")).rstrip("/")
    api_token: str | None = os.getenv("MLSYSTEM_API_TOKEN")
    upload_root: Path = Path(os.getenv("MLSYSTEM_FRONTEND_UPLOAD_ROOT", "/data/mlsystem/frontend/uploads"))
    status_root: Path = Path(os.getenv("MLSYSTEM_AIRFLOW_STATUS_ROOT", "/data/mlsystem/airflow/status"))
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
