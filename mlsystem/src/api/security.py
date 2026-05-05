from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

SECRET_KEY_RE = re.compile(r"(password|secret|token|access_key|secret_key|aws_secret_access_key|mlflow_tracking_password|minio_secret_key|s3_secret_key)", re.IGNORECASE)
SECRET_ENV_RE = re.compile(r"(SECRET|TOKEN|PASSWORD)", re.IGNORECASE)
MASK = "***"


def mask_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        masked: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if SECRET_KEY_RE.search(key_text):
                masked[key_text] = MASK
            else:
                masked[key_text] = mask_secrets(item)
        return masked
    if isinstance(value, list):
        return [mask_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_secrets(item) for item in value)
    return value


def mask_text(text: str | None) -> str | None:
    """Mask known secret values from free-form diagnostic text."""
    if text is None:
        return None
    masked = text
    for key, value in os.environ.items():
        if not value or len(value) < 4:
            continue
        if SECRET_ENV_RE.search(key) or SECRET_KEY_RE.search(key):
            masked = masked.replace(value, MASK)
    return masked


def masked_env_snapshot(prefixes: tuple[str, ...] = ("MLSYSTEM_", "MLFLOW_", "MINIO_", "AWS_", "S3_", "RABBITMQ_")) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefixes):
            continue
        snapshot[key] = MASK if SECRET_ENV_RE.search(key) else value
    return snapshot


def configured_api_token() -> str | None:
    token = os.getenv("MLSYSTEM_API_TOKEN")
    return token or None


def verify_token_header(header_value: str | None) -> bool:
    token = configured_api_token()
    if not token:
        return True
    if not header_value:
        return False
    prefix = "Bearer "
    supplied = header_value[len(prefix) :] if header_value.startswith(prefix) else header_value
    return supplied == token
