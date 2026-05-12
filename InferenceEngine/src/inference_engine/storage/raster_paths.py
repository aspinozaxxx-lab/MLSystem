from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def rasterio_path_for_uri(value: str | os.PathLike[str]) -> str:
    """Return the cheapest rasterio-readable path for local, file://, or s3:// URIs."""
    path_text = str(value)
    if path_text.startswith("file://"):
        return path_text[7:]
    if path_text.startswith("s3://"):
        local_path = local_minio_path_for_s3_uri(path_text)
        if local_path is not None:
            return str(local_path)
        bucket_key = path_text[5:]
        bucket, _, key = bucket_key.partition("/")
        return f"/vsis3/{bucket}/{key}"
    return path_text


def local_minio_path_for_s3_uri(uri: str, *, roots: Iterable[str | os.PathLike[str]] | None = None) -> Path | None:
    """Map s3://bucket/key to the mounted MinIO data directory when the object is local."""
    if not str(uri).startswith("s3://"):
        return None
    bucket_key = str(uri)[5:]
    bucket, separator, key = bucket_key.partition("/")
    if not bucket or not separator or not key:
        return None

    for root in _candidate_roots(roots):
        candidate = root / bucket / key
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _candidate_roots(roots: Iterable[str | os.PathLike[str]] | None = None) -> list[Path]:
    candidates: list[Path] = []
    if roots is not None:
        candidates.extend(Path(root) for root in roots if str(root))
    env_value = os.getenv("INFERENCE_ENGINE_LOCAL_S3_ROOT")
    if env_value:
        candidates.extend(Path(item) for item in env_value.split(os.pathsep) if item)
    candidates.append(Path("/data/mlsystem/minio"))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        marker = str(candidate)
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    return unique
