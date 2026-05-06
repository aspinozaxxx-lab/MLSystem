from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from .config import FrontendConfig


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class UploadValidationError(ValueError):
    pass


@dataclass(frozen=True)
class StoredUploads:
    run_dir: Path
    annotation_path: Path
    scenes_path: Path
    layout_uri: str
    annotation_file: str
    scenes_file: str


def sanitize_filename(name: str, fallback: str) -> str:
    name = Path(name or fallback).name
    cleaned = SAFE_NAME_RE.sub("_", name).strip("._")
    return cleaned or fallback


async def store_uploads(
    *,
    run_id: str,
    annotation: UploadFile,
    scenes: UploadFile,
    config: FrontendConfig,
) -> StoredUploads:
    annotation_name = sanitize_filename(annotation.filename or "", "annotation.geojson")
    scenes_name = sanitize_filename(scenes.filename or "", "scenes.txt")
    if not annotation_name.lower().endswith((".geojson", ".json")):
        raise UploadValidationError("GeoJSON file must have .geojson or .json extension")
    if not scenes_name.lower().endswith(".txt"):
        raise UploadValidationError("Scene list must have .txt extension")

    max_bytes = config.max_upload_mb * 1024 * 1024
    annotation_bytes = await annotation.read()
    scenes_bytes = await scenes.read()
    if not annotation_bytes or not scenes_bytes:
        raise UploadValidationError("Uploaded GeoJSON and scene list must not be empty")
    if len(annotation_bytes) > max_bytes or len(scenes_bytes) > max_bytes:
        raise UploadValidationError(f"Upload exceeds {config.max_upload_mb} MB limit")
    try:
        json.loads(annotation_bytes.decode("utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        raise UploadValidationError(f"GeoJSON is not valid JSON: {exc}") from exc
    if not any(line.strip() and not line.strip().startswith("#") for line in scenes_bytes.decode("utf-8-sig", errors="replace").splitlines()):
        raise UploadValidationError("Scene list must contain at least one non-comment scene row")

    run_dir = config.upload_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = run_dir / annotation_name
    scenes_path = run_dir / scenes_name
    annotation_path.write_bytes(annotation_bytes)
    scenes_path.write_bytes(scenes_bytes)

    _upload_to_s3(config, f"{config.s3_prefix.rstrip('/')}/{run_id}/{annotation_name}", annotation_bytes, "application/geo+json")
    _upload_to_s3(config, f"{config.s3_prefix.rstrip('/')}/{run_id}/{scenes_name}", scenes_bytes, "text/plain")
    return StoredUploads(
        run_dir=run_dir,
        annotation_path=annotation_path,
        scenes_path=scenes_path,
        layout_uri=f"s3://{config.s3_bucket}/{config.s3_prefix.rstrip('/')}/{run_id}/",
        annotation_file=annotation_name,
        scenes_file=scenes_name,
    )


def _upload_to_s3(config: FrontendConfig, key: str, content: bytes, content_type: str) -> None:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint_url,
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )
    client.put_object(Bucket=config.s3_bucket, Key=key, Body=content, ContentType=content_type)

