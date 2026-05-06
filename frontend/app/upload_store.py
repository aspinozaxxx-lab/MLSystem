from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    annotation_size_bytes: int
    scenes_size_bytes: int
    scene_count: int
    scene_preview: list[str]
    annotation_s3_key: str
    scenes_s3_key: str


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
    scene_names = parse_scene_names(scenes_bytes)
    if not scene_names:
        raise UploadValidationError("Scene list must contain at least one non-comment scene row")

    run_dir = config.upload_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = run_dir / annotation_name
    scenes_path = run_dir / scenes_name
    annotation_path.write_bytes(annotation_bytes)
    scenes_path.write_bytes(scenes_bytes)

    annotation_s3_key = f"{config.s3_prefix.rstrip('/')}/{run_id}/{annotation_name}"
    scenes_s3_key = f"{config.s3_prefix.rstrip('/')}/{run_id}/{scenes_name}"
    _upload_to_s3(config, annotation_s3_key, annotation_bytes, "application/geo+json")
    _upload_to_s3(config, scenes_s3_key, scenes_bytes, "text/plain")
    return StoredUploads(
        run_dir=run_dir,
        annotation_path=annotation_path,
        scenes_path=scenes_path,
        layout_uri=f"s3://{config.s3_bucket}/{config.s3_prefix.rstrip('/')}/{run_id}/",
        annotation_file=annotation_name,
        scenes_file=scenes_name,
        annotation_size_bytes=len(annotation_bytes),
        scenes_size_bytes=len(scenes_bytes),
        scene_count=len(scene_names),
        scene_preview=scene_names[:5],
        annotation_s3_key=annotation_s3_key,
        scenes_s3_key=scenes_s3_key,
    )


def parse_scene_names(content: bytes | str) -> list[str]:
    text = content.decode("utf-8-sig", errors="replace") if isinstance(content, bytes) else content
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def uploads_diagnostics(uploads: StoredUploads, bucket: str) -> dict[str, Any]:
    return {
        "run_dir": str(uploads.run_dir),
        "annotation_file": uploads.annotation_file,
        "scenes_file": uploads.scenes_file,
        "annotation_size_bytes": uploads.annotation_size_bytes,
        "scenes_size_bytes": uploads.scenes_size_bytes,
        "scene_count": uploads.scene_count,
        "scene_preview": uploads.scene_preview,
        "s3_bucket": bucket,
        "annotation_s3_key": uploads.annotation_s3_key,
        "scenes_s3_key": uploads.scenes_s3_key,
        "layout_uri": uploads.layout_uri,
    }


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
