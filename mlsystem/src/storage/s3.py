from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

from ..pipeline_config import PipelineConfig


def s3_parts(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def credentials_from_mc(config: PipelineConfig) -> tuple[str, str]:
    access = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access and secret:
        return access, secret

    alias = config.s3_alias or "mlplatform"
    mc_config = Path.home() / ".mc" / "config.json"
    payload = json.loads(mc_config.read_text(encoding="utf-8"))
    item = (payload.get("aliases") or {}).get(alias) or {}
    access = item.get("accessKey")
    secret = item.get("secretKey")
    if not access or not secret:
        raise RuntimeError("S3 credentials were not found in env or mc alias")
    return access, secret


def s3_client(config: PipelineConfig):
    import boto3
    from botocore.config import Config

    access, secret = credentials_from_mc(config)
    return boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint_url,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )


def aws_session(config: PipelineConfig):
    import boto3
    from rasterio.session import AWSSession

    access, secret = credentials_from_mc(config)
    endpoint = config.s3_endpoint_url.replace("http://", "").replace("https://", "").rstrip("/")
    session = boto3.Session(
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
    )
    return AWSSession(session, endpoint_url=endpoint, aws_unsigned=False)


def list_s3_objects(config: PipelineConfig, uri: str, suffixes: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    client = s3_client(config)
    bucket, prefix = s3_parts(uri)
    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item.get("Key", "")
            if suffixes and not key.lower().endswith(suffixes):
                continue
            objects.append(
                {
                    "bucket": bucket,
                    "key": key,
                    "name": PurePosixPath(key).name,
                    "size": int(item.get("Size") or 0),
                    "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                    "etag": str(item.get("ETag", "")).strip('"'),
                }
            )
        if not response.get("IsTruncated"):
            return objects
        token = response.get("NextContinuationToken")


def read_s3_text(config: PipelineConfig, uri: str) -> str:
    client = s3_client(config)
    bucket, key = s3_parts(uri)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8-sig")


def read_s3_json(config: PipelineConfig, uri: str) -> Any:
    return json.loads(read_s3_text(config, uri))


def find_layout_files(config: PipelineConfig, layout_uri: str, scenes_file: str, annotation_file: str) -> tuple[str, str]:
    objects = list_s3_objects(config, layout_uri)
    keys = [item["key"] for item in objects]
    bucket, _ = s3_parts(layout_uri)

    if annotation_file and annotation_file != "auto":
        annotation_key = annotation_file
        if not annotation_key.startswith("layouts/"):
            _, layout_prefix = s3_parts(layout_uri)
            annotation_key = layout_prefix.rstrip("/") + "/" + annotation_file
    else:
        geojsons = [key for key in keys if key.lower().endswith(".geojson")]
        if not geojsons:
            raise RuntimeError(f"No GeoJSON annotation found under {layout_uri}")
        annotation_key = sorted(geojsons)[-1]

    scene_candidates = [key for key in keys if PurePosixPath(key).name.lower() == scenes_file.lower()]
    if not scene_candidates:
        raise RuntimeError(f"No {scenes_file} found under {layout_uri}")
    scenes_key = sorted(scene_candidates)[-1]
    return f"s3://{bucket}/{annotation_key}", f"s3://{bucket}/{scenes_key}"
