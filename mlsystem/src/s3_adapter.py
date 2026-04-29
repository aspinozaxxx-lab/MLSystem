from __future__ import annotations
import json, os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .io_utils import write_json
from .pipeline_config import PipelineConfig

TARGET_PREFIXES = [
    "images/incoming/",
    "images/kanopus/",
    "layouts/deforest/",
    "datasets/",
    "cache/",
    "models/",
    "predictions/",
    "pseudolabels/",
    "reports/",
    "experiments/",
    "system/",
]

def endpoint_probe(url: str) -> dict[str, Any]:
    try:
        with urlopen(Request(url, method="HEAD"), timeout=5) as response:
            return {"reachable": True, "status": response.status, "server": response.headers.get("Server")}
    except HTTPError as exc:
        return {"reachable": True, "status": exc.code, "server": exc.headers.get("Server"), "note": "HTTP error still proves endpoint is reachable"}
    except URLError as exc:
        return {"reachable": False, "error": str(exc)}
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}

def _credentials_from_env() -> tuple[str | None, str | None, str]:
    access, secret = os.getenv("AWS_ACCESS_KEY_ID"), os.getenv("AWS_SECRET_ACCESS_KEY")
    return (access, secret, "env") if access and secret else (None, None, "missing")

def _credentials_from_mc(alias: str | None) -> tuple[str | None, str | None, str]:
    if not alias:
        return None, None, "missing"
    config_path = Path.home() / ".mc" / "config.json"
    if not config_path.exists():
        return None, None, "missing"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        item = (payload.get("aliases") or {}).get(alias) or {}
        access, secret = item.get("accessKey") or None, item.get("secretKey") or None
        if access and secret:
            return access, secret, f"mc:{alias}"
    except Exception:
        pass
    return None, None, "missing"

def _client(config: PipelineConfig):
    access, secret, source = _credentials_from_env()
    if not access or not secret:
        access, secret, source = _credentials_from_mc(config.s3_alias)
    if not access or not secret:
        return None, source, "No S3 credentials in env or configured mc alias"
    import boto3
    from botocore.config import Config
    client = boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint_url,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"}),
    )
    return client, source, None

def check_s3(config: PipelineConfig, write_test: bool = False) -> dict[str, Any]:
    probe = endpoint_probe(config.s3_endpoint_url)
    client, source, error = _client(config)
    result: dict[str, Any] = {"ok": False, "endpoint_url": config.s3_endpoint_url, "endpoint": probe, "credentials_source": source, "buckets": [], "write_test": {"attempted": False}}
    if error:
        result["error"] = error
        return result
    try:
        result["buckets"] = [b.get("Name") for b in client.list_buckets().get("Buckets", [])]
        result["ok"] = True
        if write_test and config.s3_write_test_enabled and config.s3_test_bucket:
            key = "mlsystem-healthcheck/test-object.txt"
            client.put_object(Bucket=config.s3_test_bucket, Key=key, Body=b"mlsystem healthcheck\n")
            result["write_test"] = {"attempted": True, "bucket": config.s3_test_bucket, "key": key, "ok": True}
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

def _list_objects(client: Any, bucket: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            objects.append(
                {
                    "key": item.get("Key", ""),
                    "size": int(item.get("Size") or 0),
                    "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                    "etag": str(item.get("ETag", "")).strip('"'),
                }
            )
        if not response.get("IsTruncated"):
            return objects
        token = response.get("NextContinuationToken")

def _extension(key: str) -> str:
    name = key.rsplit("/", 1)[-1]
    if "." not in name:
        return "<none>"
    return "." + name.rsplit(".", 1)[-1].lower()

def build_s3_layout_status(config: PipelineConfig) -> dict[str, Any]:
    bucket = config.storage.s3_bucket
    probe = endpoint_probe(config.s3_endpoint_url)
    client, source, error = _client(config)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "active_bucket": bucket,
        "legacy_bucket_with_typo": None,
        "endpoint": config.s3_endpoint_url,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "credentials_source": source,
        "storage": config.storage.model_dump(mode="json"),
        "s3_paths": config.s3_paths.model_dump(mode="json"),
        "endpoint_status": probe,
        "prefixes": {},
        "object_counts": {"objects": 0, "total_size_bytes": 0},
        "extensions": {},
        "test_image_status": {"found": False},
        "layout_status": {"found": False},
        "warnings": [],
    }
    if error:
        payload["warnings"].append(error)
        write_json(config.system_root / "s3_layout.json", payload)
        return payload
    try:
        buckets = [b.get("Name") for b in client.list_buckets().get("Buckets", [])]
        payload["available_buckets"] = buckets
        if "mlsysems" in buckets:
            payload["legacy_bucket_with_typo"] = "mlsysems"
            payload["warnings"].append("Legacy bucket with typo exists: mlsysems. Keep it until migration is explicitly cleaned up.")
        if bucket not in buckets:
            payload["warnings"].append(f"Active bucket is missing: {bucket}")
            write_json(config.system_root / "s3_layout.json", payload)
            return payload

        objects = _list_objects(client, bucket)
        total_size = sum(item["size"] for item in objects)
        payload["object_counts"] = {"objects": len(objects), "total_size_bytes": total_size}
        payload["extensions"] = dict(Counter(_extension(item["key"]) for item in objects if not item["key"].endswith("/")))
        for prefix in TARGET_PREFIXES:
            prefixed = [item for item in objects if item["key"].startswith(prefix)]
            payload["prefixes"][prefix] = {
                "exists": bool(prefixed),
                "object_count": len(prefixed),
                "total_size_bytes": sum(item["size"] for item in prefixed),
            }
            if not prefixed:
                payload["warnings"].append(f"Missing S3 prefix: s3://{bucket}/{prefix}")

        image = next((item for item in objects if item["key"].lower().startswith("images/") and _extension(item["key"]) in {".tif", ".tiff"}), None)
        if image:
            payload["test_image_status"] = {"found": True, **image, "uri": f"s3://{bucket}/{image['key']}"}
        else:
            payload["warnings"].append("No TIFF image found under images/.")

        geojsons = [item for item in objects if item["key"].lower().startswith("layouts/") and _extension(item["key"]) == ".geojson"]
        scenes = [item for item in objects if item["key"].lower().startswith("layouts/") and item["key"].lower().endswith("scenes.txt")]
        payload["layout_status"] = {
            "found": bool(geojsons),
            "geojson_count": len(geojsons),
            "scenes_txt_count": len(scenes),
            "geojson_examples": [f"s3://{bucket}/{item['key']}" for item in geojsons[:5]],
            "scenes_txt_examples": [f"s3://{bucket}/{item['key']}" for item in scenes[:5]],
        }
        if not geojsons:
            payload["warnings"].append("No GeoJSON layout found under layouts/.")
        if not scenes:
            payload["warnings"].append("No scenes.txt found under layouts/.")
    except Exception as exc:
        payload["warnings"].append(f"{type(exc).__name__}: {exc}")
    write_json(config.system_root / "s3_layout.json", payload)
    return payload
