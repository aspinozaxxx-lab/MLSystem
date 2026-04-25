from __future__ import annotations

import json
import os
import re
import socket
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .io_utils import write_json
from .pipeline_config import PipelineConfig, ensure_storage_layout
from .s3_adapter import _client, _credentials_from_env, _credentials_from_mc

INCOMING_PREFIX = "images/incoming/"
KANOPUS_PREFIX = "images/kanopus/"
MANIFEST_PREFIX = "system/manifests/"
TIFF_SUFFIXES = (".tif", ".tiff")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._") or "unknown"


def _scene_id(key: str) -> str:
    name = Path(key).name
    stem = re.sub(r"\.(tif|tiff)$", "", name, flags=re.IGNORECASE)
    stem = re.sub(r"_cog$", "", stem, flags=re.IGNORECASE)
    return _safe_name(stem)


def _delivery_name(key: str) -> str:
    rel = key[len(INCOMING_PREFIX) :] if key.startswith(INCOMING_PREFIX) else key
    parts = [part for part in rel.split("/") if part]
    if len(parts) <= 1:
        return "default"
    return _safe_name(parts[0])


def _sensor_from_name(key: str) -> str | None:
    name = Path(key).name.upper()
    match = re.match(r"^(KV[0-9I]+)_", name)
    return match.group(1) if match else None


def _list_incoming_tiffs(client: Any, bucket: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": INCOMING_PREFIX}
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            key = item.get("Key", "")
            if key.lower().endswith(TIFF_SUFFIXES):
                rows.append(
                    {
                        "key": key,
                        "size": int(item.get("Size") or 0),
                        "etag": str(item.get("ETag", "")).strip('"'),
                        "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else None,
                    }
                )
        if not response.get("IsTruncated"):
            return rows
        token = response.get("NextContinuationToken")


def _rasterio_env(config: PipelineConfig) -> dict[str, str]:
    access, secret, _source = _credentials_from_env()
    if not access or not secret:
        access, secret, _source = _credentials_from_mc(config.s3_alias)
    endpoint = config.s3_endpoint_url
    parsed = urlparse(endpoint)
    endpoint_host = parsed.netloc or endpoint.replace("http://", "").replace("https://", "")
    https = "YES" if endpoint.startswith("https://") else "NO"
    env = {
        "AWS_ACCESS_KEY_ID": access or "",
        "AWS_SECRET_ACCESS_KEY": secret or "",
        "AWS_S3_ENDPOINT": endpoint_host,
        "AWS_HTTPS": https,
        "AWS_VIRTUAL_HOSTING": "FALSE",
        "AWS_REGION": "us-east-1",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    }
    return {k: v for k, v in env.items() if v}


def _read_raster_metadata(config: PipelineConfig, bucket: str, key: str) -> dict[str, Any]:
    try:
        import rasterio

        path = f"/vsis3/{bucket}/{key}"
        env = _rasterio_env(config)
        old_env = {key: os.environ.get(key) for key in env}
        try:
            os.environ.update(env)
            with rasterio.Env():
                with rasterio.open(path) as ds:
                    return {
                        "read_ok": True,
                        "driver": ds.driver,
                        "crs": str(ds.crs) if ds.crs else None,
                        "width": ds.width,
                        "height": ds.height,
                        "band_count": ds.count,
                        "dtypes": list(ds.dtypes),
                        "nodata": ds.nodata,
                        "bounds": list(ds.bounds),
                    }
        finally:
            for env_key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(env_key, None)
                else:
                    os.environ[env_key] = old_value
    except Exception as exc:
        return {"read_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _put_json(client: Any, bucket: str, key: str, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json; charset=utf-8")


def build_preprocess_inventory(config: PipelineConfig, dry_run: bool = False) -> dict[str, Any]:
    ensure_storage_layout(config)
    bucket = config.storage.s3_bucket
    client, source, error = _client(config)
    started = _utc_now()
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "failed" if error else "ok",
        "dry_run": dry_run,
        "started_at": started,
        "finished_at": None,
        "host": socket.gethostname(),
        "active_bucket": bucket,
        "credentials_source": source,
        "incoming_prefix": f"s3://{bucket}/{INCOMING_PREFIX}",
        "images_manifest_s3": f"s3://{bucket}/{MANIFEST_PREFIX}images_manifest.json",
        "deliveries_manifest_s3": f"s3://{bucket}/{MANIFEST_PREFIX}deliveries_manifest.json",
        "local_images_manifest": str(config.system_root / "images_manifest.json"),
        "local_deliveries_manifest": str(config.system_root / "deliveries_manifest.json"),
        "counts": {"incoming_tiffs": 0, "registered": 0, "failed": 0, "deliveries": 0},
        "warnings": [],
        "error": error,
    }
    if error:
        write_json(config.system_root / "preprocess_status.json", status)
        return status

    images: list[dict[str, Any]] = []
    deliveries: dict[str, dict[str, Any]] = {}
    for item in _list_incoming_tiffs(client, bucket):
        key = item["key"]
        delivery = _delivery_name(key)
        scene_id = _scene_id(key)
        raster = _read_raster_metadata(config, bucket, key)
        scene = {
            "schema_version": 1,
            "scene_id": scene_id,
            "delivery_name": delivery,
            "original_s3_uri": f"s3://{bucket}/{key}",
            "incoming_key": key,
            "file_size": item["size"],
            "etag": item["etag"],
            "last_modified": item["last_modified"],
            "detected_sensor": _sensor_from_name(key),
            "status": "registered" if raster.get("read_ok") else "failed",
            "metadata_s3_uri": f"s3://{bucket}/{KANOPUS_PREFIX}{delivery}/{scene_id}/metadata.json",
            "source_s3_uri": f"s3://{bucket}/{KANOPUS_PREFIX}{delivery}/{scene_id}/source.json",
            "raster": raster,
            "registered_at": _utc_now(),
        }
        if not raster.get("read_ok"):
            scene["error"] = raster.get("error")
        images.append(scene)
        if delivery not in deliveries:
            deliveries[delivery] = {
                "delivery_name": delivery,
                "incoming_prefix": f"s3://{bucket}/{INCOMING_PREFIX}{'' if delivery == 'default' else delivery + '/'}",
                "scene_count": 0,
                "registered": 0,
                "failed": 0,
                "total_size": 0,
                "sensors": {},
                "scenes": [],
            }
        delivery_row = deliveries[delivery]
        delivery_row["scene_count"] += 1
        delivery_row["registered"] += 1 if scene["status"] == "registered" else 0
        delivery_row["failed"] += 1 if scene["status"] == "failed" else 0
        delivery_row["total_size"] += item["size"]
        delivery_row["scenes"].append(scene_id)
        sensor = scene.get("detected_sensor") or "unknown"
        delivery_row.setdefault("_sensor_counter", Counter())[sensor] += 1

    for delivery_row in deliveries.values():
        counter = delivery_row.pop("_sensor_counter", Counter())
        delivery_row["sensors"] = dict(counter)
        delivery_row["scenes"] = sorted(set(delivery_row["scenes"]))

    images.sort(key=lambda row: (row["delivery_name"], row["scene_id"], row["incoming_key"]))
    delivery_list = sorted(deliveries.values(), key=lambda row: row["delivery_name"])
    status["counts"] = {
        "incoming_tiffs": len(images),
        "registered": sum(1 for row in images if row["status"] == "registered"),
        "failed": sum(1 for row in images if row["status"] == "failed"),
        "deliveries": len(delivery_list),
    }
    if any(row["delivery_name"] == "default" for row in images):
        status["warnings"].append("Some incoming TIFF files are flat under images/incoming/. Prefer images/incoming/<delivery_name>/<files>.")

    images_manifest = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "active_bucket": bucket,
        "incoming_prefix": f"s3://{bucket}/{INCOMING_PREFIX}",
        "counts": status["counts"],
        "images": images,
    }
    deliveries_manifest = {
        "schema_version": 1,
        "generated_at": images_manifest["generated_at"],
        "active_bucket": bucket,
        "deliveries": delivery_list,
    }
    if not dry_run:
        write_json(config.system_root / "images_manifest.json", images_manifest)
        write_json(config.system_root / "deliveries_manifest.json", deliveries_manifest)
        _put_json(client, bucket, f"{MANIFEST_PREFIX}images_manifest.json", images_manifest)
        _put_json(client, bucket, f"{MANIFEST_PREFIX}deliveries_manifest.json", deliveries_manifest)
        for scene in images:
            metadata_key = f"{KANOPUS_PREFIX}{scene['delivery_name']}/{scene['scene_id']}/metadata.json"
            source_key = f"{KANOPUS_PREFIX}{scene['delivery_name']}/{scene['scene_id']}/source.json"
            _put_json(client, bucket, metadata_key, scene)
            _put_json(
                client,
                bucket,
                source_key,
                {
                    "schema_version": 1,
                    "scene_id": scene["scene_id"],
                    "delivery_name": scene["delivery_name"],
                    "original_s3_uri": scene["original_s3_uri"],
                    "incoming_key": scene["incoming_key"],
                    "etag": scene["etag"],
                    "file_size": scene["file_size"],
                    "last_modified": scene["last_modified"],
                },
            )

    status["status"] = "ok"
    status["finished_at"] = _utc_now()
    write_json(config.system_root / "preprocess_status.json", status)
    return status


def run_preprocess_forever(config: PipelineConfig, interval: float = 300.0) -> None:
    while True:
        build_preprocess_inventory(config, dry_run=False)
        time.sleep(max(interval, 30.0))
