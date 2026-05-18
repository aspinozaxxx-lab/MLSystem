from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..settings.contracts import PipelineConfig
from .local_io import write_json


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


def s3_parts(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got {uri}")
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def _is_s3_uri(uri: str) -> bool:
    return str(uri).startswith("s3://")


def _local_path_from_uri(uri: str) -> Path:
    value = str(uri)
    if value.startswith("file://"):
        value = value[7:]
    return Path(value)


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


def endpoint_probe(url: str) -> dict[str, Any]:
    try:
        with urlopen(Request(url, method="HEAD"), timeout=5) as response:
            return {"reachable": True, "status": response.status, "server": response.headers.get("Server")}
    except HTTPError as exc:
        return {
            "reachable": True,
            "status": exc.code,
            "server": exc.headers.get("Server"),
            "note": "HTTP error still proves endpoint is reachable",
        }
    except URLError as exc:
        return {"reachable": False, "error": str(exc)}
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


def _credentials_from_env() -> tuple[str | None, str | None, str]:
    access = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    return (access, secret, "env") if access and secret else (None, None, "missing")


def _credentials_from_mc_alias(alias: str | None) -> tuple[str | None, str | None, str]:
    if not alias:
        return None, None, "missing"
    config_path = Path.home() / ".mc" / "config.json"
    if not config_path.exists():
        return None, None, "missing"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        item = (payload.get("aliases") or {}).get(alias) or {}
        access = item.get("accessKey") or None
        secret = item.get("secretKey") or None
        if access and secret:
            return access, secret, f"mc:{alias}"
    except Exception:
        pass
    return None, None, "missing"


def _client_with_source(config: PipelineConfig) -> tuple[Any, str, str | None]:
    access, secret, source = _credentials_from_env()
    if not access or not secret:
        access, secret, source = _credentials_from_mc_alias(config.s3_alias)
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


def check_s3(config: PipelineConfig, write_test: bool = False) -> dict[str, Any]:
    probe = endpoint_probe(config.s3_endpoint_url)
    client, source, error = _client_with_source(config)
    result: dict[str, Any] = {
        "ok": False,
        "endpoint_url": config.s3_endpoint_url,
        "endpoint": probe,
        "credentials_source": source,
        "buckets": [],
        "write_test": {"attempted": False},
    }
    if error:
        result["error"] = error
        return result
    try:
        result["buckets"] = [bucket.get("Name") for bucket in client.list_buckets().get("Buckets", [])]
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
    client, source, error = _client_with_source(config)
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
        buckets = [bucket_item.get("Name") for bucket_item in client.list_buckets().get("Buckets", [])]
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


def cached_s3_object_path(config: PipelineConfig, bucket: str, key: str) -> Path:
    env_root = os.getenv("MLSYSTEM_S3_CACHE_DIR")
    cache_roots = ([Path(env_root)] if env_root else []) + [Path(root) for root in (config.known_data_roots or [])]
    cache_roots.append(Path("/data/mlsystem/cache"))
    root: Path | None = None
    for candidate in cache_roots:
        candidate_root = candidate / "s3" / bucket
        try:
            candidate_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        root = candidate_root
        break
    if root is None:
        raise RuntimeError(f"No writable S3 cache root found for s3://{bucket}/{key}")
    path = root / key
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    client = s3_client(config)
    try:
        client.download_file(bucket, key, str(tmp_path))
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return path


def raster_path_for_s3_key(config: PipelineConfig, key: str) -> str:
    if str(config.storage.heavy_backend).lower() == "s3":
        if str(os.getenv("MLSYSTEM_DISABLE_S3_FILE_CACHE") or "").lower() in {"1", "true", "yes", "on"}:
            return f"/vsis3/{config.storage.s3_bucket}/{key}"
        cached = cached_s3_object_path(config, config.storage.s3_bucket, key)
        return str(cached)
    return f"/vsis3/{config.storage.s3_bucket}/{key}"


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
    if not _is_s3_uri(uri):
        root = _local_path_from_uri(uri)
        if not root.exists():
            raise FileNotFoundError(f"Local layout path does not exist: {root}")
        normalized_suffixes = tuple(item.lower() for item in suffixes) if suffixes else None
        objects: list[dict[str, Any]] = []
        for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
            if not path.is_file():
                continue
            if normalized_suffixes and not path.name.lower().endswith(normalized_suffixes):
                continue
            rel = path.relative_to(root).as_posix()
            objects.append(
                {
                    "bucket": "local",
                    "key": rel,
                    "name": path.name,
                    "size": int(path.stat().st_size),
                    "last_modified": None,
                    "etag": "",
                    "path": str(path),
                }
            )
        return objects

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
    if not _is_s3_uri(uri):
        return _local_path_from_uri(uri).read_text(encoding="utf-8-sig")
    client = s3_client(config)
    bucket, key = s3_parts(uri)
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return body.decode("utf-8-sig")


def read_s3_json(config: PipelineConfig, uri: str) -> Any:
    return json.loads(read_s3_text(config, uri))


def find_layout_files(config: PipelineConfig, layout_uri: str, scenes_file: str, annotation_file: str) -> tuple[str, str]:
    if not _is_s3_uri(layout_uri):
        root = _local_path_from_uri(layout_uri)
        if not root.exists():
            raise FileNotFoundError(f"Local layout path does not exist: {root}")
        files = [path for path in root.rglob("*") if path.is_file()]
        if annotation_file and annotation_file != "auto":
            annotation_path = root / annotation_file
            if not annotation_path.exists():
                matches = [path for path in files if path.name.lower() == annotation_file.lower()]
                if not matches:
                    raise RuntimeError(f"No {annotation_file} found under {layout_uri}")
                annotation_path = sorted(matches, key=lambda item: str(item).lower())[-1]
        else:
            geojsons = [path for path in files if path.name.lower().endswith(".geojson")]
            if not geojsons:
                raise RuntimeError(f"No GeoJSON annotation found under {layout_uri}")
            annotation_path = sorted(geojsons, key=lambda item: str(item).lower())[-1]

        scene_candidates = [path for path in files if path.name.lower() == scenes_file.lower()]
        if not scene_candidates:
            raise RuntimeError(f"No {scenes_file} found under {layout_uri}")
        scenes_path = sorted(scene_candidates, key=lambda item: str(item).lower())[-1]
        return str(annotation_path), str(scenes_path)

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
