from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from .pipeline_config import PipelineConfig

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

def check_s3(config: PipelineConfig, write_test: bool = False) -> dict[str, Any]:
    probe = endpoint_probe(config.s3_endpoint_url)
    access, secret, source = _credentials_from_env()
    if not access or not secret:
        access, secret, source = _credentials_from_mc(config.s3_alias)
    result: dict[str, Any] = {"ok": False, "endpoint_url": config.s3_endpoint_url, "endpoint": probe, "credentials_source": source, "buckets": [], "write_test": {"attempted": False}}
    if not access or not secret:
        result["error"] = "No S3 credentials in env or configured mc alias"
        return result
    try:
        import boto3
        from botocore.config import Config
        client = boto3.client("s3", endpoint_url=config.s3_endpoint_url, aws_access_key_id=access, aws_secret_access_key=secret, region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"), config=Config(s3={"addressing_style": "path"}))
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
