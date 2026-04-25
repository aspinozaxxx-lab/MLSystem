from __future__ import annotations
import os, shutil, socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .io_utils import write_json
from .job_queue import list_queue
from .mlflow_adapter import check_mlflow
from .pipeline_config import PipelineConfig, ensure_storage_layout
from .s3_adapter import check_s3

def _meminfo() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                values[key] = float(raw.strip().split()[0]) / 1024 / 1024
    except Exception:
        pass
    return values

def _cpu_status() -> dict[str, Any]:
    load1, load5, load15 = os.getloadavg()
    cpu_count = os.cpu_count() or 1
    return {"cpu_count": cpu_count, "load1": load1, "load5": load5, "load15": load15, "load1_per_cpu": load1 / cpu_count}

def _disk_status(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {"path": str(path), "total_gb": round(usage.total / 1024**3, 3), "used_gb": round(usage.used / 1024**3, 3), "free_gb": round(usage.free / 1024**3, 3), "used_percent": round(usage.used / usage.total * 100, 2)}

def collect_status(config: PipelineConfig, include_services: bool = True) -> dict[str, Any]:
    ensure_storage_layout(config)
    mem = _meminfo(); disk = _disk_status(config.storage_root)
    warnings: list[str] = []
    if disk["free_gb"] < config.disk_warning_free_gb:
        warnings.append(f"Low disk free space: {disk['free_gb']} GB below threshold {config.disk_warning_free_gb} GB")
    if config.cpu_only:
        warnings.append("GPU backend is disabled/not available; running CPU-only")
    payload: dict[str, Any] = {
        "schema_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(),
        "cpu": _cpu_status(),
        "ram": {"total_gb": round(mem.get("MemTotal", 0.0), 3), "available_gb": round(mem.get("MemAvailable", 0.0), 3), "swap_total_gb": round(mem.get("SwapTotal", 0.0), 3), "swap_free_gb": round(mem.get("SwapFree", 0.0), 3)},
        "disk": disk,
        "gpu": {"available": False, "status": "not_available", "max_gpu_train_jobs": config.max_gpu_train_jobs},
        "limits": {"cpu_only": config.cpu_only, "max_cpu_train_jobs": config.max_cpu_train_jobs, "max_cpu_preprocess_jobs": config.max_cpu_preprocess_jobs, "default_workers": config.default_workers},
        "queue": list_queue(config)["counts"], "warnings": warnings,
    }
    if include_services:
        payload["mlflow"] = check_mlflow(config)
        payload["s3"] = check_s3(config, write_test=False)
    write_json(config.system_root / "resource_status.json", payload)
    return payload
