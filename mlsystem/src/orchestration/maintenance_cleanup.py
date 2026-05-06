from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AIRFLOW_STATE_DIR = Path(os.getenv("MLSYSTEM_AIRFLOW_STATE_DIR", "/opt/airflow/mlsystem_runs"))
AIRFLOW_LOG_DIR = Path(os.getenv("MLSYSTEM_AIRFLOW_LOG_DIR", "/opt/airflow/logs"))
MLSYSTEM_CACHE_DIR = Path(os.getenv("MLSYSTEM_CACHE_DIR", "/data/mlsystem/cache"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _file_tree_stats(path: Path) -> tuple[int, int]:
    file_count = 0
    total_size = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total_size += item.stat().st_size
            file_count += 1
        except FileNotFoundError:
            pass
    return file_count, total_size


def _remove_empty_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return removed
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def _write_report(report: dict[str, Any]) -> dict[str, Any]:
    report_dir = AIRFLOW_STATE_DIR / "_maintenance"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"cleanup_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(path)
    return report


def cleanup_runtime_intermediates() -> dict[str, Any]:
    root = AIRFLOW_STATE_DIR.resolve()
    min_age_hours = _safe_float_env("MLSYSTEM_CLEANUP_INTERMEDIATE_MIN_AGE_HOURS", 24.0)
    cutoff = time.time() - min_age_hours * 3600.0
    allowed_names = {"pseudolabel_scene_results", "vectorization_work"}
    report: dict[str, Any] = {
        "task": "cleanup_runtime_intermediates",
        "created_at": _utc_now(),
        "root": str(root),
        "min_age_hours": min_age_hours,
        "deleted_dirs": [],
        "deleted_files": 0,
        "deleted_bytes": 0,
        "warnings": [],
    }
    if not root.exists():
        report["warnings"].append(f"missing root: {root}")
        return _write_report(report)
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        summary_path = run_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig")) if summary_path.exists() else {}
        except Exception:
            summary = {}
        if summary.get("status") == "running":
            continue
        for name in sorted(allowed_names):
            target = (run_dir / name).resolve()
            if not target.exists():
                continue
            if root not in target.parents or target.name not in allowed_names:
                report["warnings"].append(f"skip unsafe target: {target}")
                continue
            if target.stat().st_mtime > cutoff:
                continue
            files, size = _file_tree_stats(target)
            shutil.rmtree(target)
            report["deleted_dirs"].append({"path": str(target), "files": files, "bytes": size})
            report["deleted_files"] += files
            report["deleted_bytes"] += size
    report["deleted_gb"] = round(report["deleted_bytes"] / 1024**3, 3)
    return _write_report(report)


def cleanup_local_cache() -> dict[str, Any]:
    root = MLSYSTEM_CACHE_DIR.resolve()
    retention_hours = _safe_float_env("MLSYSTEM_CLEANUP_CACHE_RETENTION_HOURS", 24.0)
    cutoff = time.time() - retention_hours * 3600.0
    report: dict[str, Any] = {
        "task": "cleanup_local_cache",
        "created_at": _utc_now(),
        "root": str(root),
        "retention_hours": retention_hours,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "deleted_empty_dirs": 0,
        "warnings": [],
    }
    if not root.exists():
        report["warnings"].append(f"missing root: {root}")
        return _write_report(report)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            if stat.st_mtime > cutoff:
                continue
            size = stat.st_size
            path.unlink()
            report["deleted_files"] += 1
            report["deleted_bytes"] += size
        except FileNotFoundError:
            pass
    report["deleted_empty_dirs"] = _remove_empty_dirs(root)
    report["deleted_gb"] = round(report["deleted_bytes"] / 1024**3, 3)
    return _write_report(report)


def cleanup_airflow_logs() -> dict[str, Any]:
    root = AIRFLOW_LOG_DIR.resolve()
    retention_days = _safe_int_env("MLSYSTEM_CLEANUP_LOG_RETENTION_DAYS", 14)
    cutoff = time.time() - retention_days * 86400.0
    report: dict[str, Any] = {
        "task": "cleanup_airflow_logs",
        "created_at": _utc_now(),
        "root": str(root),
        "retention_days": retention_days,
        "deleted_files": 0,
        "deleted_bytes": 0,
        "deleted_empty_dirs": 0,
        "warnings": [],
    }
    if not root.exists():
        report["warnings"].append(f"missing root: {root}")
        return _write_report(report)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            if stat.st_mtime > cutoff:
                continue
            size = stat.st_size
            path.unlink()
            report["deleted_files"] += 1
            report["deleted_bytes"] += size
        except FileNotFoundError:
            pass
    report["deleted_empty_dirs"] = _remove_empty_dirs(root)
    report["deleted_gb"] = round(report["deleted_bytes"] / 1024**3, 3)
    return _write_report(report)
