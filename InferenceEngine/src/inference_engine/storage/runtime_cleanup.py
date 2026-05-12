from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import InferenceEngineSettings
from .local_io import write_json


def prune_missing_local_artifacts(artifacts: dict[str, Any]) -> dict[str, str]:
    pruned: dict[str, str] = {}
    for key, value in (artifacts or {}).items():
        if not isinstance(value, str):
            continue
        if value.startswith(("s3://", "http://", "https://")):
            pruned[key] = value
            continue
        path = Path(value)
        if path.exists():
            pruned[key] = value
    return pruned


def cleanup_scene_runtime(job_id: str, scene_id: str, *, settings: InferenceEngineSettings, job_dir: Path | None = None) -> dict[str, Any]:
    """Remove per-scene tile/block intermediates as soon as the scene is merged."""
    if not settings.cleanup_intermediates_enabled:
        return {"enabled": False, "job_id": job_id, "scene_id": scene_id}
    job_dir = job_dir or settings.job_root / job_id
    scene_dir = job_dir / "scenes" / scene_id
    report: dict[str, Any] = {
        "enabled": True,
        "scope": "scene",
        "job_id": job_id,
        "scene_id": scene_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deleted_paths": [],
        "deleted_files": 0,
        "deleted_bytes": 0,
        "warnings": [],
    }
    _delete_tree(settings.spool_root / job_id / scene_id, settings.spool_root, report, reason="scene_spool")
    for name in ("tiles", "blocks"):
        _delete_tree(scene_dir / name, job_dir, report, reason=f"scene_{name}")
    _delete_file(scene_dir / "scene_state.json", job_dir, report, reason="scene_state")
    report["deleted_mb"] = round(float(report["deleted_bytes"]) / 1024**2, 3)
    if scene_dir.exists():
        try:
            write_json(scene_dir / "cleanup.json", report)
        except OSError as exc:
            report["warnings"].append(f"scene cleanup report write failed: {exc}")
    return report


def cleanup_terminal_job_runtime(job_id: str, *, settings: InferenceEngineSettings, job_dir: Path | None = None) -> dict[str, Any]:
    """Remove heavy per-job intermediates after a job reaches a terminal state.

    Final user-facing artifacts are kept. If the request did not provide an
    external run_dir, final artifacts live under job_dir/run_artifacts and are
    also preserved.
    """
    if not settings.cleanup_intermediates_enabled:
        return {"enabled": False, "job_id": job_id}
    job_dir = job_dir or settings.job_root / job_id
    report: dict[str, Any] = {
        "enabled": True,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deleted_paths": [],
        "deleted_files": 0,
        "deleted_bytes": 0,
        "warnings": [],
    }
    _delete_tree(settings.spool_root / job_id, settings.spool_root, report, reason="spool")
    for name in ("scenes",):
        _delete_tree(job_dir / name, job_dir, report, reason=f"job_{name}")
    for name in ("plan.json", "progress.json"):
        _delete_file(job_dir / name, job_dir, report, reason=f"job_{name}")
    _compact_events(job_dir / "events.jsonl", int(settings.cleanup_events_max_bytes), report)
    report["deleted_mb"] = round(float(report["deleted_bytes"]) / 1024**2, 3)
    if job_dir.exists():
        try:
            write_json(job_dir / "cleanup.json", report)
        except OSError as exc:
            report["warnings"].append(f"cleanup report write failed: {exc}")
    return report


def _delete_tree(path: Path, root: Path, report: dict[str, Any], *, reason: str) -> None:
    try:
        target = path.resolve()
        base = root.resolve()
    except OSError as exc:
        report["warnings"].append(f"resolve failed for {path}: {exc}")
        return
    if not target.exists():
        return
    if target == base or base not in target.parents:
        report["warnings"].append(f"skip unsafe delete: {target}")
        return
    files, size = _tree_stats(target)
    try:
        shutil.rmtree(target)
    except OSError as exc:
        report["warnings"].append(f"delete failed for {target}: {exc}")
        return
    report["deleted_paths"].append({"path": str(target), "reason": reason, "files": files, "bytes": size})
    report["deleted_files"] += files
    report["deleted_bytes"] += size


def _delete_file(path: Path, root: Path, report: dict[str, Any], *, reason: str) -> None:
    try:
        target = path.resolve()
        base = root.resolve()
        stat = target.stat()
    except FileNotFoundError:
        return
    except OSError as exc:
        report["warnings"].append(f"stat failed for {path}: {exc}")
        return
    if target == base or base not in target.parents:
        report["warnings"].append(f"skip unsafe delete: {target}")
        return
    try:
        target.unlink()
    except OSError as exc:
        report["warnings"].append(f"delete failed for {target}: {exc}")
        return
    report["deleted_paths"].append({"path": str(target), "reason": reason, "files": 1, "bytes": stat.st_size})
    report["deleted_files"] += 1
    report["deleted_bytes"] += stat.st_size


def _tree_stats(path: Path) -> tuple[int, int]:
    if path.is_file():
        try:
            return 1, path.stat().st_size
        except OSError:
            return 0, 0
    files = 0
    size = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            size += item.stat().st_size
            files += 1
        except OSError:
            pass
    return files, size


def _compact_events(path: Path, max_bytes: int, report: dict[str, Any]) -> None:
    if max_bytes <= 0:
        _delete_file(path, path.parent, report, reason="events_disabled")
        return
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return
    except OSError as exc:
        report["warnings"].append(f"events stat failed for {path}: {exc}")
        return
    if size <= max_bytes:
        return
    keep_bytes = max(1024, max_bytes)
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, size - keep_bytes))
            tail = handle.read()
        if size > keep_bytes:
            tail = tail.splitlines()[1:]
            tail_text = b"\n".join(tail).decode("utf-8", errors="replace")
        else:
            tail_text = tail.decode("utf-8", errors="replace")
        compacted = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ts_unix": time.time(),
            "job_id": path.parent.name,
            "type": "events.compacted",
            "payload": {"original_bytes": size, "kept_tail_bytes": keep_bytes},
        }
        path.write_text(json.dumps(compacted, ensure_ascii=False, sort_keys=True) + "\n" + tail_text, encoding="utf-8")
    except OSError as exc:
        report["warnings"].append(f"events compact failed for {path}: {exc}")
        return
    report["deleted_paths"].append({"path": str(path), "reason": "events_compacted", "files": 0, "bytes": size - path.stat().st_size})
    report["deleted_bytes"] += max(0, size - path.stat().st_size)
