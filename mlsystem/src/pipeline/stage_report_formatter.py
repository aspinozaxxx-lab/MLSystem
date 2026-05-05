from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..api.security import mask_text

DEFAULT_HOST_STATUS_ROOT = "/data/mlsystem/airflow/status"
DEFAULT_CONTAINER_STATUS_ROOT = "/opt/airflow/mlsystem_runs"
MAX_COMPACT_XCOM_BYTES = 10_000


def path_views(
    path: str | Path | None,
    *,
    container_status_root: str | Path | None = None,
    host_status_root: str | Path | None = None,
) -> dict[str, str]:
    """Return operator-facing path variants without changing stored artifacts."""
    if path is None:
        return {}
    text = str(path)
    container_root = str(container_status_root or os.getenv("MLSYSTEM_AIRFLOW_CONTAINER_STATUS_DIR") or DEFAULT_CONTAINER_STATUS_ROOT).rstrip("/\\")
    configured_host_root = host_status_root or os.getenv("MLSYSTEM_AIRFLOW_HOST_STATUS_DIR")
    default_container_selected = container_root.replace("\\", "/").rstrip("/") == DEFAULT_CONTAINER_STATUS_ROOT
    host_root = str(configured_host_root or (DEFAULT_HOST_STATUS_ROOT if default_container_selected else "")).rstrip("/\\")
    views = {"path": text}
    normalized = text.replace("\\", "/")
    normalized_container = container_root.replace("\\", "/").rstrip("/")
    normalized_host = host_root.replace("\\", "/").rstrip("/")
    if host_root and (normalized.startswith(normalized_container + "/") or normalized == normalized_container):
        suffix = normalized[len(normalized_container) :].lstrip("/")
        views["container_path"] = text
        views["host_path"] = f"{host_root}/{suffix}" if suffix else host_root
    elif host_root and (normalized.startswith(normalized_host + "/") or normalized == normalized_host):
        suffix = normalized[len(normalized_host) :].lstrip("/")
        views["host_path"] = text
        views["container_path"] = f"{container_root}/{suffix}" if suffix else container_root
    return views


def format_stage_report(
    report: Mapping[str, Any] | None,
    *,
    stage: str | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
    duration_sec: float | None = None,
    stage_json_path: str | Path | None = None,
    report_path: str | Path | None = None,
    container_status_root: str | Path | None = None,
    host_status_root: str | Path | None = None,
) -> str:
    payload = dict(report or {})
    stage_name = stage or str(payload.get("stage") or payload.get("stage_id") or (payload.get("stage_report") or {}).get("stage_id") or "unknown")
    status = str(payload.get("status") or "unknown")
    duration = duration_sec if duration_sec is not None else payload.get("duration_sec")
    errors = _list(payload.get("errors") or _nested(payload, "stage_report", "errors"))
    warnings = _list(payload.get("warnings") or _nested(payload, "stage_report", "warnings"))
    checks = _list(payload.get("checks") or _nested(payload, "stage_report", "checks"))
    counters = _mapping(payload.get("counters") or _nested(payload, "stage_report", "counters"))
    metrics = _mapping(payload.get("metrics") or _nested(payload, "stage_report", "details", "metrics") or _nested(payload, "details", "metrics"))
    artifacts = _mapping(payload.get("artifacts") or _nested(payload, "stage_report", "artifacts"))
    details = _mapping(payload.get("details") or _nested(payload, "stage_report", "details"))
    input_lineage = _mapping(details.get("input_lineage") or payload.get("input_lineage"))
    smoke_mode = _mapping(details.get("smoke_mode") or payload.get("smoke_mode"))
    resources = _mapping(payload.get("resources"))

    lines = [
        f"# MLSystem stage report: `{stage_name}`",
        "",
        "## Summary",
        f"- stage: `{stage_name}`",
        f"- status: `{status}`",
    ]
    if run_id:
        lines.append(f"- run_id: `{run_id}`")
    if job_id:
        lines.append(f"- job_id: `{job_id}`")
    if duration is not None:
        lines.append(f"- duration_sec: `{duration}`")
    summary = payload.get("summary") or _nested(payload, "stage_report", "summary")
    if summary:
        lines.append(f"- summary: {mask_text(str(summary))}")

    if stage_json_path or payload.get("stage_json_path"):
        _append_path(lines, "stage_json", stage_json_path or payload.get("stage_json_path"), container_status_root, host_status_root)
    if report_path or payload.get("report_path"):
        _append_path(lines, "stage_report", report_path or payload.get("report_path"), container_status_root, host_status_root)

    if smoke_mode or payload.get("is_smoke_synthetic"):
        lines += ["", "## Smoke mode"]
        lines.append(f"- synthetic: `{bool(smoke_mode.get('synthetic', payload.get('is_smoke_synthetic')))}`")
        lines.append("- external S3/dataset/training/inference skipped: `true`")
        lines.append("- this run validates orchestration only, not data processing")
        if payload.get("skip_reason"):
            lines.append(f"- skip_reason: {mask_text(str(payload.get('skip_reason')))}")

    if input_lineage:
        lines += ["", "## Input lineage"]
        lines.append(f"- source: `{input_lineage.get('source') or 'unknown'}`")
        lines.append(f"- inventory matched scenes: `{input_lineage.get('inventory_matched_count')}`")
        lines.append(f"- selected scenes for dataset: `{input_lineage.get('selected_count')}`")
        limit = input_lineage.get("dataset_input_limit")
        lines.append(f"- explicit dataset limit: `{limit if limit is not None else 'none'}`")
        if input_lineage.get("limit_source"):
            lines.append(f"- limit source: `{input_lineage.get('limit_source')}`")
        if input_lineage.get("limit_reason"):
            lines.append(f"- limit reason: {mask_text(str(input_lineage.get('limit_reason')))}")
        lines.append(f"- input invariant: `{input_lineage.get('invariant_status')}`")
        excluded_count = int(input_lineage.get("excluded_count") or 0)
        if excluded_count:
            lines.append(f"- excluded scenes: `{excluded_count}`, see `prepare_dataset_input_audit.txt`")

    lines += ["", "## Checks"]
    if checks:
        for check in checks:
            if isinstance(check, Mapping):
                lines.append(f"- {check.get('name', 'check')} - {str(check.get('status', 'unknown')).upper()}: {mask_text(str(check.get('message', '')))}")
            else:
                lines.append(f"- {mask_text(str(check))}")
    else:
        lines.append("- no checks reported")

    lines += ["", "## Counters"]
    if counters:
        for key in sorted(counters):
            lines.append(f"- {key}: `{_short_value(counters[key])}`")
    else:
        lines.append("- no counters reported")

    lines += ["", "## Metrics"]
    if metrics:
        for key in sorted(metrics):
            lines.append(f"- {key}: `{_short_value(metrics[key])}`")
    else:
        lines.append("- no metrics reported")

    lines += ["", "## Warnings"]
    if warnings:
        for item in warnings[:1000]:
            lines.append(f"- {mask_text(str(item))}")
        if len(warnings) > 1000:
            lines.append(f"- ... {len(warnings) - 1000} more warnings")
    else:
        lines.append("- none")

    lines += ["", "## Errors"]
    if errors:
        for item in errors[:1000]:
            lines.append(f"- {mask_text(str(item))}")
    else:
        lines.append("- none")

    lines += ["", "## Artifacts"]
    if artifacts:
        for name, path in artifacts.items():
            _append_path(lines, str(name), path, container_status_root, host_status_root)
    else:
        lines.append("- no artifacts reported")

    lines += ["", "## Resources"]
    resource_lines = _resource_summary_lines(resources, container_status_root, host_status_root)
    lines.extend(resource_lines or ["- no resource summary reported"])

    if status == "failed" or errors:
        lines += ["", "## Diagnostics"]
        if job_id:
            lines.append(f"- API job: `/data/mlsystem/api/jobs/{job_id}`")
            lines.append(f"- command: `ssh gpu-mlserver \"cat /data/mlsystem/api/jobs/{job_id}/error.json\"`")
        stage_host = path_views(stage_json_path or payload.get("stage_json_path"), container_status_root=container_status_root, host_status_root=host_status_root).get("host_path")
        report_host = path_views(report_path or payload.get("report_path"), container_status_root=container_status_root, host_status_root=host_status_root).get("host_path")
        if stage_host:
            lines.append(f"- command: `ssh gpu-mlserver \"cat {stage_host}\"`")
        if report_host:
            lines.append(f"- command: `ssh gpu-mlserver \"cat {report_host}\"`")
        lines.append("- command: `ssh gpu-mlserver \"docker logs --tail 300 mlsystem-gpu-api\"`")

    return "\n".join(lines) + "\n"


def compact_xcom_summary(
    report: Mapping[str, Any] | None,
    *,
    stage: str,
    run_id: str | None = None,
    job_id: str | None = None,
    duration_sec: float | None = None,
    stage_json_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = dict(report or {})
    errors = _list(payload.get("errors") or _nested(payload, "stage_report", "errors"))
    error_payload = _mapping(payload.get("error"))
    if error_payload and not errors and error_payload.get("message"):
        errors = [str(error_payload["message"])]
    warnings = _list(payload.get("warnings") or _nested(payload, "stage_report", "warnings"))
    counters = _mapping(payload.get("counters") or _nested(payload, "stage_report", "counters"))
    metrics = _mapping(payload.get("metrics") or _nested(payload, "stage_report", "details", "metrics") or _nested(payload, "details", "metrics"))
    summary = payload.get("summary") or _nested(payload, "stage_report", "summary") or (error_payload.get("message") if error_payload else None)
    compact = {
        "stage": stage,
        "status": payload.get("status") or ("failed" if errors else "unknown"),
        "job_id": job_id,
        "run_id": run_id,
        "summary": mask_text(str(summary)) if summary is not None else None,
        "report_path": str(report_path or payload.get("report_path") or ""),
        "stage_json_path": str(stage_json_path or payload.get("stage_json_path") or ""),
        "warnings_count": len(warnings),
        "errors_count": len(errors),
        "duration_sec": duration_sec if duration_sec is not None else payload.get("duration_sec"),
        "key_counters": _compact_counters(counters),
        "key_metrics": _compact_counters(metrics),
    }
    if payload.get("skip_reason"):
        compact["skip_reason"] = mask_text(str(payload.get("skip_reason")))
    if payload.get("is_smoke_synthetic") is not None or _nested(payload, "details", "smoke_mode", "synthetic") is not None:
        compact["is_smoke_synthetic"] = bool(payload.get("is_smoke_synthetic") or _nested(payload, "details", "smoke_mode", "synthetic"))
    urls = _extract_url_fields(payload)
    compact.update(urls)
    encoded = json.dumps(compact, ensure_ascii=False, default=str)
    if len(encoded.encode("utf-8")) > MAX_COMPACT_XCOM_BYTES:
        compact["key_counters"] = {"truncated": True, "counter_count": len(counters)}
        compact["key_metrics"] = {"truncated": True, "metric_count": len(metrics)}
        compact["summary"] = (compact.get("summary") or "")[:1000]
    return compact


def write_stage_report_file(
    report: Mapping[str, Any],
    *,
    path: str | Path,
    stage: str,
    run_id: str | None,
    job_id: str | None = None,
    stage_json_path: str | Path | None = None,
    container_status_root: str | Path | None = None,
    host_status_root: str | Path | None = None,
) -> Path:
    report_path = Path(path)
    text = format_stage_report(
        report,
        stage=stage,
        run_id=run_id,
        job_id=job_id,
        duration_sec=report.get("duration_sec"),
        stage_json_path=stage_json_path,
        report_path=report_path,
        container_status_root=container_status_root,
        host_status_root=host_status_root,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def _append_path(lines: list[str], label: str, value: Any, container_root: str | Path | None, host_root: str | Path | None) -> None:
    if value is None:
        return
    views = path_views(str(value), container_status_root=container_root, host_status_root=host_root)
    lines.append(f"- {label}:")
    if views.get("container_path"):
        lines.append(f"  - container_path: `{views['container_path']}`")
    if views.get("host_path"):
        lines.append(f"  - host_path: `{views['host_path']}`")
    if not views.get("container_path") and not views.get("host_path"):
        lines.append(f"  - path: `{views.get('path')}`")


def _resource_summary_lines(resources: Mapping[str, Any], container_root: str | Path | None, host_root: str | Path | None) -> list[str]:
    if not resources:
        return []
    lines: list[str] = []
    if resources.get("sample_count") is not None:
        lines.append(f"- sample_count: `{resources.get('sample_count')}`")
    if resources.get("monitor"):
        _append_path(lines, "monitor", resources.get("monitor"), container_root, host_root)
    for label in ("before", "after"):
        snap = _mapping(resources.get(label))
        if not snap:
            continue
        cpu = _mapping(snap.get("cpu"))
        mem = _mapping(snap.get("memory"))
        gpu = snap.get("gpu")
        parts = []
        if cpu.get("load1_pct_of_cores") is not None:
            parts.append(f"cpu_load1_pct={cpu.get('load1_pct_of_cores')}")
        if mem.get("used_pct") is not None:
            parts.append(f"mem_used_pct={mem.get('used_pct')}")
        if isinstance(gpu, list) and gpu:
            first = _mapping(gpu[0])
            parts.append(f"gpu={first.get('name')}")
            if first.get("memory_used_mb") is not None:
                parts.append(f"gpu_mem_mb={first.get('memory_used_mb')}/{first.get('memory_total_mb')}")
            if first.get("gpu_util_pct") is not None:
                parts.append(f"gpu_util_pct={first.get('gpu_util_pct')}")
        if parts:
            lines.append(f"- {label}: " + ", ".join(str(part) for part in parts))
    return lines


def _compact_counters(counters: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for idx, key in enumerate(sorted(counters)):
        if idx >= 30:
            compact["truncated"] = True
            compact["counter_count"] = len(counters)
            break
        value = counters[key]
        if isinstance(value, (str, int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list):
            compact[key] = value[:10]
        else:
            compact[key] = _short_value(value)
    return compact


def _extract_url_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    details = _mapping(payload.get("details") or _nested(payload, "stage_report", "details"))
    mlflow = _mapping(payload.get("mlflow") or details.get("mlflow"))
    result: dict[str, Any] = {}
    run_url = (
        payload.get("url_mlflow_run")
        or payload.get("mlflow_run_url")
        or payload.get("run_url")
        or mlflow.get("run_url")
        or mlflow.get("run_url_external")
    )
    experiment_url = (
        payload.get("url_mlflow_experiment")
        or payload.get("mlflow_experiment_url")
        or payload.get("experiment_url")
        or mlflow.get("experiment_url")
        or mlflow.get("experiment_url_external")
    )
    run_id = payload.get("mlflow_run_id") or mlflow.get("run_id")
    final_status = payload.get("mlflow_final_status") or mlflow.get("final_status")
    summary_path = payload.get("summary_path") or payload.get("codex_summary") or payload.get("run_summary")
    if run_url:
        result["url_mlflow_run"] = run_url
    if experiment_url:
        result["url_mlflow_experiment"] = experiment_url
    if run_id:
        result["mlflow_run_id"] = run_id
    if final_status:
        result["mlflow_final_status"] = final_status
    if summary_path:
        result["summary_path"] = summary_path
    return result


def _short_value(value: Any, max_len: int = 500) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value)
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = mask_text(text) or ""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value
