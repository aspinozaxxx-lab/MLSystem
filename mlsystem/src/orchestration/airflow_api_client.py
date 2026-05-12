from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from ..api.security import mask_secrets, mask_text
from ..pipeline.stage_report_formatter import compact_xcom_summary, format_stage_report


class AirflowApiStageError(RuntimeError):
    pass


class MLSystemApiClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("MLSYSTEM_API_URL") or "http://mlsystem-api:8088").rstrip("/")
        self.token = token if token is not None else os.getenv("MLSYSTEM_API_TOKEN")

    def start_stage(self, run_id: str, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/runs/{run_id}/stages/{stage}/start", payload)

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/jobs/{job_id}")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(mask_secrets(payload), ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=body, method=method)
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        attempts = max(1, int(os.getenv("MLSYSTEM_API_HTTP_RETRIES", "12")))
        retry_delay_sec = max(0.1, float(os.getenv("MLSYSTEM_API_HTTP_RETRY_DELAY_SEC", "5")))
        timeout_sec = float(os.getenv("MLSYSTEM_API_HTTP_TIMEOUT_SEC", "30"))
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace")
                raise AirflowApiStageError(f"MLSystem API HTTP {exc.code}: {text}") from exc
            except urllib.error.URLError as exc:
                if attempt >= attempts:
                    raise AirflowApiStageError(f"MLSystem API request failed after {attempts} attempts: {exc}") from exc
                time.sleep(retry_delay_sec)
        raise AirflowApiStageError("MLSystem API request failed unexpectedly")


def run_stage_via_api(stage: str, dag_run_conf: dict[str, Any], airflow_run_id: str, state_dir: Path | str) -> dict[str, Any]:
    client = MLSystemApiClient()
    poll_sec = float(os.getenv("MLSYSTEM_AIRFLOW_API_POLL_SEC", "10"))
    no_progress_timeout = float(os.getenv("MLSYSTEM_AIRFLOW_API_NO_PROGRESS_TIMEOUT_SEC", "3600"))
    payload = {
        "experiment_config": dag_run_conf or {},
        "airflow_run_id": airflow_run_id,
        "status_root": str(state_dir),
        "source": "airflow",
    }
    started = client.start_stage(airflow_run_id, stage, payload)
    job_id = started["job_id"]
    print(f"MLSystem API stage started: stage={stage} job_id={job_id} state={started.get('state')}", flush=True)
    last_state = None
    last_change = time.time()
    while True:
        status = client.job_status(job_id)
        state = status.get("state")
        if state != last_state:
            print(f"MLSystem API job status: stage={stage} job_id={job_id} state={state}", flush=True)
            last_state = state
            last_change = time.time()
        if state == "succeeded":
            report = status.get("report") or {}
            _print_human_report(stage, airflow_run_id, job_id, status, report)
            summary = compact_xcom_summary(
                report,
                stage=stage,
                run_id=airflow_run_id,
                job_id=job_id,
                duration_sec=status.get("duration_sec"),
                stage_json_path=report.get("stage_json_path"),
                report_path=report.get("report_path"),
            )
            print(f"MLSystem API stage succeeded: stage={stage} job_id={job_id} summary={summary.get('summary')}", flush=True)
            return summary
        if state in {"failed", "cancelled", "timed_out"}:
            error = status.get("error") or {}
            message = mask_text(str(error.get("message"))) if error.get("message") is not None else None
            traceback_tail = mask_text(str(error.get("traceback_tail"))) if error.get("traceback_tail") is not None else None
            report = status.get("report") or {}
            _print_human_report(stage, airflow_run_id, job_id, status, report)
            summary = compact_xcom_summary(
                report,
                stage=stage,
                run_id=airflow_run_id,
                job_id=job_id,
                duration_sec=status.get("duration_sec"),
                stage_json_path=report.get("stage_json_path"),
                report_path=report.get("report_path"),
            )
            print(
                "MLSystem API stage failed: "
                f"stage={stage} job_id={job_id} state={state} "
                f"message={message} report_path={summary.get('report_path')} "
                f"stage_json_path={summary.get('stage_json_path')} traceback_tail={traceback_tail}",
                flush=True,
            )
            raise AirflowApiStageError(
                f"MLSystem stage {stage} failed via API job {job_id}: {message}; "
                f"report_path={summary.get('report_path')}; stage_json_path={summary.get('stage_json_path')}"
            )
        if time.time() - last_change > no_progress_timeout:
            raise AirflowApiStageError(f"MLSystem API job {job_id} made no state progress for {no_progress_timeout} sec")
        time.sleep(poll_sec)


def _print_human_report(stage: str, run_id: str, job_id: str, status: dict[str, Any], report: dict[str, Any]) -> None:
    text = format_stage_report(
        report,
        stage=stage,
        run_id=run_id,
        job_id=job_id,
        duration_sec=status.get("duration_sec"),
        stage_json_path=report.get("stage_json_path"),
        report_path=report.get("report_path"),
    )
    print(text, flush=True)
