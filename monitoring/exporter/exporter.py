from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def env(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


AIRFLOW_URL = env("AIRFLOW_BASE_URL", "http://airflow-webserver:8080/airflow")
AIRFLOW_USER = os.getenv("AIRFLOW_ADMIN_USER") or os.getenv("_AIRFLOW_WWW_USER_USERNAME") or "admin"
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_ADMIN_PASSWORD") or os.getenv("_AIRFLOW_WWW_USER_PASSWORD") or ""
INFERENCE_ENGINE_URL = env("INFERENCE_ENGINE_API_URL", "http://inference-engine-api:8095")
INFERENCE_ENGINE_TOKEN = os.getenv("INFERENCE_ENGINE_API_TOKEN")
MLSYSTEM_API_URL = env("MLSYSTEM_API_URL", "http://mlsystem-api:8088")
FRONTEND_URL = env("MLSYSTEM_FRONTEND_INTERNAL_URL", "http://mlsystem-frontend:8090")
MLFLOW_URL = env("MLFLOW_INTERNAL_URL", "http://mlflow:5000/mlflow")
MINIO_URL = env("MINIO_INTERNAL_URL", "http://minio:9000")
TRITON_URL = env("INFERENCE_ENGINE_TRITON_URL", "http://triton:8000")
RABBITMQ_URL = env("INFERENCE_ENGINE_RABBITMQ_MANAGEMENT_URL", "http://rabbitmq:15672/rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_DEFAULT_USER", "mlsystem")
RABBITMQ_PASSWORD = os.getenv("RABBITMQ_DEFAULT_PASS", "")


def http_json(url: str, *, timeout: int = 5, headers: dict[str, str] | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(10 * 1024 * 1024).decode("utf-8", errors="replace")
            try:
                return response.status, json.loads(body)
            except json.JSONDecodeError:
                return response.status, body
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return 0, {"error": f"{type(exc).__name__}: {exc}"}


def service_checks() -> list[tuple[str, str]]:
    return [
        ("frontend", f"{FRONTEND_URL}/health"),
        ("mlsystem_api", f"{MLSYSTEM_API_URL}/health"),
        ("inference_engine", f"{INFERENCE_ENGINE_URL}/health"),
        ("airflow", f"{AIRFLOW_URL}/api/v1/health"),
        ("mlflow", f"{MLFLOW_URL}/health"),
        ("minio", f"{MINIO_URL}/minio/health/live"),
        ("triton", f"{TRITON_URL}/v2/health/ready"),
        ("rabbitmq_management", f"{RABBITMQ_URL}/api/overview"),
    ]


def auth_headers() -> dict[str, str]:
    if not AIRFLOW_PASSWORD:
        return {}
    token = base64.b64encode(f"{AIRFLOW_USER}:{AIRFLOW_PASSWORD}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def rabbitmq_headers() -> dict[str, str]:
    if not RABBITMQ_PASSWORD:
        return {}
    token = base64.b64encode(f"{RABBITMQ_USER}:{RABBITMQ_PASSWORD}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def ie_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {INFERENCE_ENGINE_TOKEN}"} if INFERENCE_ENGINE_TOKEN else {}


def render_metrics() -> str:
    lines = [
        "# HELP mlsystem_monitor_exporter_up Exporter process health.",
        "# TYPE mlsystem_monitor_exporter_up gauge",
        "mlsystem_monitor_exporter_up 1",
    ]
    lines.extend(render_service_metrics())
    lines.extend(render_inference_engine_metrics())
    lines.extend(render_airflow_metrics())
    lines.append("")
    return "\n".join(lines)


def render_service_metrics() -> list[str]:
    lines = [
        "# HELP mlsystem_service_up Service HTTP health probe result.",
        "# TYPE mlsystem_service_up gauge",
        "# HELP mlsystem_service_http_status Service health probe HTTP status.",
        "# TYPE mlsystem_service_http_status gauge",
    ]
    for service, url in service_checks():
        headers = auth_headers() if service == "airflow" else rabbitmq_headers() if service == "rabbitmq_management" else {}
        status, _payload = http_json(url, timeout=5, headers=headers)
        up = 1 if 200 <= status < 400 else 0
        lines.append(f'mlsystem_service_up{{service="{label(service)}"}} {up}')
        lines.append(f'mlsystem_service_http_status{{service="{label(service)}"}} {status}')
    return lines


def render_inference_engine_metrics() -> list[str]:
    status, payload = http_json(f"{INFERENCE_ENGINE_URL}/metrics", timeout=8, headers=ie_headers())
    lines = [
        "# HELP mlsystem_inference_engine_metrics_scrape_up InferenceEngine JSON metrics scrape result.",
        "# TYPE mlsystem_inference_engine_metrics_scrape_up gauge",
        f"mlsystem_inference_engine_metrics_scrape_up {1 if status == 200 else 0}",
    ]
    if status != 200 or not isinstance(payload, dict):
        return lines
    jobs = payload.get("jobs") or {}
    counts: dict[str, int] = {}
    for job in jobs.values():
        state = str((job or {}).get("status") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    lines.append("# TYPE mlsystem_inference_engine_jobs gauge")
    for state, count in sorted(counts.items()):
        lines.append(f'mlsystem_inference_engine_jobs{{status="{label(state)}"}} {count}')
    aggregate = payload.get("aggregate") or {}
    for key in [
        "tiles_total",
        "tiles_done",
        "blocks_total",
        "blocks_done",
        "triton_batches",
        "triton_batch_fill_ratio",
        "triton_request_duration_ms",
        "streaming_overlap_sec",
        "spool_bytes",
    ]:
        lines.append(f'mlsystem_inference_engine_{key} {number(aggregate.get(key))}')
    return lines


def render_airflow_metrics() -> list[str]:
    headers = auth_headers()
    lines = [
        "# HELP airflow_scheduler_healthy Airflow scheduler health from REST API.",
        "# TYPE airflow_scheduler_healthy gauge",
    ]
    status, health = http_json(f"{AIRFLOW_URL}/api/v1/health", headers=headers)
    scheduler_ok = 0
    if status == 200 and isinstance(health, dict):
        scheduler_ok = 1 if (health.get("scheduler") or {}).get("status") == "healthy" else 0
    lines.append(f"airflow_scheduler_healthy {scheduler_ok}")

    dag_id = os.getenv("MLSYSTEM_MONITOR_AIRFLOW_DAG_ID", "mlsystem_experiment_pipeline")
    status, runs = http_json(f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns?limit=5&order_by=-start_date", headers=headers)
    if status != 200 or not isinstance(runs, dict):
        return lines
    dag_runs = runs.get("dag_runs") or []
    lines.extend(
        [
            "# HELP airflow_dag_run_state Latest Airflow DAG run state labels.",
            "# TYPE airflow_dag_run_state gauge",
            "# HELP airflow_latest_dag_run_info Latest Airflow DAG run info.",
            "# TYPE airflow_latest_dag_run_info gauge",
            "# HELP airflow_task_state Airflow task state labels for recent runs.",
            "# TYPE airflow_task_state gauge",
            "# HELP airflow_task_duration_seconds Airflow task duration for recent runs.",
            "# TYPE airflow_task_duration_seconds gauge",
        ]
    )
    for idx, run in enumerate(dag_runs):
        run_id = str(run.get("dag_run_id") or "")
        state = str(run.get("state") or "unknown")
        lines.append(f'airflow_dag_run_state{{dag_id="{label(dag_id)}",run_id="{label(run_id)}",state="{label(state)}"}} 1')
        if idx == 0:
            lines.append(f'airflow_latest_dag_run_info{{dag_id="{label(dag_id)}",run_id="{label(run_id)}",state="{label(state)}"}} 1')
        task_status, tasks = http_json(f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{quote(run_id)}/taskInstances", headers=headers)
        if task_status != 200 or not isinstance(tasks, dict):
            continue
        for task in tasks.get("task_instances") or []:
            task_id = str(task.get("task_id") or "")
            task_state = str(task.get("state") or "unknown")
            duration = number(task.get("duration"))
            lines.append(
                f'airflow_task_state{{dag_id="{label(dag_id)}",run_id="{label(run_id)}",task_id="{label(task_id)}",state="{label(task_state)}"}} 1'
            )
            lines.append(f'airflow_task_duration_seconds{{dag_id="{label(dag_id)}",task_id="{label(task_id)}"}} {duration}')
    return lines


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def label(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b'{"status":"ok","service":"mlsystem-monitor-exporter"}\n'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {self.address_string()} {fmt % args}")


def main() -> None:
    port = int(os.getenv("MLSYSTEM_MONITOR_EXPORTER_PORT", "9200"))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
