from __future__ import annotations
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from .codex_summary import build_codex_summary
from .io_utils import read_json
from .job_queue import list_queue
from .mlflow_adapter import check_mlflow
from .pipeline_config import load_config, ensure_storage_layout
from .resource_manager import collect_status
from .s3_adapter import check_s3

config = load_config()
ensure_storage_layout(config)
app = FastAPI(title="MLSystem MVP", version="0.1.0")

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    queue = list_queue(config)["counts"]
    resources = read_json(config.system_root / "resource_status.json", default={}) or {}
    mlflow = resources.get("mlflow", {})
    s3 = resources.get("s3", {})
    return f"""
    <html><head><title>MLSystem MVP</title></head><body>
      <h1>MLSystem MVP</h1>
      <p>Storage: <code>{config.storage_root}</code></p>
      <p>Queue: <code>{json.dumps(queue)}</code></p>
      <p>MLflow: <code>{mlflow.get('ok')}</code> <a href="{config.mlflow_tracking_uri_external}">{config.mlflow_tracking_uri_external}</a></p>
      <p>S3: <code>{s3.get('ok')}</code> endpoint <code>{config.s3_endpoint_url}</code></p>
      <ul>
        <li><a href="/api/state">/api/state</a></li><li><a href="/api/resources">/api/resources</a></li>
        <li><a href="/api/jobs">/api/jobs</a></li><li><a href="/api/codex-summary">/api/codex-summary</a></li>
        <li><a href="/api/mlflow">/api/mlflow</a></li><li><a href="/api/s3">/api/s3</a></li>
      </ul>
    </body></html>
    """

@app.get("/api/state")
def api_state() -> dict:
    return {"storage_root": str(config.storage_root), "queue": list_queue(config)["counts"], "cpu_only": config.cpu_only}

@app.get("/api/resources")
def api_resources() -> dict:
    return collect_status(config, include_services=True)

@app.get("/api/jobs")
def api_jobs() -> dict:
    return list_queue(config)

@app.get("/api/codex-summary")
def api_codex_summary() -> dict:
    return build_codex_summary(config)

@app.get("/api/mlflow")
def api_mlflow() -> dict:
    return check_mlflow(config)

@app.get("/api/s3")
def api_s3() -> dict:
    return check_s3(config, write_test=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web_app:app", host=config.web.host, port=config.web.port, reload=False)
