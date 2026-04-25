from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from .codex_summary import build_codex_summary
from .io_utils import read_json
from .job_queue import list_queue
from .mlflow_adapter import check_mlflow
from .pipeline_config import load_config, ensure_storage_layout
from .preprocess_inventory import build_preprocess_inventory
from .resource_manager import collect_status
from .s3_adapter import check_s3

config = load_config()
ensure_storage_layout(config)
app = FastAPI(title="MLSystem MVP", version="0.1.0")

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    queue = list_queue(config)["counts"]
    resources = read_json(config.system_root / "resource_status.json", default={}) or {}
    preprocess = read_json(config.system_root / "preprocess_status.json", default={}) or {}
    images_manifest = read_json(config.system_root / "images_manifest.json", default={}) or {}
    deliveries = (read_json(config.system_root / "deliveries_manifest.json", default={}) or {}).get("deliveries", [])
    mlflow = resources.get("mlflow", {})
    s3 = resources.get("s3", {})
    counts = images_manifest.get("counts", {})
    recent_deliveries = ", ".join([row.get("delivery_name", "") for row in deliveries[:5]]) or "none"
    return f"""
    <html><head><title>MLSystem MVP</title></head><body>
      <h1>MLSystem MVP</h1>
      <p>Storage: <code>{config.storage_root}</code></p>
      <p>Queue: <code>{json.dumps(queue)}</code></p>
      <p>Images: incoming <code>{counts.get('incoming_tiffs', 0)}</code>, registered <code>{counts.get('registered', 0)}</code>, failed <code>{counts.get('failed', 0)}</code></p>
      <p>Preprocess: <code>{preprocess.get('status')}</code>, deliveries: <code>{recent_deliveries}</code></p>
      <p>MLflow: <code>{mlflow.get('ok')}</code> <a href="{config.mlflow_tracking_uri_external}">{config.mlflow_tracking_uri_external}</a></p>
      <p>S3: <code>{s3.get('ok')}</code> endpoint <code>{config.s3_endpoint_url}</code></p>
      <ul>
        <li><a href="/api/state">/api/state</a></li><li><a href="/api/resources">/api/resources</a></li>
        <li><a href="/api/jobs">/api/jobs</a></li><li><a href="/api/codex-summary">/api/codex-summary</a></li>
        <li><a href="/api/mlflow">/api/mlflow</a></li><li><a href="/api/s3">/api/s3</a></li>
        <li><a href="/api/preprocess">/api/preprocess</a></li><li><a href="/api/images">/api/images</a></li>
        <li><a href="/api/deliveries">/api/deliveries</a></li><li><a href="/api/queue">/api/queue</a></li>
        <li><a href="/api/mlflow-queue">/api/mlflow-queue</a></li>
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

@app.get("/api/queue")
def api_queue() -> dict:
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

@app.get("/api/preprocess")
def api_preprocess() -> dict:
    return read_json(config.system_root / "preprocess_status.json", default={"status": "not_run"})

@app.post("/api/preprocess/run")
def api_preprocess_run() -> dict:
    return build_preprocess_inventory(config, dry_run=False)

@app.get("/api/images")
def api_images() -> dict:
    return read_json(config.system_root / "images_manifest.json", default={"images": [], "counts": {}})

@app.get("/api/deliveries")
def api_deliveries() -> dict:
    return read_json(config.system_root / "deliveries_manifest.json", default={"deliveries": []})

@app.get("/api/mlflow-queue")
def api_mlflow_queue() -> dict:
    queue = list_queue(config)
    rows = []
    for state, jobs in queue.get("jobs", {}).items():
        for job in jobs:
            meta = job.get("metadata") or {}
            mlflow = meta.get("mlflow") or {}
            if state != "pending":
                detail = read_json(Path(job["path"]) / "claim.json", default={}) or {}
                mlflow = detail.get("mlflow") or mlflow
            rows.append({"state": state, "name": job.get("name"), "mlflow": mlflow})
    return {"jobs": rows}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web_app:app", host=config.web.host, port=config.web.port, reload=False)
