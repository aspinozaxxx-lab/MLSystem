from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from .auth import is_authenticated, login_session, logout_session, redirect_if_unauthorized, require_user, verify_credentials
from .config import FrontendConfig, get_config
from .mlsystem_api import MLSystemApiClient
from .report_builder import build_annotation_report
from .upload_store import UploadValidationError, store_uploads, uploads_diagnostics


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
logger = logging.getLogger("mlsystem.frontend")


def create_app(config: FrontendConfig | None = None) -> FastAPI:
    config = config or get_config()
    app = FastAPI(title="MLSystem Frontend")
    app.state.config = config
    app.add_middleware(
        SessionMiddleware,
        secret_key=config.session_secret,
        max_age=config.session_ttl_seconds,
        same_site="lax",
        https_only=config.secure_cookies,
    )
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "service": "mlsystem-frontend", "time": datetime.now(timezone.utc).isoformat()}

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request) -> HTMLResponse:
        if is_authenticated(request):
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.head("/login")
    def login_head() -> Response:
        return Response(status_code=200)

    @app.head("/")
    def index_head() -> Response:
        return Response(status_code=303, headers={"Location": "/login"})

    @app.post("/login")
    async def login(request: Request, username: str = Form(...), password: str = Form(...)) -> HTMLResponse:
        if verify_credentials(username, password, config):
            login_session(request, username)
            return RedirectResponse("/", status_code=303)
        return templates.TemplateResponse(request, "login.html", {"error": "Неверный логин или пароль"}, status_code=401)

    @app.post("/logout")
    def logout(request: Request) -> RedirectResponse:
        logout_session(request)
        return RedirectResponse("/login", status_code=303)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        redirect = redirect_if_unauthorized(request)
        if redirect:
            return redirect
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "user": request.session.get("mlsystem_user"),
                "rabbitmq_management_url": config.rabbitmq_management_url,
            },
        )

    @app.get("/annotation-check", response_class=HTMLResponse)
    def annotation_check_page(request: Request) -> HTMLResponse:
        redirect = redirect_if_unauthorized(request)
        if redirect:
            return redirect
        return templates.TemplateResponse(
            request,
            "annotation_check.html",
            {
                "defaults": {
                    "images_uri": config.default_images_uri,
                    "layout_uri": config.default_layout_uri,
                    "split_strategy": config.default_split_strategy,
                },
            },
        )

    @app.get("/docs/frontend", response_class=HTMLResponse)
    def frontend_docs(request: Request) -> HTMLResponse:
        redirect = redirect_if_unauthorized(request)
        if redirect:
            return redirect
        path = Path(__file__).resolve().parents[2] / "docs" / "frontend.md"
        text = path.read_text(encoding="utf-8") if path.exists() else "docs/frontend.md is not available in this container."
        return templates.TemplateResponse(request, "docs.html", {"title": "Frontend", "text": text})

    @app.post("/api/annotation-check")
    async def start_annotation_check(
        request: Request,
        _user: str = Depends(require_user),
        annotation_file: UploadFile = File(...),
        scenes_file: UploadFile = File(...),
        title: str = Form(""),
        images_uri: str = Form(""),
        layout_uri: str = Form(""),
        split_strategy: str = Form(""),
        annotation_crs: str = Form(""),
        allow_inferred_annotation_crs: str = Form("true"),
    ) -> JSONResponse:
        run_id = "frontend_annotation_check_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        try:
            uploads = await store_uploads(run_id=run_id, annotation=annotation_file, scenes=scenes_file, config=config)
        except UploadValidationError as exc:
            _write_frontend_status(
                config,
                run_id,
                {
                    "status": "failed",
                    "run_id": run_id,
                    "failed_step": "parse_uploads",
                    "error": str(exc),
                    "jobs": [],
                    "stage_statuses": [],
                },
            )
            return JSONResponse({"status": "failed", "run_id": run_id, "error": str(exc)}, status_code=400)
        upload_info = uploads_diagnostics(uploads, config.s3_bucket)
        logger.info(
            "annotation_check upload run_id=%s annotation=%s annotation_bytes=%s scenes=%s scenes_bytes=%s scene_count=%s scene_preview=%s layout_uri=%s",
            run_id,
            uploads.annotation_file,
            uploads.annotation_size_bytes,
            uploads.scenes_file,
            uploads.scenes_size_bytes,
            uploads.scene_count,
            uploads.scene_preview,
            uploads.layout_uri,
        )
        effective_layout_uri = uploads.layout_uri
        requested_layout_uri = (layout_uri or "").strip()
        if requested_layout_uri and requested_layout_uri != config.default_layout_uri:
            effective_layout_uri = requested_layout_uri
        preprocess: dict[str, Any] = {
            "split_strategy": (split_strategy or config.default_split_strategy).strip() or config.default_split_strategy,
            "allow_inferred_annotation_crs": str(allow_inferred_annotation_crs).strip().lower() not in {"0", "false", "no", "off"},
        }
        if annotation_crs.strip():
            preprocess["annotation_crs"] = annotation_crs.strip()
        payload = {
            "schema_version": 2,
            "experiment_id": run_id,
            "class_name": "annotation_check",
            "task": "annotation_check",
            "images_uri": (images_uri or config.default_images_uri).strip() or config.default_images_uri,
            "layout_uri": effective_layout_uri,
            "scenes_file": uploads.scenes_file,
            "annotation_file": uploads.annotation_file,
            "preprocess": preprocess,
            "mlflow": {"experiment": "mlsystem-annotation-checks"},
            "params": {"title": title.strip(), "source": "frontend"},
        }
        _write_frontend_status(
            config,
            run_id,
            {
                "status": "queued",
                "run_id": run_id,
                "payload": _public_payload(payload),
                "jobs": [],
                "stage_statuses": [],
                "uploaded_files": upload_info,
                "scene_count": uploads.scene_count,
            },
        )
        thread = threading.Thread(target=_run_annotation_check, args=(config, run_id, payload), daemon=True)
        thread.start()
        return JSONResponse({"status": "queued", "run_id": run_id})

    @app.get("/api/annotation-check/{run_id}")
    def annotation_check_status(run_id: str, _user: str = Depends(require_user)) -> dict[str, Any]:
        status = _read_frontend_status(config, run_id)
        report = build_annotation_report(run_id, config.status_root, jobs=status.get("jobs") or [], frontend_status=status)
        if status.get("status") in {"queued", "running", "failed", "succeeded"}:
            report["status"] = status["status"] if status["status"] != "succeeded" else report["status"]
        if status.get("error"):
            report["error"] = status["error"]
        return report

    @app.get("/api/inference-engine/queues")
    def inference_engine_queues(_user: str = Depends(require_user)) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {config.inference_engine_api_token}"} if config.inference_engine_api_token else {}
        request = urllib.request.Request(config.inference_engine_api_url + "/queues", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=8) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        metrics = payload.get("metrics") if isinstance(payload, dict) else []
        ready = sum(int(item.get("messages_ready") or 0) for item in metrics or [])
        unacked = sum(int(item.get("messages_unacked") or 0) for item in metrics or [])
        consumers = sum(int(item.get("consumers") or 0) for item in metrics or [])
        dead = next((item for item in metrics or [] if item.get("name") == "ie.dead_letter"), {})
        return {
            "status": "ok",
            "ready": ready,
            "unacked": unacked,
            "consumers": consumers,
            "dead_letter": int(dead.get("messages_ready") or 0) + int(dead.get("messages_unacked") or 0),
            "queues": metrics or [],
        }

    return app


def _run_annotation_check(config: FrontendConfig, run_id: str, experiment_config: dict[str, Any]) -> None:
    client = MLSystemApiClient(config.api_base_url, config.api_token)
    jobs: list[dict[str, Any]] = []
    try:
        _merge_frontend_status(
            config,
            run_id,
            {
                "status": "running",
                "current_stage": "inventory_scenes",
                "jobs": jobs,
                "stage_statuses": [{"name": "inventory_scenes", "status": "running", "summary": "Starting inventory_scenes"}],
            },
        )
        stage_payload = {
            "experiment_config": experiment_config,
            "airflow_run_id": run_id,
            "status_root": str(config.status_root),
            "source": "frontend",
        }
        logger.info(
            "annotation_check start_stage run_id=%s stage=inventory_scenes api_url=%s payload=%s",
            run_id,
            f"{config.api_base_url}/api/v1/runs/{run_id}/stages/inventory_scenes/start",
            json.dumps(_public_payload(stage_payload), ensure_ascii=False)[:2000],
        )
        inventory_job = client.start_stage(
            run_id,
            "inventory_scenes",
            stage_payload,
        )
        jobs.append({"stage": "inventory_scenes", "job_id": inventory_job["job_id"], "state": inventory_job.get("state")})
        _merge_frontend_status(config, run_id, {"status": "running", "current_stage": "inventory_scenes", "jobs": jobs})
        inventory_status = client.wait_for_job(inventory_job["job_id"])
        jobs[-1]["state"] = inventory_status.get("state")
        if inventory_status.get("state") != "succeeded":
            _merge_frontend_status(
                config,
                run_id,
                {
                    "status": "failed",
                    "failed_step": "inventory_scenes",
                    "jobs": jobs,
                    "stage_statuses": [{"name": "inventory_scenes", "status": "failed", "summary": _job_error_message(inventory_status)}],
                },
            )
            raise RuntimeError(_job_error_message(inventory_status))

        _merge_frontend_status(
            config,
            run_id,
            {
                "status": "running",
                "current_stage": "prepare_dataset",
                "jobs": jobs,
                "stage_statuses": [
                    {"name": "inventory_scenes", "status": "success", "summary": "inventory_scenes completed"},
                    {"name": "prepare_dataset", "status": "running", "summary": "Starting prepare_dataset"},
                ],
            },
        )
        stage_payload = {
            "experiment_config": experiment_config,
            "airflow_run_id": run_id,
            "status_root": str(config.status_root),
            "source": "frontend",
        }
        logger.info(
            "annotation_check start_stage run_id=%s stage=prepare_dataset api_url=%s payload=%s",
            run_id,
            f"{config.api_base_url}/api/v1/runs/{run_id}/stages/prepare_dataset/start",
            json.dumps(_public_payload(stage_payload), ensure_ascii=False)[:2000],
        )
        dataset_job = client.start_stage(
            run_id,
            "prepare_dataset",
            stage_payload,
        )
        jobs.append({"stage": "prepare_dataset", "job_id": dataset_job["job_id"], "state": dataset_job.get("state")})
        _merge_frontend_status(config, run_id, {"status": "running", "current_stage": "prepare_dataset", "jobs": jobs})
        dataset_status = client.wait_for_job(dataset_job["job_id"])
        jobs[-1]["state"] = dataset_status.get("state")
        if dataset_status.get("state") != "succeeded":
            _merge_frontend_status(
                config,
                run_id,
                {
                    "status": "failed",
                    "failed_step": "prepare_dataset",
                    "jobs": jobs,
                    "stage_statuses": [{"name": "prepare_dataset", "status": "failed", "summary": _job_error_message(dataset_status)}],
                },
            )
            raise RuntimeError(_job_error_message(dataset_status))
        _merge_frontend_status(
            config,
            run_id,
            {
                "status": "succeeded",
                "jobs": jobs,
                "stage_statuses": [
                    {"name": "inventory_scenes", "status": "success", "summary": "inventory_scenes completed"},
                    {"name": "prepare_dataset", "status": "success", "summary": "prepare_dataset completed"},
                ],
                "finished_at": time.time(),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("annotation_check failed run_id=%s error=%s", run_id, exc)
        failed_step = _read_frontend_status(config, run_id).get("failed_step") or _read_frontend_status(config, run_id).get("current_stage") or "api_stage_start"
        stage_statuses = _read_frontend_status(config, run_id).get("stage_statuses") or []
        if not jobs and failed_step in {"inventory_scenes", "api_stage_start"}:
            stage_statuses = [{"name": "inventory_scenes", "status": "failed", "summary": str(exc)}]
        _merge_frontend_status(
            config,
            run_id,
            {
                "status": "failed",
                "failed_step": failed_step,
                "jobs": jobs,
                "stage_statuses": stage_statuses,
                "error": str(exc),
                "finished_at": time.time(),
            },
        )


def _job_error_message(status: dict[str, Any]) -> str:
    error = status.get("error") or {}
    if error.get("message"):
        return str(error["message"])
    report = status.get("report") or {}
    errors = report.get("errors") or []
    if errors:
        return "; ".join(str(item) for item in errors[:5])
    return f"MLSystem API job {status.get('job_id')} ended with state={status.get('state')}"


def _public_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in {"token", "password", "secret"}}


def _status_path(config: FrontendConfig, run_id: str) -> Path:
    safe_run_id = "".join(ch for ch in run_id if ch.isalnum() or ch in {"_", "-"})
    return config.upload_root / safe_run_id / "frontend_status.json"


def _write_frontend_status(config: FrontendConfig, run_id: str, payload: dict[str, Any]) -> None:
    path = _status_path(config, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_frontend_status(config: FrontendConfig, run_id: str, updates: dict[str, Any]) -> None:
    current = _read_frontend_status(config, run_id)
    current.update({"run_id": run_id, **updates})
    _write_frontend_status(config, run_id, current)


def _read_frontend_status(config: FrontendConfig, run_id: str) -> dict[str, Any]:
    path = _status_path(config, run_id)
    if not path.exists():
        return {"status": "unknown", "run_id": run_id, "jobs": []}
    return json.loads(path.read_text(encoding="utf-8"))


app = create_app()
