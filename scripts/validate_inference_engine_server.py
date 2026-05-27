from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUN_ID = "a7838f91528a47e1931b685c2ea06686"
DEFAULT_MODEL = "segformer_b2"
STATUS_ROOT = Path("/data/mlsystem/runs")
PSEUDOLABEL_STATUS_ROOT = Path("/data/mlsystem/pseudolabel-runs")
REQUIRED_ARTIFACTS = [
    "accepted.geojson.gz",
    "coverage_report.json",
    "inference_results.json",
    "inference_timing_report.json",
    "postprocess_summary.json",
    "prediction_examples.html",
    "probability_maps_index.json",
    "pseudolabel_scene_results_manifest.json",
    "pseudolabel_summary.json",
    "vectorization_summary.json",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate deployed InferenceEngine on the GPU server")
    parser.add_argument("--api", default="http://127.0.0.1:8095")
    parser.add_argument("--mlsystem-api", default="http://127.0.0.1:8088")
    parser.add_argument("--triton", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", default=os.getenv("INFERENCE_ENGINE_REAL_MANIFEST", "auto"))
    parser.add_argument("--token", default=os.getenv("INFERENCE_ENGINE_API_TOKEN"))
    parser.add_argument("--mlsystem-token", default=os.getenv("MLSYSTEM_API_TOKEN"))
    parser.add_argument("--out", default="/tmp/inference_engine_server_validation.json")
    parser.add_argument("--two-scene-timeout-sec", type=int, default=7200)
    parser.add_argument("--twenty-scene-timeout-sec", type=int, default=14400)
    parser.add_argument("--run-pipeline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-direct-api-stage", action="store_true")
    args = parser.parse_args()

    env = _read_env_file(Path("/etc/mlsystem/gpu-platform.env"))
    ie_token = args.token or env.get("INFERENCE_ENGINE_API_TOKEN")
    mlsystem_token = args.mlsystem_token or env.get("MLSYSTEM_API_TOKEN")
    if not ie_token:
        raise RuntimeError("Missing INFERENCE_ENGINE_API_TOKEN")
    if not mlsystem_token:
        raise RuntimeError("Missing MLSYSTEM_API_TOKEN")

    stale_validation_cleanup = _cancel_stale_validation_jobs(args.api.rstrip("/"), ie_token)
    manifest = _resolve_manifest(args.manifest)
    manifest_scenes = _manifest_scene_count(manifest)
    if manifest_scenes < 20:
        raise RuntimeError(f"Real validation requires an inference_manifest.json with at least 20 scenes; best={manifest} scenes={manifest_scenes}")
    dead_letter_before = _rabbitmq_queue_counts().get("ie.dead_letter", {})
    dead_letter_purge_output = ""
    if _queue_message_total(dead_letter_before):
        dead_letter_purge_output = _purge_rabbitmq_queue("ie.dead_letter")
    health = _safe_get_json(args.api.rstrip("/") + "/health", token=ie_token)
    health_commit = health.get("commit") if isinstance(health, dict) else None
    ready = _safe_get_json(args.api.rstrip("/") + "/ready", token=ie_token)
    openapi = _safe_get_json(args.api.rstrip("/") + "/openapi.json")
    endpoints = _openapi_endpoints(openapi)
    mlsystem_stages = _safe_get_json(args.mlsystem_api.rstrip("/") + "/api/v1/stages")

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "commit": health_commit or _run_text("git -C /opt/mlsystem/repo rev-parse HEAD", check=False).strip(),
        "health": health,
        "ready": ready,
        "openapi_endpoints": endpoints,
        "queues_initial": _safe_get_json(args.api.rstrip("/") + "/queues", token=ie_token),
        "metrics_initial": _safe_get_json(args.api.rstrip("/") + "/metrics", token=ie_token),
        "mlsystem_health": _safe_get_json(args.mlsystem_api.rstrip("/") + "/health"),
        "mlsystem_ready": _safe_get_json(args.mlsystem_api.rstrip("/") + "/ready"),
        "mlsystem_stages": mlsystem_stages,
        "frontend_gateway_initial": _frontend_gateway_checks(env),
        "rabbitmq_management": _rabbitmq_management_overview(env),
        "rabbitmq_management_public_url": env.get("RABBITMQ_MANAGEMENT_PUBLIC_URL") or env.get("FRONTEND_RABBITMQ_MANAGEMENT_URL") or "/rabbitmq/",
        "manifest": str(manifest),
        "manifest_scenes": manifest_scenes,
        "stale_validation_cleanup": stale_validation_cleanup,
        "dead_letter_before": dead_letter_before,
        "dead_letter_purge_output": dead_letter_purge_output,
        "dead_letter_after_purge": _rabbitmq_queue_counts().get("ie.dead_letter", {}),
        "services": _service_snapshot(),
        "triton": _ensure_triton_model(args.triton, DEFAULT_MODEL),
        "synthetic_baseline_metrics": _safe_get_json(args.api.rstrip("/") + "/metrics", token=ie_token),
    }

    if args.run_pipeline:
        two_scene = _run_two_scene_via_pipeline(
            mlsystem_api=args.mlsystem_api.rstrip("/"),
            manifest=manifest,
            mlsystem_token=mlsystem_token,
            ie_token=ie_token,
            timeout_sec=args.two_scene_timeout_sec,
        )
    elif args.run_direct_api_stage:
        two_scene = _run_two_scene_via_mlsystem_api(
            api=args.api.rstrip("/"),
            mlsystem_api=args.mlsystem_api.rstrip("/"),
            manifest=manifest,
            ie_token=ie_token,
            mlsystem_token=mlsystem_token,
            timeout_sec=args.two_scene_timeout_sec,
        )
    else:
        raise RuntimeError("2-scene validation requires --run-pipeline or --run-direct-api-stage")
    summary["two_scene"] = two_scene

    twenty_scene = _run_direct_ie_job(
        api=args.api.rstrip("/"),
        token=ie_token,
        manifest=manifest,
        experiment_id="ie_real_20",
        max_scenes=20,
        timeout_sec=args.twenty_scene_timeout_sec,
        sample_every_sec=20,
        resource={
            "triton_batch_size": 8,
            "batches_ahead": 12,
            "max_preprocess_queue": 1024,
            "max_scenes_inflight": 4,
            "max_blocks_inflight": 32,
        },
    )
    summary["twenty_scene"] = twenty_scene
    summary["mlsystem_stages_after"] = _safe_get_json(args.mlsystem_api.rstrip("/") + "/api/v1/stages")
    summary["frontend_gateway_after"] = _frontend_gateway_checks(env)
    summary["final_queues"] = _safe_get_json(args.api.rstrip("/") + "/queues", token=ie_token)
    summary["final_metrics"] = _safe_get_json(args.api.rstrip("/") + "/metrics", token=ie_token)
    summary["dead_letter"] = _rabbitmq_queues(filter_queue="ie.dead_letter")
    summary["dead_letter_final"] = _rabbitmq_queue_counts().get("ie.dead_letter", {})

    _assert_success(summary)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("SERVER_VALIDATION_SUMMARY_START")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("SERVER_VALIDATION_SUMMARY_END")


def _run_two_scene_via_pipeline(
    *,
    mlsystem_api: str,
    manifest: Path,
    mlsystem_token: str,
    ie_token: str,
    timeout_sec: int,
) -> dict[str, Any]:
    started = int(time.time())
    experiment_id = f"ie_pipeline_real_2_{started}"
    run_id = experiment_id
    run_dir = STATUS_ROOT / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, run_dir / "inference_manifest.json")
    _make_tree_container_writable(run_dir)
    request = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "model_ref": DEFAULT_RUN_ID,
        "images_uri": "s3://mlsystems/images/",
        "layout_uri": "s3://mlsystems/layouts/deforest/",
        "scenes": _manifest_scene_names(manifest, limit=2),
    }
    (run_dir / "pseudolabel_request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    created = _submit_pseudolabel_run(mlsystem_api=mlsystem_api, token=mlsystem_token, request=request)
    samples: list[dict[str, Any]] = []
    deadline = time.time() + timeout_sec
    final: dict[str, Any] = {}
    while time.time() < deadline:
        snapshot = _safe_get_json(mlsystem_api + f"/api/v1/pseudolabel-runs/{run_id}", token=mlsystem_token)
        samples.append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "pseudolabel_run": snapshot,
                "rabbitmq": _rabbitmq_queues(),
                "gpu": _nvidia_smi(),
                "docker_stats": _docker_stats(),
            }
        )
        print(f"Pseudolabel run {run_id} state={snapshot.get('state')} progress={snapshot.get('progress_percent')}", flush=True)
        if snapshot.get("state") in {"succeeded", "failed", "cancelled"}:
            final = snapshot
            break
        time.sleep(20)
    if not final:
        raise TimeoutError(f"Pseudolabel run {run_id} did not finish in {timeout_sec}s")
    if final.get("state") != "succeeded":
        raise RuntimeError(f"Pseudolabel run {run_id} failed: {final}")
    pseudolabel_dir = PSEUDOLABEL_STATUS_ROOT / run_id
    summary_path = pseudolabel_dir / "summary.json"
    artifacts = _check_run_dir_artifacts(pseudolabel_dir, experiment_id)
    summary = _read_json(summary_path, default={}) or final.get("summary") or {}
    ie_job = summary.get("inference_engine_job") or {}
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "status": "success",
        "validation_mode": "inference_pipeline",
        "pseudolabel_run": final,
        "pseudolabel_created": created,
        "run_dir": str(pseudolabel_dir),
        "summary": summary,
        "artifacts_checked": artifacts,
        "inference_engine_job": ie_job,
        "samples": _compact_samples(samples),
        "queues_after": _safe_get_json((os.getenv("INFERENCE_ENGINE_API_URL") or "http://127.0.0.1:8095").rstrip("/") + "/queues", token=ie_token),
        "metrics_after": _safe_get_json((os.getenv("INFERENCE_ENGINE_API_URL") or "http://127.0.0.1:8095").rstrip("/") + "/metrics", token=ie_token),
    }


def _run_two_scene_via_mlsystem_api(
    *,
    api: str,
    mlsystem_api: str,
    manifest: Path,
    ie_token: str,
    mlsystem_token: str,
    timeout_sec: int,
) -> dict[str, Any]:
    return _run_two_scene_via_pipeline(
        mlsystem_api=mlsystem_api,
        manifest=manifest,
        mlsystem_token=mlsystem_token,
        ie_token=ie_token,
        timeout_sec=timeout_sec,
    )


def _run_direct_ie_job(
    *,
    api: str,
    token: str,
    manifest: Path,
    experiment_id: str,
    max_scenes: int,
    timeout_sec: int,
    sample_every_sec: int,
    resource: dict[str, Any],
) -> dict[str, Any]:
    payload = _ie_payload(experiment_id, manifest, max_scenes=max_scenes, resource=resource)
    created = _post_json(api + "/api/v1/jobs", payload, token=token)
    job_id = str(created["job_id"])
    started = time.time()
    samples: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    while True:
        status = _get_json(api + f"/api/v1/jobs/{job_id}", token=token)
        samples.append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "job_status": status.get("status"),
                "job_metrics": status.get("metrics") or {},
                "queues": _safe_get_json(api + "/queues", token=token),
                "aggregate_metrics": _safe_get_json(api + "/metrics", token=token),
                "rabbitmq": _rabbitmq_queues(),
                "gpu": _nvidia_smi(),
                "docker_stats": _docker_stats(),
            }
        )
        print(f"IE job {job_id} status={status.get('status')} metrics={status.get('metrics')}", flush=True)
        if status.get("status") in {"success", "failed", "cancelled"}:
            final = status
            break
        if time.time() - started > timeout_sec:
            raise TimeoutError(f"InferenceEngine job {job_id} did not finish in {timeout_sec}s")
        time.sleep(sample_every_sec)
    events = _safe_get_json(api + f"/api/v1/jobs/{job_id}/events", token=token)
    artifacts = _safe_get_json(api + f"/api/v1/jobs/{job_id}/artifacts", token=token)
    return {
        "job_id": job_id,
        "status": final.get("status"),
        "metrics": final.get("metrics") or {},
        "counters": final.get("counters") or {},
        "artifacts": final.get("artifacts") or {},
        "artifact_api": artifacts,
        "events_count": len(events.get("events") or []) if isinstance(events, dict) else None,
        "stage_events": _stage_event_counts(events),
        "samples": _compact_samples(samples),
    }


def _cancel_stale_validation_jobs(api: str, token: str) -> dict[str, Any]:
    metrics = _safe_get_json(api + "/metrics", token=token)
    jobs = metrics.get("jobs") if isinstance(metrics, dict) else {}
    cancelled: list[dict[str, Any]] = []
    for job_id, row in (jobs or {}).items():
        status = str((row or {}).get("status") or "")
        if status not in {"queued", "running"}:
            continue
        if not (str(job_id).startswith("ie_real_20-") or str(job_id).startswith("ie_real_2_")):
            continue
        cancelled.append(
            {
                "job_id": job_id,
                "status": status,
                "result": _safe_post_json(api + f"/api/v1/jobs/{job_id}/cancel", {}, token=token),
            }
        )
    if cancelled:
        time.sleep(2)
    return {"jobs_seen": len(jobs or {}), "cancelled": cancelled}


def _submit_pseudolabel_run(*, mlsystem_api: str, token: str, request: dict[str, Any]) -> dict[str, Any]:
    return _post_json(mlsystem_api.rstrip("/") + "/api/v1/pseudolabel-runs", request, token=token)


def _ie_payload(experiment_id: str, manifest: Path, *, max_scenes: int, resource: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "inference_manifest": str(manifest),
        "images_uri": "s3://mlsystems/images/",
        "layout_uri": "s3://mlsystems/layouts/deforest/",
        "max_scenes": max_scenes,
        "model": {"mlflow_run_id": DEFAULT_RUN_ID, "architecture": DEFAULT_MODEL, "model_name": DEFAULT_MODEL, "triton_model_name": DEFAULT_MODEL},
        "preprocess": {"patch_size": 1024, "stride": 768, "input_bands": [1, 2, 3, 4]},
        "pseudolabel": {"threshold": 0.5, "core_size_px": 4096, "halo_px": 512, "local_min_area": 0, "final_min_area": 0, "merge_epsilon": 1.0},
        "resource": {
            "triton_batch_size": resource.get("triton_batch_size", 8),
            "batches_ahead": resource.get("batches_ahead", 12),
            "max_preprocess_queue": resource.get("max_preprocess_queue", 1024),
            "max_spool_bytes": resource.get("max_spool_bytes", 30 * 1024 * 1024 * 1024),
            "max_scenes_inflight": resource.get("max_scenes_inflight", 4),
            "max_blocks_inflight": resource.get("max_blocks_inflight", 32),
        },
    }


def _assert_success(summary: dict[str, Any]) -> None:
    two_job = summary["two_scene"]["inference_engine_job"]
    twenty = summary["twenty_scene"]
    required_endpoints = {
        "GET /health",
        "GET /ready",
        "GET /queues",
        "GET /metrics",
        "POST /api/v1/jobs",
        "GET /api/v1/jobs/{job_id}",
        "GET /api/v1/jobs/{job_id}/events",
        "GET /api/v1/jobs/{job_id}/artifacts",
        "POST /api/v1/jobs/{job_id}/cancel",
    }
    missing_endpoints = sorted(required_endpoints - set(summary.get("openapi_endpoints") or []))
    if missing_endpoints:
        raise RuntimeError(f"InferenceEngine OpenAPI is missing endpoints: {missing_endpoints}")
    stages = _first_successful_payload(summary.get("mlsystem_stages_after"), summary.get("mlsystem_stages"))
    main_stages = set(stages.get("pipeline_stages") or [])
    old_stages = {
        "inference_engine_pipeline",
        "prepare_inference_scenes",
        "run_pseudolabel_inference",
        "validate_probability_maps",
        "vectorize_pseudolabel",
        "postprocess_pseudolabel",
        "export_pseudolabel_artifacts",
    }
    if old_stages & main_stages:
        raise RuntimeError(f"mlsystem-api still exposes old pseudolabel stages in pipeline_stages: {old_stages & main_stages}")
    if summary["two_scene"].get("status") != "success":
        raise RuntimeError("2-scene validation failed")
    if twenty.get("status") != "success":
        raise RuntimeError(f"20-scene InferenceEngine validation failed: {twenty}")
    for label, metrics in [("2-scene", two_job.get("metrics") or {}), ("20-scene", twenty.get("metrics") or {})]:
        first = metrics.get("first_block_vectorized_at")
        last = metrics.get("last_tile_inferred_at")
        if not (first and last and float(first) < float(last)):
            raise RuntimeError(f"{label} streaming overlap proof failed: first_block_vectorized_at={first}, last_tile_inferred_at={last}")
    final_dead = summary.get("dead_letter_final") or {}
    if _queue_message_total(final_dead):
        raise RuntimeError(f"dead_letter queue is not empty after validation run: {summary.get('dead_letter')}")


def _first_successful_payload(*payloads: Any) -> dict[str, Any]:
    for payload in payloads:
        if isinstance(payload, dict) and not payload.get("error"):
            return payload
    for payload in payloads:
        if isinstance(payload, dict):
            return payload
    return {}


def _ensure_triton_model(triton_url: str, model: str) -> dict[str, Any]:
    triton_url = triton_url.rstrip("/")
    health = _http_text(triton_url + "/v2/health/ready")
    repository_before = _safe_post_json(triton_url + "/v2/repository/index", {})
    if not _model_ready(triton_url, model):
        _post_json(triton_url + f"/v2/repository/models/{model}/load", {}, token=None, tolerate_http_error=True)
    if not _wait_model_ready(triton_url, model, timeout_sec=60):
        export = _export_model_via_compose(model)
        _post_json(triton_url + f"/v2/repository/models/{model}/load", {}, token=None, tolerate_http_error=True)
        if not _wait_model_ready(triton_url, model, timeout_sec=180):
            raise RuntimeError(f"Triton model {model} is not ready after export: {export}")
    return {
        "health": health,
        "repository_before": repository_before,
        "repository_after": _safe_post_json(triton_url + "/v2/repository/index", {}),
        "model_ready": _model_ready(triton_url, model),
    }


def _export_model_via_compose(model: str) -> str:
    compose = "docker compose --env-file /etc/mlsystem/gpu-platform.env -f /data/mlsystem/platform/docker-compose.yml"
    _ensure_mlsystem_api_deps(compose)
    shell_script = f"""set -e
if [ -f /opt/mlsystem-scripts/export_mlflow_run_to_triton.py ]; then
  python /opt/mlsystem-scripts/export_mlflow_run_to_triton.py --run-id {DEFAULT_RUN_ID} --repository /data/mlsystem/triton/model_repository --triton-model-name {model} --model-name {model} --input-bands 4 --tile-size 1024 --max-batch-size 8 --backend onnx
else
  python /opt/mlsystem/repo/scripts/export_mlflow_run_to_triton.py --run-id {DEFAULT_RUN_ID} --repository /data/mlsystem/triton/model_repository --triton-model-name {model} --model-name {model} --input-bands 4 --tile-size 1024 --max-batch-size 8 --backend onnx
fi"""
    cmd = f"{compose} exec -T mlsystem-api bash -lc {shlex.quote(shell_script)}"
    try:
        return _run_text(cmd, timeout_sec=7200)
    except RuntimeError:
        model_path = f"/data/mlsystem/triton/model_repository/{model}/1/model.onnx"
        existing = _run_text(f"test -f {shlex.quote(model_path)} && echo {shlex.quote(model_path)}", check=False).strip()
        if existing:
            return existing
        raise


def _ensure_mlsystem_api_deps(compose: str) -> None:
    shell_script = """python -c "import importlib.util,sys; mods=('mlflow','boto3','torch','rasterio','shapely','tritonclient','onnx'); sys.exit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)" || python -m pip install ${_PIP_ADDITIONAL_REQUIREMENTS}"""
    _run_text(f"{compose} exec -T mlsystem-api bash -lc {shlex.quote(shell_script)}", timeout_sec=7200)


def _model_ready(triton_url: str, model: str) -> bool:
    try:
        with urllib.request.urlopen(triton_url.rstrip("/") + f"/v2/models/{model}/ready", timeout=10) as response:
            return response.status == 200
    except Exception:
        return False


def _wait_model_ready(triton_url: str, model: str, *, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if _model_ready(triton_url, model):
            return True
        time.sleep(3)
    return False


def _resolve_manifest(value: str) -> Path:
    if value and value != "auto":
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    candidates = []
    listing = _run_text(
        "find /data/mlsystem/runs -name inference_manifest.json -type f -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -200",
        check=False,
    )
    for line in listing.splitlines():
        _, _, path_text = line.partition(" ")
        path = Path(path_text.strip())
        if path.exists():
            candidates.append((path, _manifest_scene_count(path)))
    if not candidates:
        raise FileNotFoundError("No inference_manifest.json found under /data/mlsystem/runs")
    candidates.sort(key=lambda item: (item[1], item[0].stat().st_mtime), reverse=True)
    return candidates[0][0]


def _manifest_scene_count(path: Path) -> int:
    payload = _read_json(path, default={}) or {}
    return len(payload.get("scenes") or payload.get("scene_results") or [])


def _manifest_scene_names(path: Path, *, limit: int) -> list[str]:
    payload = _read_json(path, default={}) or {}
    scenes = payload.get("scenes") or payload.get("scene_results") or []
    names: list[str] = []
    for item in scenes[:limit]:
        if isinstance(item, dict):
            value = item.get("entry") or item.get("name") or item.get("scene") or item.get("scene_id")
        else:
            value = item
        if value:
            names.append(str(value))
    return names


def _check_run_dir_artifacts(run_dir: Path, experiment_id: str) -> dict[str, Any]:
    names = list(REQUIRED_ARTIFACTS) + [f"{experiment_id}.accepted.geojson", "pseudolabel_scenes.txt"]
    present = {name: str(run_dir / name) for name in names if (run_dir / name).exists()}
    missing = [name for name in names if name not in present]
    if missing:
        raise RuntimeError(f"Missing compatibility artifacts in {run_dir}: {missing}")
    return {"present": present, "missing": missing}


def _make_tree_container_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        try:
            mode = path.stat().st_mode
            path.chmod(mode | 0o777 if path.is_dir() else mode | 0o666)
        except OSError:
            continue


def _stage_event_counts(events_payload: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    events = events_payload.get("events") if isinstance(events_payload, dict) else []
    for event in events or []:
        event_type = str(event.get("event_type") or event.get("type") or event.get("stage") or "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def _compact_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(samples) <= 6:
        return samples
    return [samples[0], samples[1], samples[len(samples) // 2], samples[-3], samples[-2], samples[-1]]


def _service_snapshot() -> str:
    return _run_text("docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'", check=False)


def _compose_cmd() -> str:
    return "docker compose --env-file /etc/mlsystem/gpu-platform.env -f /data/mlsystem/platform/docker-compose.yml"


def _openapi_endpoints(openapi: Any) -> list[str]:
    if not isinstance(openapi, dict):
        return []
    endpoints: list[str] = []
    for path, methods in sorted((openapi.get("paths") or {}).items()):
        if not isinstance(methods, dict):
            continue
        for method in sorted(methods):
            if method.lower() in {"get", "post", "put", "patch", "delete"}:
                endpoints.append(f"{method.upper()} {path}")
    endpoints.extend(["GET /openapi.json", "GET /docs", "GET /redoc"])
    return sorted(set(endpoints))


def _rabbitmq_queues(filter_queue: str | None = None) -> str:
    output = _run_text("docker exec mlsystem-gpu-rabbitmq rabbitmqctl list_queues name messages_ready messages_unacknowledged consumers", check=False)
    if not filter_queue:
        return output
    lines = [line for line in output.splitlines() if filter_queue in line or line.startswith("name")]
    return "\n".join(lines)


def _rabbitmq_queue_counts() -> dict[str, dict[str, int]]:
    output = _rabbitmq_queues()
    result: dict[str, dict[str, int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] == "name":
            continue
        try:
            result[parts[0]] = {
                "messages_ready": int(parts[1]),
                "messages_unacknowledged": int(parts[2]),
                "consumers": int(parts[3]),
            }
        except ValueError:
            continue
    return result


def _queue_message_total(row: dict[str, Any]) -> int:
    return int(row.get("messages_ready") or 0) + int(row.get("messages_unacknowledged") or 0)


def _purge_rabbitmq_queue(queue: str) -> str:
    return _run_text(f"docker exec mlsystem-gpu-rabbitmq rabbitmqctl purge_queue {shlex.quote(queue)}", check=False)


def _rabbitmq_management_overview(env: dict[str, str]) -> dict[str, Any]:
    user = env.get("RABBITMQ_DEFAULT_USER")
    password = env.get("RABBITMQ_DEFAULT_PASS")
    if not user or not password:
        return {"status": "skipped", "reason": "RabbitMQ credentials are not available in env file"}
    curl = (
        "curl -fsS "
        + shlex.quote(f"http://127.0.0.1:{env.get('RABBITMQ_MANAGEMENT_PORT') or '15672'}/api/overview")
        + " -u "
        + shlex.quote(f"{user}:{password}")
    )
    output = _run_text(curl, check=False)
    try:
        payload = json.loads(output)
    except Exception:
        return {"status": "failed", "output": output[:1000]}
    return {
        "status": "ok",
        "management_version": payload.get("management_version"),
        "rabbitmq_version": payload.get("rabbitmq_version"),
        "auth_required": True,
    }


def _frontend_gateway_checks(env: dict[str, str]) -> dict[str, Any]:
    user = env.get("MLSYSTEM_FRONTEND_USER")
    password = env.get("MLSYSTEM_FRONTEND_PASSWORD")
    result: dict[str, Any] = {
        "public_entrypoint": "http://127.0.0.1/",
        "without_session": {},
        "with_session": {},
    }
    for path in ["/mlflow/", "/rabbitmq/", "/minio/"]:
        result["without_session"][path] = _curl_status(f"http://127.0.0.1{path}")
    if not user or not password:
        result["with_session"] = {"status": "skipped", "reason": "frontend credentials are not available in env file"}
        return result
    cookie = f"/tmp/mlsystem-frontend-validation-{int(time.time())}.cookie"
    login_cmd = (
        "curl -sS -o /dev/null -w '%{http_code}' "
        f"-c {shlex.quote(cookie)} "
        "--data-urlencode "
        + shlex.quote(f"username={user}")
        + " --data-urlencode "
        + shlex.quote(f"password={password}")
        + " http://127.0.0.1/login"
    )
    result["login_status"] = _run_text(login_cmd, check=False).strip()
    result["proxy_check"] = _curl_status("http://127.0.0.1/auth/proxy-check", cookie=cookie)
    for path in ["/", "/mlflow/", "/rabbitmq/", "/minio/"]:
        result["with_session"][path] = _curl_status(f"http://127.0.0.1{path}", cookie=cookie)
    _run_text(f"rm -f {shlex.quote(cookie)}", check=False)
    return result


def _curl_status(url: str, *, cookie: str | None = None) -> dict[str, Any]:
    cmd = "curl -sS -o /dev/null -w '%{http_code}' "
    if cookie:
        cmd += f"-b {shlex.quote(cookie)} "
    cmd += shlex.quote(url)
    output = _run_text(cmd, check=False).strip()
    return {"url": url, "http_status": output}


def _nvidia_smi() -> str:
    return _run_text("nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits", check=False).strip()


def _docker_stats() -> str:
    return _run_text(
        "docker stats --no-stream --format '{{.Name}}\\t{{.CPUPerc}}\\t{{.MemUsage}}' "
        "mlsystem-gpu-inference-engine-preprocess-worker mlsystem-gpu-inference-engine-triton-worker "
        "mlsystem-gpu-inference-engine-block-worker mlsystem-gpu-triton 2>/dev/null",
        check=False,
    ).strip()


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip("'\"")
    return result


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def _get_json(url: str, *, token: str | None = None) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=_headers(token))
    last_exc: Exception | None = None
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if 500 <= exc.code < 600:
                last_exc = exc
                time.sleep(min(30, 2**attempt))
                continue
            raise
        except (ConnectionError, TimeoutError, OSError, urllib.error.URLError) as exc:
            last_exc = exc
            time.sleep(min(30, 2**attempt))
    assert last_exc is not None
    raise last_exc


def _safe_get_json(url: str, *, token: str | None = None) -> Any:
    try:
        return _get_json(url, token=token)
    except Exception as exc:
        return {"error": str(exc), "url": url}


def _safe_post_json(url: str, payload: dict[str, Any], *, token: str | None = None) -> Any:
    try:
        return _post_json(url, payload, token=token)
    except Exception as exc:
        return {"error": str(exc), "url": url}


def _post_json(url: str, payload: dict[str, Any], *, token: str | None = None, tolerate_http_error: bool = False) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={**_headers(token), "Content-Type": "application/json"})
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            if 500 <= exc.code < 600:
                last_exc = exc
                time.sleep(min(30, 2**attempt))
                continue
            if tolerate_http_error:
                return {"http_error": exc.code, "message": exc.read().decode("utf-8", errors="replace")}
            raise
        except (ConnectionError, TimeoutError, urllib.error.URLError) as exc:
            last_exc = exc
            time.sleep(min(30, 2**attempt))
    assert last_exc is not None
    raise last_exc


def _http_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _run_text(command: str, *, timeout_sec: int = 120, check: bool = True) -> str:
    completed = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_sec)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed rc={completed.returncode}: {command}\n{completed.stdout}")
    return completed.stdout


if __name__ == "__main__":
    main()
