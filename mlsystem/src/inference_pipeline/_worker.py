from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Callable

from ._inference_engine_client import InferenceEngineClient
from ._payload import DEFAULT_POLL_SEC, DEFAULT_TIMEOUT_SEC, build_inference_engine_payload
from ._run_store import PseudolabelRunStore, utc_now
from ._summary import build_final_summary


ClientFactory = Callable[[], Any]


def run_worker(run_id: str, run_root: str | Path | None = None, *, client_factory: ClientFactory | None = None) -> None:
    store = PseudolabelRunStore(run_root)
    request = store.read_request(run_id)
    started = time.time()
    store.update_run(run_id, state="running", progress_percent=5, started_at=utc_now())
    store.append_log(run_id, "worker started")
    try:
        payload = build_inference_engine_payload(request, run_id=run_id, run_dir=store.run_dir(run_id))
        store.write_json(store.run_dir(run_id) / "inference_engine_payload.json", payload)
        store.update_run(run_id, progress_percent=15)
        if request.dry_run:
            final_state = {"status": "success", "metrics": {}, "dry_run": True}
            job_id = "dry-run"
            artifacts = {"inference_engine_payload.json": str(store.run_dir(run_id) / "inference_engine_payload.json")}
            warnings: list[str] = []
            store.append_log(run_id, "dry_run=true; InferenceEngine job was not submitted")
        else:
            client = client_factory() if client_factory is not None else InferenceEngineClient()
            created = client.create_job(payload)
            job_id = str(created.get("job_id") or created.get("id") or "")
            if not job_id:
                raise RuntimeError(f"InferenceEngine create_job response did not include job_id: {created}")
            store.update_run(run_id, progress_percent=30, inference_engine_job_id=job_id)
            store.append_log(run_id, f"submitted InferenceEngine job job_id={job_id}")
            poll_sec = float(os.getenv("INFERENCE_PIPELINE_POLL_SEC") or DEFAULT_POLL_SEC)
            timeout_sec = float(os.getenv("INFERENCE_PIPELINE_TIMEOUT_SEC") or DEFAULT_TIMEOUT_SEC)
            final_state = client.wait(job_id, poll_sec=poll_sec, timeout_sec=timeout_sec)
            store.update_run(run_id, progress_percent=80)
            artifacts = _read_artifacts(client, job_id)
            warnings = _artifact_warnings(artifacts)
            store.append_log(run_id, f"InferenceEngine job finished status={final_state.get('status')}")
        if str(final_state.get("status") or "").lower() not in {"success", "succeeded", "completed"}:
            raise RuntimeError(str(final_state.get("error") or f"InferenceEngine job ended with status={final_state.get('status')}"))
        summary = build_final_summary(
            request=request,
            run_id=run_id,
            job_id=job_id,
            final_state=final_state,
            artifacts=artifacts,
            warnings=warnings,
        )
        store.write_summary(run_id, summary)
        mlflow_result = _log_mlflow_metadata(request, run_id, store, summary)
        if mlflow_result:
            summary["mlflow"] = mlflow_result
            store.write_summary(run_id, summary)
        duration = round(time.time() - started, 3)
        store.update_run(
            run_id,
            state="succeeded",
            progress_percent=100,
            finished_at=utc_now(),
            duration_sec=duration,
            artifacts=artifacts,
            summary=summary,
            mlflow=mlflow_result or {},
        )
        store.append_log(run_id, "worker finished successfully")
    except Exception as exc:  # noqa: BLE001
        duration = round(time.time() - started, 3)
        error = {"type": type(exc).__name__, "message": str(exc)}
        store.write_summary(run_id, {"run_id": run_id, "status": "failed", "error": error})
        store.update_run(run_id, state="failed", progress_percent=100, finished_at=utc_now(), duration_sec=duration, error=error)
        store.append_log(run_id, f"worker failed: {type(exc).__name__}: {exc}")
        raise


def _read_artifacts(client: Any, job_id: str) -> dict[str, Any]:
    try:
        response = client.get_artifacts(job_id)
    except Exception as exc:  # noqa: BLE001
        return {"_warning": f"artifacts endpoint unavailable: {type(exc).__name__}: {exc}"}
    return response if isinstance(response, dict) else {"value": response}


def _artifact_warnings(artifacts: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not artifacts:
        warnings.append("InferenceEngine artifacts response was empty.")
    if artifacts.get("_warning"):
        warnings.append(str(artifacts["_warning"]))
    return warnings


def _log_mlflow_metadata(request: Any, run_id: str, store: PseudolabelRunStore, summary: dict[str, Any]) -> dict[str, Any]:
    try:
        from ..mlflow_adapter.api import log_lightweight_run
        from ..settings.api import load_config

        config = load_config()
        return log_lightweight_run(
            config,
            config.mlflow_default_experiment,
            f"pseudolabel-{request.experiment_id}-{run_id}",
            params={
                "inference_pipeline.run_id": run_id,
                "inference_pipeline.experiment_id": request.experiment_id,
                "inference_pipeline.model_ref": request.model_ref,
                "inference_pipeline.images_uri": request.images_uri,
            },
            metrics={},
            artifacts=[store.summary_path(run_id), store.run_dir(run_id) / "inference_engine_payload.json"],
            tags={"orchestrator": "inference_pipeline", "job_type": "pseudolabel"},
        )
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", default=None)
    args = parser.parse_args(argv)
    run_worker(args.run_id, args.run_root)


if __name__ == "__main__":
    main()
