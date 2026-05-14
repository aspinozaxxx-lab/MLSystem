from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MLSystem pipeline smoke traces through the FastAPI runner.")
    parser.add_argument("--api", default="http://127.0.0.1:8088", help="MLSystem API base URL.")
    parser.add_argument("--token", default=None, help="API bearer token. Defaults to MLSYSTEM_API_TOKEN or token file.")
    parser.add_argument("--token-file", default="/etc/mlsystem/gpu-platform.env", help="Env file containing MLSYSTEM_API_TOKEN.")
    parser.add_argument("--dry-run", action="store_true", help="Run a dry orchestration smoke trace.")
    parser.add_argument("--real-train", action="store_true", help="Run a short real training smoke trace.")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--dry-timeout-sec", type=float, default=300.0)
    parser.add_argument("--train-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    run_dry = args.dry_run or not args.real_train
    token = args.token or os.getenv("MLSYSTEM_API_TOKEN") or _read_env_value(Path(args.token_file), "MLSYSTEM_API_TOKEN")
    if not token:
        raise SystemExit("MLSYSTEM_API_TOKEN is required via --token, env, or --token-file")

    api = args.api.rstrip("/")
    result: dict[str, Any] = {"api": api, "runs": []}

    if run_dry:
        dry_result = _run_trace(
            api,
            token,
            _dry_trace(),
            poll_sec=max(1.0, args.poll_sec),
            timeout_sec=max(1.0, args.dry_timeout_sec),
        )
        result["runs"].append(dry_result)

    if args.real_train:
        train_result = _run_trace(
            api,
            token,
            _train_trace(),
            poll_sec=max(1.0, args.poll_sec),
            timeout_sec=max(1.0, args.train_timeout_sec),
        )
        result["runs"].append(train_result)

    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    failures = [item for item in result["runs"] if item.get("state") != "succeeded"]
    return 1 if failures else 0


def _run_trace(api: str, token: str, trace: dict[str, Any], *, poll_sec: float, timeout_sec: float) -> dict[str, Any]:
    response = _request_json(api + "/api/v1/pipeline-runs", token, {"trace": trace})
    run_id = str(response["run_id"])
    print(f"started {run_id}", flush=True)
    deadline = time.monotonic() + timeout_sec
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = _request_json(api + f"/api/v1/pipeline-runs/{run_id}", token)
        state = str(status.get("state"))
        print(
            f"{run_id}: state={state} progress={status.get('progress_percent')} stage={status.get('current_stage')}",
            flush=True,
        )
        if state in TERMINAL_STATES:
            break
        time.sleep(poll_sec)
    else:
        raise SystemExit(f"timeout waiting for pipeline run {run_id}")

    log = _request_json(api + f"/api/v1/pipeline-runs/{run_id}/log?tail=12000", token)
    stages = _request_json(api + f"/api/v1/pipeline-runs/{run_id}/stages", token)
    run_result = {
        "run_id": run_id,
        "experiment_id": trace["experiment_id"],
        "state": status.get("state"),
        "progress_percent": status.get("progress_percent"),
        "current_stage": status.get("current_stage"),
        "mlflow": status.get("mlflow") or {},
        "artifacts": status.get("artifacts") or {},
        "error": status.get("error"),
        "log_tail": log.get("log_tail", "")[-12000:],
        "stages": stages.get("stages") or [],
        "reports": stages.get("reports") or [],
    }
    if run_result["state"] != "succeeded":
        raise SystemExit(json.dumps(run_result, ensure_ascii=False, indent=2, sort_keys=True))
    return run_result


def _request_json(url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {token}"}
    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"{url} returned HTTP {exc.code}: {body}") from exc


def _read_env_value(path: Path, key: str) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
    except OSError:
        return ""
    return ""


def _dry_trace() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "server_api_dry_run_no_airflow",
        "pipeline": {
            "dry_run": True,
            "stages": ["inventory", "prepare-dataset", "train", "finalize"],
        },
    }


def _train_trace() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "server_train_smoke_no_airflow_" + time.strftime("%Y%m%d_%H%M%S"),
        "class_name": "cuttings",
        "task": "train_predict_pseudolabel",
        "pipeline": {
            "stages": ["inventory", "prepare-dataset", "train", "evaluate", "compute-f1", "finalize"],
            "stop_on_failure": True,
            "dry_run": False,
            "log_mlflow": True,
        },
        "images_uri": "s3://mlsystems/images/",
        "layout_uri": "s3://mlsystems/layouts/deforest/",
        "scenes_file": "scenes.txt",
        "annotation_file": "auto",
        "pseudolabel": {"enabled": False},
        "preprocess": {
            "max_dataset_scenes": 4,
            "dataset_limit_reason": "server real train smoke cap",
            "tile_size": 512,
            "stride": 512,
            "max_train_tiles": 4,
            "max_val_tiles": 2,
            "train_sampling": {"enabled": True},
        },
        "train": {
            "max_epochs": 1,
            "max_train_batches": 2,
            "max_val_batches": 1,
            "max_train_tiles": 4,
            "max_val_tiles": 2,
            "batch_size": 2,
            "require_gpu": True,
            "time_limit_sec": 900,
        },
        "model": {"name": "tiny_unet_4ch"},
        "mlflow": {"experiment": "mlsystem-deforest"},
    }


if __name__ == "__main__":
    sys.exit(main())
