#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any


STAGES = [
    "inventory_scenes",
    "prepare_dataset",
    "create_mlflow_run",
    "train_model",
    "evaluate_pixel_metrics",
    "predict_validation_scenes",
    "vectorize_validation_predictions",
    "compute_f1",
    "log_mlflow_artifacts",
    "write_codex_api_summary",
    "finalize_mlflow_run",
]

BASELINE_CHECKPOINT = Path(
    "/data/mlsystem/airflow/status/"
    "cuttings_mlmarkup_b2unfreeze3_ufz3_lr5e8_enc005_do35_wd15e4_clip03_ftv_20260510_143356/"
    "segformer_b2.best.pt"
)

CLASS_SCENE_MATCHING_PREFIXES = {
    "lakes": ["images/kanopus/wave_2_Upload_01/"],
    "abrasion": ["images/kanopus/Olhonskij/"],
    "wind_erosion": ["images/kanopus/irkutsk/", "images/kanopus/Olhonskij/"],
}


class ApiClient:
    def __init__(self, base_url: str, token: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def post(self, path: str, payload: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        request.add_header("Accept", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def get(self, path: str, *, timeout: int = 60) -> dict[str, Any]:
        request = urllib.request.Request(self.base_url + path, method="GET")
        request.add_header("Accept", "application/json")
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))


class TuningController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.class_name = args.class_name
        self.class_slug = args.class_slug
        self.root = Path(args.tuning_root) / self.class_slug
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.history_path = self.root / "history.jsonl"
        self.latest_config_path = self.root / "latest_config.json"
        self.log_path = self.root / "tuning.log"
        self.env = load_env(Path(args.env_file))
        self.api = ApiClient(
            args.api_url or self.env.get("MLSYSTEM_API_BASE_URL") or self.env.get("MLSYSTEM_API_URL") or "http://127.0.0.1:8088",
            args.api_token or self.env.get("MLSYSTEM_API_TOKEN"),
        )
        self.rng = random.Random(args.seed)

    def run_forever(self) -> None:
        self.log(f"controller start class={self.class_name} slug={self.class_slug} smoke={self.args.smoke}")
        state = self.load_state()
        while not self.stop_requested():
            if self.args.max_trials and int(state.get("trials_completed") or 0) >= self.args.max_trials:
                self.log("max_trials reached; stopping")
                return
            trial = self.next_trial(state)
            try:
                result = self.run_trial(trial)
                state = self.update_state_after_trial(state, trial, result)
                self.append_history({**result, "trial": trial})
                self.refresh_frontend_report()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_now(),
                    "trial_id": trial["trial_id"],
                    "config_hash": trial["config_hash"],
                }
                self.log(f"trial failed id={trial['trial_id']} error={result['error']}")
                state = self.update_state_after_trial(state, trial, result)
                self.append_history({**result, "trial": trial})
            if self.args.max_trials:
                continue
            sleep_sec = max(1, int(self.args.sleep_sec))
            self.log(f"sleep {sleep_sec}s")
            time.sleep(sleep_sec)

    def next_trial(self, state: dict[str, Any]) -> dict[str, Any]:
        tried = set(state.get("tried_config_hashes") or [])
        for _ in range(200):
            config = sample_config(self.class_slug, self.rng, smoke=bool(self.args.smoke), epochs=self.args.epochs)
            config_hash = stable_hash(config)
            if config_hash not in tried:
                break
        else:
            config_hash = stable_hash(config)
        trial_index = int(state.get("trials_started") or 0) + 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trial_id = f"tune_{self.class_slug}_{timestamp}_{trial_index:04d}_{config_hash[:8]}"
        return {"trial_id": trial_id, "config": config, "config_hash": config_hash, "trial_index": trial_index}

    def run_trial(self, trial: dict[str, Any]) -> dict[str, Any]:
        self.write_state({**self.load_state(), "status": "running", "current_trial_id": trial["trial_id"], "last_update_at": utc_now()})
        class_dir = find_class_dir(Path(self.args.mlmarkup_dir), self.class_name)
        layout_dir = self.sync_layout(class_dir)
        stats = dataset_stats(class_dir)
        experiment_config = self.build_experiment_config(trial, layout_dir, stats)
        write_json_atomic(self.latest_config_path, experiment_config)
        self.log(f"trial start id={trial['trial_id']} hash={trial['config_hash']} config={compact_config(trial['config'])}")
        stage_results: list[dict[str, Any]] = []
        started = time.time()
        run_id: str | None = None
        for stage in STAGES:
            with gpu_lock(stage, Path(self.args.tuning_root), enabled=(stage in {"train_model", "predict_validation_scenes"} and bool(self.args.gpu_lock))):
                result = self.run_stage(trial["trial_id"], stage, experiment_config)
            stage_results.append(result)
            if stage == "create_mlflow_run":
                run_id = extract_mlflow_run_id(result)
            if result.get("state") != "succeeded":
                raise RuntimeError(f"stage {stage} ended with state={result.get('state')}: {result.get('error')}")
        summary = self.safe_run_summary(trial["trial_id"])
        if not run_id:
            run_id = ((summary.get("mlflow") or {}).get("run_id") or summary.get("mlflow_run_id"))
        metrics = extract_metrics(summary, Path(self.args.status_root) / trial["trial_id"])
        duration_sec = round(time.time() - started, 3)
        self.log(f"trial complete id={trial['trial_id']} mlflow_run={run_id} f1={metrics.get('best_val_pixel_f1')}")
        return {
            "status": "succeeded",
            "trial_id": trial["trial_id"],
            "config_hash": trial["config_hash"],
            "mlflow_run_id": run_id,
            "metrics": metrics,
            "duration_sec": duration_sec,
            "finished_at": utc_now(),
            "stage_results": stage_results,
        }

    def run_stage(self, airflow_run_id: str, stage: str, experiment_config: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "experiment_config": experiment_config,
            "airflow_run_id": airflow_run_id,
            "status_root": self.args.status_root,
            "source": "tuning-controller",
        }
        self.log(f"stage start trial={airflow_run_id} stage={stage}")
        started = self.api.post(f"/api/v1/runs/{airflow_run_id}/stages/{stage}/start", payload)
        job_id = started.get("job_id")
        timeout = self.args.train_timeout_sec if stage == "train_model" else self.args.stage_timeout_sec
        status = wait_for_job(self.api, str(job_id), timeout_sec=timeout, poll_sec=self.args.poll_sec)
        status["stage"] = stage
        self.log(f"stage done trial={airflow_run_id} stage={stage} job={job_id} state={status.get('state')}")
        return status

    def build_experiment_config(self, trial: dict[str, Any], layout_dir: Path, stats: dict[str, Any]) -> dict[str, Any]:
        config = trial["config"]
        scenes_file = stats["scenes_file"]
        annotation_file = stats["annotation_file"]
        validation_kind = "scene_level" if self.class_slug == "lakes" else "scene_level_limited"
        params = {
            "class_name": self.class_name,
            "tuning.enabled": "true",
            "tuning.class": self.class_name,
            "tuning.class_slug": self.class_slug,
            "tuning.controller_version": current_git_sha(),
            "tuning.config_hash": trial["config_hash"],
            "dataset.class_name": self.class_name,
            "dataset.date": stats.get("dataset_date"),
            "dataset.objects": stats.get("objects_count"),
            "dataset.scenes": stats.get("scenes_count"),
            "dataset.fingerprint": stats.get("fingerprint"),
            "dataset.mlmarkup_commit": stats.get("mlmarkup_commit"),
            "validation.kind": validation_kind,
            "validation.split_strategy": config["split_strategy"],
        }
        train = {
            "enabled": True,
            "require_gpu": True,
            "epochs": config["epochs"],
            "max_epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "optimizer": "adamw",
            "scheduler": {"name": config["scheduler"]},
            "loss": {"name": config["loss"]},
            "early_stopping": {"enabled": True, "patience": config["early_stopping_patience"]},
            "augmentations": config["augmentations"],
            "metric_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
            "objective_metric": "val/pixel_f1",
            "maximize": True,
            "cache_samples_on_gpu": config.get("cache_samples_on_gpu", True),
        }
        if BASELINE_CHECKPOINT.exists():
            train["initial_checkpoint_path"] = str(BASELINE_CHECKPOINT)
            train["initial_checkpoint_strict"] = False
        if config.get("max_train_tiles"):
            train["max_train_tiles"] = config["max_train_tiles"]
            train["max_val_tiles"] = config["max_val_tiles"]
        return {
            "schema_version": 2,
            "experiment_id": trial["trial_id"],
            "class_name": self.class_name,
            "task": "train_predict_pseudolabel",
            "smoke": False,
            "images_uri": self.args.images_uri,
            "layout_uri": str(layout_dir),
            "scenes_file": scenes_file,
            "annotation_file": annotation_file,
            "model": {"name": config["model_name"], "architecture": config["model_name"], "input_bands": [1, 2, 3, 4]},
            "preprocess": {
                "patch_size": config["patch_size"],
                "tile_size": config["patch_size"],
                "stride": config["stride"],
                "split_strategy": config["split_strategy"],
                "target_val_fraction": config["target_val_fraction"],
                "split_seed": config["split_seed"],
                "include_negative_scenes": True,
                "allow_inferred_annotation_crs": True,
                "scene_matching_prefer_prefixes": CLASS_SCENE_MATCHING_PREFIXES.get(self.class_slug, []),
                "allow_train_val_sample_fallback": True,
                "max_train_tiles": config.get("max_train_tiles"),
                "max_val_tiles": config.get("max_val_tiles"),
            },
            "train": train,
            "evaluate": {"threshold": 0.5, "metric_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7]},
            "postprocess": {"enabled": True, "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7]},
            "mlflow": {"experiment": "mlsystem-class-training"},
            "pseudolabel": {"enabled": False},
            "params": params,
        }

    def sync_layout(self, class_dir: Path) -> Path:
        layout_dir = self.root / "layout"
        tmp_dir = self.root / f"layout.tmp.{os.getpid()}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        shutil.copytree(class_dir, tmp_dir)
        if layout_dir.exists():
            shutil.rmtree(layout_dir)
        tmp_dir.replace(layout_dir)
        return layout_dir

    def safe_run_summary(self, airflow_run_id: str) -> dict[str, Any]:
        try:
            return self.api.get(f"/api/v1/runs/{airflow_run_id}/summary")
        except Exception as exc:  # noqa: BLE001
            self.log(f"run_summary unavailable trial={airflow_run_id}: {type(exc).__name__}: {exc}")
            return {}

    def refresh_frontend_report(self) -> None:
        try:
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
            login_data = f"username={self.args.frontend_user}&password={self.args.frontend_password}".encode("utf-8")
            opener.open(
                urllib.request.Request(
                    self.args.frontend_url.rstrip("/") + "/login",
                    data=login_data,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                ),
                timeout=20,
            )
            opener.open(urllib.request.Request(self.args.frontend_url.rstrip("/") + "/api/training-report/refresh", method="POST"), timeout=60)
            self.log("frontend training report refresh requested")
        except Exception as exc:  # noqa: BLE001
            self.log(f"frontend refresh skipped: {type(exc).__name__}: {exc}")

    def load_state(self) -> dict[str, Any]:
        payload = read_json(self.state_path, default={})
        return payload if isinstance(payload, dict) else {}

    def write_state(self, state: dict[str, Any]) -> None:
        write_json_atomic(self.state_path, state)

    def update_state_after_trial(self, state: dict[str, Any], trial: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        tried = list(dict.fromkeys([*(state.get("tried_config_hashes") or []), trial["config_hash"]]))
        trials_completed = int(state.get("trials_completed") or 0) + 1
        best_f1 = state.get("best_f1")
        current_f1 = ((result.get("metrics") or {}).get("best_val_pixel_f1") if result.get("status") == "succeeded" else None)
        if current_f1 is not None and (best_f1 is None or float(current_f1) > float(best_f1)):
            best_f1 = current_f1
            state["best_run_id"] = result.get("mlflow_run_id")
        state.update(
            {
                "status": "idle" if result.get("status") == "succeeded" else "error",
                "class_name": self.class_name,
                "class_slug": self.class_slug,
                "trials_started": max(int(state.get("trials_started") or 0), int(trial["trial_index"])),
                "trials_completed": trials_completed,
                "tried_config_hashes": tried,
                "last_trial_id": trial["trial_id"],
                "last_run_id": result.get("mlflow_run_id"),
                "last_status": result.get("status"),
                "last_error": result.get("error"),
                "best_f1": best_f1,
                "last_update_at": utc_now(),
            }
        )
        self.write_state(state)
        return state

    def append_history(self, payload: dict[str, Any]) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def stop_requested(self) -> bool:
        return (self.root / "STOP").exists()

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sample_config(class_slug: str, rng: random.Random, *, smoke: bool, epochs: int | None) -> dict[str, Any]:
    if class_slug == "lakes":
        model_name = rng.choice(["segformer_b2", "segformer_b1", "segformer_b2"])
        patch_size = rng.choice([768, 1024, 1280])
        stride = rng.choice([512, 640, 768])
        learning_rate = rng.choice([1e-4, 2e-4, 3e-4, 5e-4, 8e-4])
        weight_decay = rng.choice([0.0, 1e-5, 1e-4, 1e-3])
        loss = rng.choice(["bce_dice", "focal_dice", "tversky_dice"])
        if patch_size >= 1280:
            batch_size = rng.choice([1, 2, "auto"])
        elif patch_size >= 1024:
            batch_size = rng.choice([2, 4, "auto"])
        else:
            batch_size = rng.choice([2, 4, "auto"])
        target_val_fraction = 0.2
        default_epochs = rng.choice([15, 20, 25, 40, 60])
        max_train_tiles = 96 if smoke else None
        max_val_tiles = 48 if smoke else None
    elif class_slug == "wind_erosion":
        model_name = rng.choice(["segformer_b2", "segformer_b1", "segformer_b2"])
        patch_size = rng.choice([512, 768, 1024])
        stride = rng.choice([384, 512, 768])
        learning_rate = rng.choice([3e-5, 5e-5, 1e-4, 2e-4, 3e-4])
        weight_decay = rng.choice([1e-5, 1e-4, 1e-3, 0.0015])
        loss = rng.choice(["focal_dice", "tversky_dice", "bce_dice"])
        batch_size = rng.choice([1, 2, "auto"])
        target_val_fraction = 0.5
        default_epochs = rng.choice([20, 30, 40, 60, 80])
        max_train_tiles = 64 if smoke else None
        max_val_tiles = 32 if smoke else None
    else:
        model_name = "segformer_b2"
        patch_size = rng.choice([512, 768, 1024])
        stride = rng.choice([384, 512, 768])
        learning_rate = rng.choice([3e-5, 5e-5, 1e-4, 2e-4])
        weight_decay = rng.choice([1e-5, 1e-4, 1e-3])
        loss = rng.choice(["focal_dice", "tversky_dice", "bce_dice"])
        batch_size = rng.choice([1, 2, "auto"])
        target_val_fraction = 0.34
        default_epochs = rng.choice([20, 30, 40, 60, 80])
        max_train_tiles = 64 if smoke else None
        max_val_tiles = 32 if smoke else None
    if smoke:
        default_epochs = 2
    return {
        "model_name": model_name,
        "patch_size": patch_size,
        "stride": stride,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": rng.choice(["cosine", "none"]),
        "loss": loss,
        "epochs": int(epochs or default_epochs),
        "early_stopping_patience": rng.choice([8, 10, 12, 15]),
        "split_strategy": "object_balanced",
        "target_val_fraction": target_val_fraction,
        "split_seed": rng.randint(1, 999999),
        "augmentations": {
            "flips": True,
            "rot90": True,
            "brightness_contrast": True,
            "gamma": class_slug != "lakes",
            "noise": True,
            "blur": class_slug != "lakes",
        },
        "cache_samples_on_gpu": patch_size < 1280,
        "max_train_tiles": max_train_tiles,
        "max_val_tiles": max_val_tiles,
    }


@contextlib.contextmanager
def gpu_lock(stage: str, tuning_root: Path, *, enabled: bool):
    if not enabled:
        yield
        return
    lock_path = tuning_root / "gpu_training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_UN)
            except Exception:
                pass


def wait_for_job(api: ApiClient, job_id: str, *, timeout_sec: int, poll_sec: int) -> dict[str, Any]:
    started = time.time()
    while True:
        status = api.get(f"/api/v1/jobs/{job_id}")
        if status.get("state") in {"succeeded", "failed", "cancelled", "timed_out"}:
            return status
        if time.time() - started > timeout_sec:
            raise TimeoutError(f"Timed out waiting for job {job_id}")
        time.sleep(max(1, poll_sec))


def find_class_dir(mlmarkup_dir: Path, class_name: str) -> Path:
    exact = mlmarkup_dir / class_name
    if exact.exists():
        return exact
    matches = [path for path in mlmarkup_dir.iterdir() if path.is_dir() and path.name.casefold() == class_name.casefold()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"class directory not found: {class_name} under {mlmarkup_dir}")


def dataset_stats(class_dir: Path) -> dict[str, Any]:
    geojson = sorted(class_dir.glob("*.geojson"))[0]
    scenes = sorted(class_dir.glob("*.txt"))[0]
    objects_count = count_geojson_features(geojson)
    scenes_count = count_scenes(scenes, class_dir)
    return {
        "class_dir": str(class_dir),
        "annotation_file": geojson.name,
        "scenes_file": scenes.name,
        "objects_count": objects_count,
        "scenes_count": scenes_count,
        "dataset_date": datetime.fromtimestamp(max(geojson.stat().st_mtime, scenes.stat().st_mtime), tz=timezone.utc).date().isoformat(),
        "fingerprint": stable_hash({"geojson": file_digest(geojson), "scenes": file_digest(scenes), "objects": objects_count, "scenes_count": scenes_count}),
        "mlmarkup_commit": current_git_sha(class_dir.parent),
    }


def count_geojson_features(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        return len(payload.get("features") or [])
    return 1 if isinstance(payload, dict) and payload.get("type") == "Feature" else 0


def count_scenes(path: Path, class_dir: Path) -> int:
    total = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        candidate = class_dir / line
        if candidate.is_dir():
            total += sum(1 for item in candidate.rglob("*") if item.suffix.lower() in {".tif", ".tiff"})
        else:
            total += 1
    return total


def extract_mlflow_run_id(status: dict[str, Any]) -> str | None:
    report = status.get("report") or {}
    counters = report.get("counters") or {}
    return status.get("mlflow_run_id") or counters.get("mlflow_run_id") or ((report.get("details") or {}).get("mlflow") or {}).get("run_id")


def extract_metrics(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    payload = summary.get("training_result") or read_json(run_dir / "training_result.json", default={}) or {}
    pixel_f1 = first_float(
        payload,
        [
            "best_val_pixel_f1",
            "val/pixel_f1_best_threshold",
            "last_epoch_metrics/val/pixel_f1_best_threshold",
            "val/pixel_f1",
            "last_epoch_metrics/val/pixel_f1",
            "val/class_pixel_f1",
            "last_epoch_metrics/val/class_pixel_f1",
            "val/pixel_dice",
            "last_epoch_metrics/val/pixel_dice",
            "val/dice",
            "last_epoch_metrics/val/dice",
        ],
    )
    result = {
        "best_val_pixel_f1": pixel_f1,
        "epochs_completed": payload.get("epochs_completed"),
        "best_epoch": payload.get("best_epoch") or payload.get("best_val_epoch"),
    }
    for metric in ("val/pixel_precision", "val/pixel_recall", "val/pixel_iou", "val/object_f1"):
        value = find_scalar(payload, metric)
        if value is not None:
            result[metric.replace("/", "_")] = value
    return result


def walk_scalars(payload: Any, prefix: str = ""):
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            yield from walk_scalars(value, next_prefix)
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            yield from walk_scalars(value, f"{prefix}/{idx}")
    else:
        yield prefix, payload


def find_scalar(payload: Any, wanted: str) -> Any:
    wanted_l = wanted.lower()
    for key, value in walk_scalars(payload):
        if key.lower().endswith(wanted_l):
            return value
    return None


def first_float(payload: Any, wanted_keys: list[str]) -> float | None:
    for wanted in wanted_keys:
        value = find_scalar(payload, wanted)
        if value is None:
            continue
        with contextlib.suppress(TypeError, ValueError):
            return float(value)
    return None


def read_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def load_env(path: Path) -> dict[str, str]:
    values = read_env(path)
    values.update({key: value for key, value in os.environ.items() if key.startswith(("MLSYSTEM_", "MLFLOW_", "AWS_", "FRONTEND_"))})
    return values


def read_json(path: Path, *, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(repo: Path | None = None) -> str:
    repo = repo or Path(__file__).resolve().parents[1]
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref:"):
        ref = repo / ".git" / value.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else "unknown"
    return value


def compact_config(config: dict[str, Any]) -> str:
    keys = ["model_name", "patch_size", "stride", "batch_size", "learning_rate", "weight_decay", "loss", "epochs"]
    return ",".join(f"{key}={config.get(key)}" for key in keys)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous MLSystem class tuning until a stop file appears.")
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--class-slug", required=True)
    parser.add_argument("--mlmarkup-dir", default="/data/MLMarkup")
    parser.add_argument("--tuning-root", default="/data/mlsystem/tuning")
    parser.add_argument("--status-root", default="/data/mlsystem/airflow/status")
    parser.add_argument("--env-file", default="/etc/mlsystem/gpu-platform.env")
    parser.add_argument("--api-url", default="http://127.0.0.1:8088")
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--images-uri", default="s3://mlsystems/images/")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8090")
    parser.add_argument("--frontend-user", default="mluser")
    parser.add_argument("--frontend-password", default="qazwsxedc")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--sleep-sec", type=int, default=30)
    parser.add_argument("--poll-sec", type=int, default=10)
    parser.add_argument("--stage-timeout-sec", type=int, default=1800)
    parser.add_argument("--train-timeout-sec", type=int, default=21600)
    parser.add_argument("--smoke", action="store_true", help="Run reduced real trials; does not set MLSystem smoke mode.")
    parser.add_argument("--gpu-lock", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    TuningController(args).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
