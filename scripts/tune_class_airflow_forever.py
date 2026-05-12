#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_STATUS_ROOT = Path("/data/mlsystem/airflow/status")
DEFAULT_TUNING_ROOT = Path("/data/mlsystem/tuning")
DEFAULT_AIRFLOW_CONTAINER = "mlsystem-gpu-airflow-scheduler"
DEFAULT_DAG_ID = "mlsystem_experiment_pipeline"
DEFAULT_IMAGES_URI = "s3://mlsystems/images/"
DEFAULT_MLFLOW_EXPERIMENT = "mlsystem-class-training"
DEFAULT_LAKES_CHECKPOINT = Path(
    "/data/mlsystem/airflow/status/tune_lakes_20260512_041935_0189_9a6a02e7/segformer_b2.pt"
)

CLASS_SCENE_PREFIXES = {
    "lakes": ["images/kanopus/wave_2_Upload_01/"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def run_text(args: list[str], *, check: bool = True, timeout: int | None = None) -> str:
    proc = subprocess.run(args, check=check, capture_output=True, text=True, timeout=timeout)
    return proc.stdout.strip()


def find_class_dir(mlmarkup_dir: Path, class_name: str) -> Path:
    direct = mlmarkup_dir / class_name
    if direct.exists():
        return direct
    matches = [path for path in mlmarkup_dir.rglob("*") if path.is_dir() and path.name == class_name]
    if not matches:
        raise FileNotFoundError(f"Class directory not found: {class_name} under {mlmarkup_dir}")
    return matches[0]


def sync_layout(class_dir: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(class_dir, destination)
    for path in destination.rglob("*"):
        try:
            path.chmod(0o777 if path.is_dir() else 0o666)
        except OSError:
            pass
    destination.chmod(0o777)
    return destination


def count_geojson_objects(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    features = data.get("features") if isinstance(data, dict) else []
    return len(features) if isinstance(features, list) else 0


def dataset_stats(class_dir: Path) -> dict[str, Any]:
    geojsons = sorted(class_dir.glob("*.geojson"))
    txts = sorted(class_dir.glob("*.txt"))
    if not geojsons or not txts:
        raise FileNotFoundError(f"Expected one .geojson and one .txt in {class_dir}")
    scenes = [
        line.strip()
        for line in txts[0].read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    objects = sum(count_geojson_objects(path) for path in geojsons)
    fingerprint = stable_hash(
        {
            "geojson": [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in geojsons],
            "txt": [(path.name, path.stat().st_size, path.stat().st_mtime_ns) for path in txts],
        }
    )
    return {
        "class_dir": str(class_dir),
        "annotation_file": geojsons[0].name,
        "scenes_file": txts[0].name,
        "objects": objects,
        "scenes": len(scenes),
        "fingerprint": fingerprint,
    }


def sample_lakes_trial(rng: random.Random, trial_index: int, *, initial_checkpoint: str | None) -> dict[str, Any]:
    base_variants = [
        {
            "hypothesis": "Fine-tune the best lake B2 checkpoint with lower LR to improve stable validation F1 without drifting from the current optimum.",
            "model_name": "segformer_b2",
            "patch_size": 1024,
            "stride": 512,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
            "loss": "focal_dice",
            "scheduler": "none",
            "epochs": 50,
            "early_stopping_patience": 14,
            "split_seed": 20260513,
        },
        {
            "hypothesis": "Use cosine decay from the same checkpoint; test whether smoother late training improves pixel F1 stability.",
            "model_name": "segformer_b2",
            "patch_size": 1024,
            "stride": 512,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "loss": "focal_dice",
            "scheduler": "cosine",
            "epochs": 60,
            "early_stopping_patience": 16,
            "split_seed": 20260513,
        },
        {
            "hypothesis": "Increase weight decay with BCE+Dice to reduce false positives seen in some lake runs while preserving recall.",
            "model_name": "segformer_b2",
            "patch_size": 768,
            "stride": 512,
            "batch_size": "auto",
            "learning_rate": 1e-4,
            "weight_decay": 1e-3,
            "loss": "bce_dice",
            "scheduler": "none",
            "epochs": 45,
            "early_stopping_patience": 12,
            "split_seed": 20260513,
        },
        {
            "hypothesis": "Use wider context and stronger regularization to test whether lake boundaries benefit from 1024 tiles with conservative updates.",
            "model_name": "segformer_b2",
            "patch_size": 1024,
            "stride": 768,
            "batch_size": 2,
            "learning_rate": 8e-5,
            "weight_decay": 1e-3,
            "loss": "bce_dice",
            "scheduler": "cosine",
            "epochs": 60,
            "early_stopping_patience": 16,
            "split_seed": 20260513,
        },
    ]
    if trial_index <= len(base_variants):
        trial = dict(base_variants[trial_index - 1])
    else:
        trial = {
            "hypothesis": "Mutation around the best recent lake configurations; change one or two training controls and keep scene-level validation fixed.",
            "model_name": "segformer_b2",
            "patch_size": rng.choice([768, 1024]),
            "stride": rng.choice([512, 640, 768]),
            "batch_size": rng.choice([2, "auto"]),
            "learning_rate": rng.choice([3e-5, 5e-5, 8e-5, 1e-4, 2e-4]),
            "weight_decay": rng.choice([1e-5, 1e-4, 3e-4, 1e-3]),
            "loss": rng.choice(["focal_dice", "bce_dice", "focal_tversky"]),
            "scheduler": rng.choice(["none", "cosine"]),
            "epochs": rng.choice([40, 50, 60, 70]),
            "early_stopping_patience": rng.choice([12, 14, 16]),
            "split_seed": 20260513,
        }
    trial["initial_checkpoint_path"] = initial_checkpoint
    trial["target_val_fraction"] = 0.2
    trial["split_strategy"] = "object_balanced"
    trial["augmentations"] = {
        "flips": True,
        "rot90": True,
        "brightness_contrast": True,
        "noise": True,
        "blur": False,
        "gamma": False,
    }
    return trial


def build_experiment_config(
    *,
    run_id: str,
    class_name: str,
    class_slug: str,
    layout_dir: Path,
    stats: dict[str, Any],
    trial: dict[str, Any],
    config_hash: str,
    images_uri: str,
) -> dict[str, Any]:
    train = {
        "enabled": True,
        "require_gpu": True,
        "epochs": trial["epochs"],
        "max_epochs": trial["epochs"],
        "batch_size": trial["batch_size"],
        "learning_rate": trial["learning_rate"],
        "weight_decay": trial["weight_decay"],
        "optimizer": "adamw",
        "scheduler": {"name": trial["scheduler"]},
        "loss": {"name": trial["loss"]},
        "early_stopping": {"enabled": True, "patience": trial["early_stopping_patience"]},
        "augmentations": trial["augmentations"],
        "metric_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
        "objective_metric": "val/pixel_f1",
        "maximize": True,
        "cache_samples_on_gpu": True,
        "allow_train_val_sample_fallback": True,
    }
    if trial.get("initial_checkpoint_path"):
        train["initial_checkpoint_path"] = trial["initial_checkpoint_path"]
        train["initial_checkpoint_strict"] = False
    return {
        "schema_version": 2,
        "experiment_id": run_id,
        "class_name": class_name,
        "task": "train_predict_pseudolabel",
        "smoke": False,
        "images_uri": images_uri,
        "layout_uri": str(layout_dir),
        "scenes_file": stats["scenes_file"],
        "annotation_file": stats["annotation_file"],
        "model": {"name": trial["model_name"], "architecture": trial["model_name"], "input_bands": [1, 2, 3, 4]},
        "preprocess": {
            "patch_size": trial["patch_size"],
            "tile_size": trial["patch_size"],
            "stride": trial["stride"],
            "split_strategy": trial["split_strategy"],
            "target_val_fraction": trial["target_val_fraction"],
            "split_seed": trial["split_seed"],
            "include_negative_scenes": True,
            "allow_inferred_annotation_crs": True,
            "allow_train_val_sample_fallback": True,
            "scene_matching_prefer_prefixes": CLASS_SCENE_PREFIXES.get(class_slug, []),
        },
        "train": train,
        "evaluate": {"threshold": 0.5, "metric_thresholds": [0.3, 0.4, 0.5, 0.6, 0.7]},
        "postprocess": {"enabled": True, "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7]},
        "predict": {"enabled": True},
        "pseudolabel": {"enabled": False},
        "mlflow": {"experiment": DEFAULT_MLFLOW_EXPERIMENT},
        "params": {
            "class_name": class_name,
            "dataset.class_name": class_name,
            "dataset.objects": stats["objects"],
            "dataset.scenes": stats["scenes"],
            "dataset.fingerprint": stats["fingerprint"],
            "tuning.enabled": "true",
            "tuning.class": class_name,
            "tuning.class_slug": class_slug,
            "tuning.config_hash": config_hash,
            "tuning.hypothesis": trial["hypothesis"],
            "validation.kind": "scene_level",
            "validation.split_strategy": trial["split_strategy"],
            "pseudolabeling.enabled": False,
        },
    }


class AirflowTuningController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.class_name = args.class_name
        self.class_slug = args.class_slug
        self.root = Path(args.tuning_root) / self.class_slug
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.history_path = self.root / "history.jsonl"
        self.ledger_path = self.root / "ledger.md"
        self.latest_config_path = self.root / "latest_config.json"
        self.log_path = self.root / "tuning.log"
        self.rng = random.Random(args.seed)

    def log(self, message: str) -> None:
        line = f"{utc_now()} {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def stop_requested(self) -> bool:
        root = Path(self.args.tuning_root)
        return any(
            path.exists()
            for path in [
                root / "STOP_TUNING",
                root / self.class_slug / "STOP_TUNING",
                root / self.class_slug / "STOP",
            ]
        )

    def load_state(self) -> dict[str, Any]:
        return read_json(self.state_path, {"trials_started": 0, "trials_completed": 0, "tried_config_hashes": []})

    def write_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        write_json_atomic(self.state_path, state)

    def run_forever(self) -> None:
        class_dir = find_class_dir(Path(self.args.mlmarkup_dir), self.class_name)
        layout_dir = sync_layout(class_dir, self.root / "layout")
        stats = dataset_stats(layout_dir)
        self.log(f"start class={self.class_name} slug={self.class_slug} layout={layout_dir}")
        state = self.load_state()
        while not self.stop_requested():
            if self.args.max_trials and int(state.get("trials_completed") or 0) >= self.args.max_trials:
                self.log("max_trials reached")
                return
            state = self.load_state()
            trial_index = int(state.get("trials_started") or 0) + 1
            trial = self.next_trial(state, trial_index)
            state["trials_started"] = trial_index
            state["current_trial"] = trial
            state.setdefault("tried_config_hashes", []).append(trial["config_hash"])
            self.write_state(state)
            try:
                result = self.run_trial(layout_dir, stats, trial)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "status": "failed",
                    "run_id": trial["run_id"],
                    "config_hash": trial["config_hash"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_now(),
                }
                self.log(f"trial failed run={trial['run_id']} error={result['error']}")
            append_jsonl(self.history_path, {"trial": trial, "result": result})
            self.append_ledger(trial, result)
            state = self.load_state()
            state["trials_completed"] = int(state.get("trials_completed") or 0) + 1
            state["last_result"] = result
            best = state.get("best") or {}
            f1 = ((result.get("metrics") or {}).get("best_val_pixel_f1") if result.get("status") == "succeeded" else None)
            if f1 is not None and float(f1) > float(best.get("best_val_pixel_f1") or -1):
                state["best"] = {
                    "run_id": result["run_id"],
                    "mlflow_run_id": result.get("mlflow_run_id"),
                    "best_val_pixel_f1": f1,
                    "checkpoint_path": (result.get("metrics") or {}).get("checkpoint_path"),
                    "config_hash": trial["config_hash"],
                }
            self.write_state(state)
            self.refresh_frontend()
            if not self.args.max_trials:
                time.sleep(max(1, int(self.args.sleep_sec)))
        self.log("STOP_TUNING detected; exiting")

    def next_trial(self, state: dict[str, Any], trial_index: int) -> dict[str, Any]:
        tried = set(state.get("tried_config_hashes") or [])
        initial = self.args.initial_checkpoint
        if not initial and DEFAULT_LAKES_CHECKPOINT.exists():
            initial = str(DEFAULT_LAKES_CHECKPOINT)
        for _ in range(500):
            config = sample_lakes_trial(self.rng, trial_index, initial_checkpoint=initial)
            config_hash = stable_hash(config)
            if config_hash not in tried:
                break
            trial_index += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"tune_{self.class_slug}_airflow_{timestamp}_{trial_index:04d}_{config_hash[:8]}"
        return {"run_id": run_id, "trial_index": trial_index, "config": config, "config_hash": config_hash}

    def run_trial(self, layout_dir: Path, stats: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
        run_id = trial["run_id"]
        config = build_experiment_config(
            run_id=run_id,
            class_name=self.class_name,
            class_slug=self.class_slug,
            layout_dir=layout_dir,
            stats=stats,
            trial=trial["config"],
            config_hash=trial["config_hash"],
            images_uri=self.args.images_uri,
        )
        write_json_atomic(self.latest_config_path, config)
        self.log(f"trigger DAG run={run_id} hash={trial['config_hash']} hypothesis={trial['config']['hypothesis']}")
        run_text(
            [
                "docker",
                "exec",
                self.args.airflow_container,
                "airflow",
                "dags",
                "trigger",
                self.args.dag_id,
                "--run-id",
                run_id,
                "--conf",
                json.dumps(config, ensure_ascii=False),
            ],
            timeout=120,
        )
        status = self.wait_for_run(run_id)
        summary = read_json(Path(self.args.status_root) / run_id / "summary.json", {})
        training = summary.get("training_result") or read_json(Path(self.args.status_root) / run_id / "training_result.json", {})
        metrics = {
            "best_val_pixel_f1": training.get("best_val_pixel_f1"),
            "last_val_pixel_f1": (training.get("last_epoch_metrics") or {}).get("val/pixel_f1"),
            "best_epoch": training.get("best_epoch"),
            "epochs_completed": training.get("epochs_completed"),
            "checkpoint_path": training.get("checkpoint_path"),
            "precision": (training.get("last_epoch_metrics") or {}).get("val/precision"),
            "recall": (training.get("last_epoch_metrics") or {}).get("val/recall"),
            "iou": training.get("best_val_iou"),
        }
        mlflow = summary.get("mlflow") or training.get("mlflow") or {}
        result = {
            "status": status,
            "run_id": run_id,
            "mlflow_run_id": mlflow.get("run_id"),
            "mlflow_run_url": mlflow.get("run_url_external") or mlflow.get("external_run_url") or mlflow.get("run_url"),
            "metrics": metrics,
            "finished_at": utc_now(),
        }
        self.log(f"run done run={run_id} status={status} f1={metrics.get('best_val_pixel_f1')} epoch={metrics.get('best_epoch')}")
        return result

    def wait_for_run(self, run_id: str) -> str:
        deadline = time.time() + int(self.args.run_timeout_sec)
        status_path = Path(self.args.status_root) / run_id / "summary.json"
        while time.time() < deadline:
            if self.stop_requested():
                raise RuntimeError("STOP_TUNING requested while DAG run is active; not killing Airflow task automatically")
            summary = read_json(status_path, {})
            status = str(summary.get("status") or "").lower()
            current = summary.get("current_stage")
            if status in {"success", "failed"}:
                return status
            airflow_state = self.airflow_run_state(run_id)
            if airflow_state in {"success", "failed"}:
                return airflow_state
            self.log(f"poll run={run_id} status={status or 'unknown'} stage={current or 'pending'}")
            time.sleep(max(5, int(self.args.poll_sec)))
        raise TimeoutError(f"DAG run timed out: {run_id}")

    def airflow_run_state(self, run_id: str) -> str | None:
        try:
            raw = run_text(
                [
                    "docker",
                    "exec",
                    self.args.airflow_container,
                    "airflow",
                    "dags",
                    "list-runs",
                    "-d",
                    self.args.dag_id,
                    "--no-backfill",
                    "--output",
                    "json",
                ],
                check=False,
                timeout=60,
            )
            for row in json.loads(raw or "[]"):
                if str(row.get("dag_run_id") or "") == run_id:
                    return str(row.get("state") or "").lower() or None
        except Exception:
            return None
        return None

    def append_ledger(self, trial: dict[str, Any], result: dict[str, Any]) -> None:
        first = not self.ledger_path.exists()
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            if first:
                handle.write("# Lakes tuning ledger\n\n")
            metrics = result.get("metrics") or {}
            handle.write(f"## {utc_now()} {trial['run_id']}\n\n")
            handle.write(f"- status: {result.get('status')}\n")
            handle.write(f"- hypothesis: {trial['config']['hypothesis']}\n")
            handle.write(f"- config_hash: `{trial['config_hash']}`\n")
            handle.write(f"- mlflow_run_id: `{result.get('mlflow_run_id')}`\n")
            handle.write(f"- best_val_pixel_f1: {metrics.get('best_val_pixel_f1')}\n")
            handle.write(f"- best_epoch: {metrics.get('best_epoch')}\n")
            handle.write(f"- checkpoint: `{metrics.get('checkpoint_path')}`\n\n")

    def refresh_frontend(self) -> None:
        if not self.args.refresh_url:
            return
        try:
            run_text(["curl", "-fsS", "-X", "POST", self.args.refresh_url], check=False, timeout=30)
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run continuous class tuning through the Airflow DAG.")
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--class-slug", required=True)
    parser.add_argument("--mlmarkup-dir", default="/data/MLMarkup")
    parser.add_argument("--tuning-root", default=str(DEFAULT_TUNING_ROOT))
    parser.add_argument("--status-root", default=str(DEFAULT_STATUS_ROOT))
    parser.add_argument("--airflow-container", default=DEFAULT_AIRFLOW_CONTAINER)
    parser.add_argument("--dag-id", default=DEFAULT_DAG_ID)
    parser.add_argument("--images-uri", default=DEFAULT_IMAGES_URI)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--poll-sec", type=int, default=30)
    parser.add_argument("--sleep-sec", type=int, default=10)
    parser.add_argument("--run-timeout-sec", type=int, default=8 * 3600)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--refresh-url", default="http://127.0.0.1:8090/api/training-report/refresh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.class_slug != "lakes":
        raise SystemExit("This controller currently supports the accumulated lakes tuning line only.")
    AirflowTuningController(args).run_forever()


if __name__ == "__main__":
    main()
