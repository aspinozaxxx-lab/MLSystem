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

DEFAULT_CLASS_CHECKPOINTS = {
    "lakes": Path("/data/mlsystem/airflow/status/train_all_lakes_v4_20260512_225353/segformer_b2.pt"),
    "desertification": Path(
        "/data/mlsystem/airflow/status/train_all_desertification_v2_20260512_223035/segformer_b2.pt"
    ),
}

CLASS_BASELINE_RUNS = {
    "lakes": ["train_all_lakes_v4_20260512_225353", "tune_lakes_20260512_041935_0189_9a6a02e7"],
    "desertification": ["train_all_desertification_v2_20260512_223035"],
}

CLASS_SCENE_PREFIXES = {
    "lakes": ["images/kanopus/wave_2_Upload_01/"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def trial_config_signature(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key not in {"hypothesis", "decision_context"}}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return rows


def run_text(args: list[str], *, check: bool = True, timeout: int | None = None) -> str:
    proc = subprocess.run(args, check=check, capture_output=True, text=True, timeout=timeout)
    return proc.stdout.strip()


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def status_root_checkpoint_path(path: Any, run_id: str, status_root: Path = DEFAULT_STATUS_ROOT) -> str | None:
    if not path:
        return None
    value = str(path)
    prefix = f"/opt/airflow/mlsystem_runs/{run_id}/"
    if value.startswith(prefix):
        return str(status_root / run_id / value[len(prefix) :])
    return value


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


def sample_desertification_trial(rng: random.Random, trial_index: int, *, initial_checkpoint: str | None) -> dict[str, Any]:
    base_variants = [
        {
            "hypothesis": "Continue from the best desertification checkpoint with lower LR; current F1 is recall-limited, so keep threshold sweep and avoid aggressive regularization.",
            "model_name": "segformer_b2",
            "patch_size": 1024,
            "stride": 512,
            "batch_size": 2,
            "learning_rate": 8e-5,
            "weight_decay": 1e-4,
            "loss": "focal_dice",
            "scheduler": "none",
            "epochs": 55,
            "early_stopping_patience": 14,
            "split_seed": 20260513,
        },
        {
            "hypothesis": "Use cosine decay from a slightly higher LR to improve recall without sacrificing the high precision from the baseline.",
            "model_name": "segformer_b2",
            "patch_size": 1024,
            "stride": 512,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "weight_decay": 1e-4,
            "loss": "focal_dice",
            "scheduler": "cosine",
            "epochs": 65,
            "early_stopping_patience": 16,
            "split_seed": 20260513,
        },
        {
            "hypothesis": "Try BCE+Dice with lower threshold-friendly training to raise recall on the object-balanced scene split.",
            "model_name": "segformer_b2",
            "patch_size": 768,
            "stride": 512,
            "batch_size": "auto",
            "learning_rate": 1e-4,
            "weight_decay": 3e-4,
            "loss": "bce_dice",
            "scheduler": "none",
            "epochs": 55,
            "early_stopping_patience": 14,
            "split_seed": 20260513,
        },
        {
            "hypothesis": "Use stronger Tversky-style loss to reduce false negatives; baseline precision is high and recall is the limiting side of F1.",
            "model_name": "segformer_b2",
            "patch_size": 1024,
            "stride": 640,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "loss": "focal_tversky",
            "scheduler": "cosine",
            "epochs": 65,
            "early_stopping_patience": 16,
            "split_seed": 20260513,
        },
    ]
    if trial_index <= len(base_variants):
        trial = dict(base_variants[trial_index - 1])
    else:
        trial = {
            "hypothesis": "Mutation around the best desertification configs; optimize recall-limited pixel F1 on the fixed scene-level split.",
            "model_name": "segformer_b2",
            "patch_size": rng.choice([768, 1024]),
            "stride": rng.choice([512, 640, 768]),
            "batch_size": rng.choice([2, "auto"]),
            "learning_rate": rng.choice([5e-5, 8e-5, 1e-4, 2e-4, 3e-4]),
            "weight_decay": rng.choice([1e-5, 1e-4, 3e-4, 1e-3]),
            "loss": rng.choice(["focal_dice", "bce_dice", "focal_tversky"]),
            "scheduler": rng.choice(["none", "cosine"]),
            "epochs": rng.choice([45, 55, 65, 75]),
            "early_stopping_patience": rng.choice([12, 14, 16, 18]),
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
        "blur": rng.choice([False, True]) if trial_index > len(base_variants) else False,
        "gamma": rng.choice([False, True]) if trial_index > len(base_variants) else True,
    }
    return trial


def sample_class_trial(
    class_slug: str,
    rng: random.Random,
    trial_index: int,
    *,
    initial_checkpoint: str | None,
) -> dict[str, Any]:
    if class_slug == "lakes":
        return sample_lakes_trial(rng, trial_index, initial_checkpoint=initial_checkpoint)
    if class_slug == "desertification":
        return sample_desertification_trial(rng, trial_index, initial_checkpoint=initial_checkpoint)
    raise ValueError(f"Unsupported class slug for tuning: {class_slug}")


def _metric(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = as_float(metrics.get(name))
        if value is not None:
            return value
    return None


def _host_history_path(training: dict[str, Any], run_id: str, status_root: Path) -> Path:
    history = status_root_checkpoint_path(training.get("history_path"), run_id, status_root)
    if history:
        return Path(history)
    return status_root / run_id / "history.json"


def read_run_record(status_root: Path, run_id: str) -> dict[str, Any] | None:
    run_dir = status_root / run_id
    summary = read_json(run_dir / "summary.json", None)
    if not isinstance(summary, dict):
        return None
    training = summary.get("training_result") or read_json(run_dir / "training_result.json", {}) or {}
    config = summary.get("experiment_config") or read_json(run_dir / "experiment_config.json", {}) or {}
    history = read_json(_host_history_path(training, run_id, status_root), []) or []
    best_epoch = training.get("best_epoch")
    best_row: dict[str, Any] | None = None
    if isinstance(history, list) and history:
        if best_epoch is not None:
            try:
                best_epoch_float = float(best_epoch)
                best_row = min(history, key=lambda row: abs(float((row or {}).get("epoch") or 0) - best_epoch_float))
            except Exception:
                best_row = None
        if best_row is None:
            best_row = max(
                history,
                key=lambda row: _metric(
                    row or {},
                    "val/class_pixel_f1",
                    "val/pixel_f1",
                    "epoch/pixel_f1",
                    "objective/value",
                )
                or -1.0,
            )
    last_row = history[-1] if isinstance(history, list) and history else {}
    metrics = {
        "best_val_pixel_f1": _metric(training, "best_val_pixel_f1", "best_objective_value"),
        "last_val_pixel_f1": _metric(last_row or {}, "val/class_pixel_f1", "val/pixel_f1", "epoch/pixel_f1", "objective/value"),
        "best_epoch": best_epoch,
        "epochs_completed": training.get("epochs_completed"),
        "precision": _metric(best_row or {}, "val/class_pixel_precision", "val/precision", "val/precision_best_threshold"),
        "recall": _metric(best_row or {}, "val/class_pixel_recall", "val/recall", "val/recall_best_threshold"),
        "iou": _metric(training, "best_val_iou") or _metric(best_row or {}, "val/class_pixel_iou", "val/iou"),
        "best_threshold": first_value(
            (best_row or {}).get("val/best_threshold"),
            (best_row or {}).get("val/threshold"),
            (last_row or {}).get("val/best_threshold"),
        ),
    }
    checkpoint = status_root_checkpoint_path(training.get("checkpoint_path"), run_id, status_root)
    initial = training.get("initial_checkpoint")
    if isinstance(initial, dict):
        initial = initial.get("path")
    mlflow = summary.get("mlflow") or training.get("mlflow") or {}
    status = str(summary.get("status") or training.get("status") or "").lower()
    if metrics["best_val_pixel_f1"] is None and status == "success":
        status = "failed"
    stable_gap = None
    if metrics["best_val_pixel_f1"] is not None and metrics["last_val_pixel_f1"] is not None:
        stable_gap = float(metrics["best_val_pixel_f1"]) - float(metrics["last_val_pixel_f1"])
    return {
        "run_id": run_id,
        "status": status,
        "summary_status": summary.get("status"),
        "class_name": config.get("class_name") or (config.get("params") or {}).get("class_name"),
        "config": config,
        "train": config.get("train") or {},
        "preprocess": config.get("preprocess") or {},
        "metrics": metrics,
        "stable_gap": stable_gap,
        "overfit": stable_gap is not None and stable_gap > 0.03,
        "checkpoint_path": checkpoint,
        "checkpoint_exists": bool(checkpoint and Path(checkpoint).exists()),
        "initial_checkpoint": initial,
        "mlflow_run_id": mlflow.get("run_id"),
        "mlflow_run_url": mlflow.get("run_url_external") or mlflow.get("external_run_url") or mlflow.get("run_url"),
        "errors": summary.get("errors") or [],
        "finished_at": summary.get("finished_at"),
    }


def summarize_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    metrics = record.get("metrics") or {}
    return {
        "run_id": record.get("run_id"),
        "status": record.get("status"),
        "f1": metrics.get("best_val_pixel_f1"),
        "last_f1": metrics.get("last_val_pixel_f1"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "iou": metrics.get("iou"),
        "best_epoch": metrics.get("best_epoch"),
        "epochs_completed": metrics.get("epochs_completed"),
        "checkpoint_path": record.get("checkpoint_path"),
        "overfit": record.get("overfit"),
        "stable_gap": record.get("stable_gap"),
    }


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
    loss_cfg = {"name": trial["loss"]}
    loss_cfg.update(trial.get("loss_params") or {})
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
        "loss": loss_cfg,
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
            "tuning.initial_checkpoint_path": trial.get("initial_checkpoint_path"),
            "training.phase": "continuous_tuning",
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
        self.refresh_baseline_state(state)
        while not self.stop_requested():
            if self.args.max_trials and int(state.get("trials_completed") or 0) >= self.args.max_trials:
                self.log("max_trials reached")
                return
            state = self.load_state()
            self.refresh_baseline_state(state)
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
            self.refresh_baseline_state(state)
            self.write_state(state)
            self.refresh_frontend()
            if not self.args.max_trials:
                time.sleep(max(1, int(self.args.sleep_sec)))
        self.log("STOP_TUNING detected; exiting")

    def build_leaderboard(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for run_id in CLASS_BASELINE_RUNS.get(self.class_slug, []):
            record = read_run_record(Path(self.args.status_root), run_id)
            if record:
                records.append(record)
                seen.add(run_id)
        for path in sorted(Path(self.args.status_root).glob(f"*{self.class_slug}*/summary.json")):
            run_id = path.parent.name
            if run_id in seen:
                continue
            record = read_run_record(Path(self.args.status_root), run_id)
            if record:
                records.append(record)
                seen.add(run_id)
        normalized = self.class_name.strip().lower()
        for path in sorted(Path(self.args.status_root).glob("*desertification*/summary.json")):
            run_id = path.parent.name
            if self.class_slug != "desertification" or run_id in seen:
                continue
            record = read_run_record(Path(self.args.status_root), run_id)
            if record:
                records.append(record)
                seen.add(run_id)
        records = [
            record
            for record in records
            if record.get("class_name") in {None, "", self.class_name}
            or str(record.get("class_name") or "").strip().lower() == normalized
            or self.class_slug in str(record.get("run_id") or "")
        ]
        return sorted(
            records,
            key=lambda record: (
                as_float((record.get("metrics") or {}).get("best_val_pixel_f1")) or -1.0,
                -abs(as_float(record.get("stable_gap")) or 0.0),
            ),
            reverse=True,
        )

    def refresh_baseline_state(self, state: dict[str, Any]) -> dict[str, Any]:
        leaderboard = self.build_leaderboard()
        state["leaderboard"] = [summarize_record(record) for record in leaderboard[:12]]
        best = next(
            (
                record
                for record in leaderboard
                if record.get("status") == "success"
                and (record.get("metrics") or {}).get("best_val_pixel_f1") is not None
                and record.get("checkpoint_exists")
            ),
            None,
        )
        if best:
            metrics = best.get("metrics") or {}
            state["best"] = {
                "run_id": best.get("run_id"),
                "mlflow_run_id": best.get("mlflow_run_id"),
                "best_val_pixel_f1": metrics.get("best_val_pixel_f1"),
                "last_val_pixel_f1": metrics.get("last_val_pixel_f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "iou": metrics.get("iou"),
                "best_epoch": metrics.get("best_epoch"),
                "checkpoint_path": best.get("checkpoint_path"),
                "stable_gap": best.get("stable_gap"),
                "overfit": best.get("overfit"),
            }
        self.write_state(state)
        return state

    def best_checkpoint(self, state: dict[str, Any]) -> str | None:
        best = state.get("best") or {}
        checkpoint = best.get("checkpoint_path")
        if checkpoint and Path(str(checkpoint)).exists():
            return str(checkpoint)
        default_checkpoint = DEFAULT_CLASS_CHECKPOINTS.get(self.class_slug) or DEFAULT_LAKES_CHECKPOINT
        if default_checkpoint.exists():
            return str(default_checkpoint)
        if self.args.initial_checkpoint and Path(self.args.initial_checkpoint).exists():
            return self.args.initial_checkpoint
        return self.args.initial_checkpoint

    def next_trial(self, state: dict[str, Any], trial_index: int) -> dict[str, Any]:
        tried = set(state.get("tried_config_hashes") or [])
        initial = self.best_checkpoint(state)
        leaderboard = self.build_leaderboard()
        for offset in range(500):
            config = self.adaptive_trial_config(state, leaderboard, trial_index + offset, initial_checkpoint=initial)
            config_hash = stable_hash(trial_config_signature(config))
            if config_hash not in tried:
                trial_index += offset
                break
        else:
            raise RuntimeError("Could not generate a new untried adaptive config")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"tune_{self.class_slug}_airflow_{timestamp}_{trial_index:04d}_{config_hash[:8]}"
        return {"run_id": run_id, "trial_index": trial_index, "config": config, "config_hash": config_hash}

    def adaptive_trial_config(
        self,
        state: dict[str, Any],
        leaderboard: list[dict[str, Any]],
        trial_index: int,
        *,
        initial_checkpoint: str | None,
    ) -> dict[str, Any]:
        if self.class_slug == "lakes" and trial_index <= 4:
            return sample_lakes_trial(self.rng, trial_index, initial_checkpoint=initial_checkpoint)
        if self.class_slug == "desertification":
            return self.adaptive_desertification_config(state, leaderboard, trial_index, initial_checkpoint=initial_checkpoint)
        return sample_class_trial(self.class_slug, self.rng, trial_index, initial_checkpoint=initial_checkpoint)

    def adaptive_desertification_config(
        self,
        state: dict[str, Any],
        leaderboard: list[dict[str, Any]],
        trial_index: int,
        *,
        initial_checkpoint: str | None,
    ) -> dict[str, Any]:
        best_record = leaderboard[0] if leaderboard else None
        best_metrics = (best_record or {}).get("metrics") or {}
        precision = as_float(best_metrics.get("precision"))
        recall = as_float(best_metrics.get("recall"))
        best_f1 = as_float(best_metrics.get("best_val_pixel_f1"))
        last = state.get("last_result") or {}
        last_status = str(last.get("status") or "").lower()
        overfit = bool((best_record or {}).get("overfit"))
        base = self.trial_from_record(best_record, initial_checkpoint=initial_checkpoint)
        base["hypothesis"] = "Continue the best desertification branch and change one controlled factor based on the latest pixel-F1 diagnostics."
        scenarios: list[dict[str, Any]] = []
        if last_status == "failed":
            scenarios.append(
                {
                    "hypothesis": "Previous tuning run failed at orchestration/runtime level, not by metric. Retry from the best valid checkpoint with baseline-safe training controls and no gamma change.",
                    "learning_rate": 1e-4,
                    "loss": "focal_dice",
                    "scheduler": "none",
                    "epochs": 50,
                    "early_stopping_patience": 14,
                    "augmentations": {**base["augmentations"], "gamma": False, "blur": False},
                }
            )
        if precision is not None and recall is not None and precision > recall + 0.08:
            scenarios.extend(
                [
                    {
                        "hypothesis": f"Recall is limiting F1 (precision={precision:.3f}, recall={recall:.3f}); switch to Tversky-biased loss to penalize false negatives.",
                        "loss": "focal_tversky",
                        "loss_params": {"tversky_alpha": 0.35, "tversky_beta": 0.65},
                    },
                    {
                        "hypothesis": f"Recall is limiting F1 (precision={precision:.3f}, recall={recall:.3f}); keep focal_dice but use a slightly higher LR with cosine decay to escape the conservative solution.",
                        "learning_rate": 2e-4,
                        "scheduler": "cosine",
                    },
                    {
                        "hypothesis": "Improve recall by increasing tile overlap while keeping model/loss fixed, so border objects appear in more positive crops.",
                        "stride": 384,
                    },
                ]
            )
        elif precision is not None and recall is not None and recall > precision + 0.08:
            scenarios.extend(
                [
                    {
                        "hypothesis": f"Precision is limiting F1 (precision={precision:.3f}, recall={recall:.3f}); increase regularization to reduce false positives.",
                        "weight_decay": 1e-3,
                    },
                    {
                        "hypothesis": f"Precision is limiting F1 (precision={precision:.3f}, recall={recall:.3f}); use BCE+Dice to make probability calibration more conservative.",
                        "loss": "bce_dice",
                        "learning_rate": 8e-5,
                    },
                ]
            )
        else:
            scenarios.extend(
                [
                    {
                        "hypothesis": f"F1 is balanced near {best_f1}; check a conservative LR decay variant from the best checkpoint.",
                        "learning_rate": 8e-5,
                        "scheduler": "cosine",
                    },
                    {
                        "hypothesis": "Metric is close/balanced; test one crop geometry change while preserving loss and split.",
                        "patch_size": 768,
                        "stride": 512,
                    },
                ]
            )
        if overfit:
            scenarios.insert(
                0,
                {
                    "hypothesis": "Best run has a large best-vs-last F1 gap; reduce LR and add weight decay to improve end-of-training stability.",
                    "learning_rate": 5e-5,
                    "weight_decay": 3e-4,
                    "scheduler": "cosine",
                    "early_stopping_patience": 12,
                },
            )
        index = 0 if last_status == "failed" else max(1, trial_index) - 1
        scenario = scenarios[index % len(scenarios)]
        cycle = index // len(scenarios)
        trial = {**base}
        trial.update({key: value for key, value in scenario.items() if key != "loss_params"})
        if "loss_params" in scenario:
            trial["loss_params"] = scenario["loss_params"]
        if cycle:
            lr_multipliers = [0.75, 1.25, 0.5, 1.5]
            wd_multipliers = [1.0, 3.0, 0.3, 1.0]
            trial["learning_rate"] = round(float(trial["learning_rate"]) * lr_multipliers[(cycle - 1) % len(lr_multipliers)], 8)
            trial["weight_decay"] = round(float(trial["weight_decay"]) * wd_multipliers[(cycle - 1) % len(wd_multipliers)], 8)
            trial["epochs"] = int(trial["epochs"]) + 5 * min(cycle, 3)
            trial["hypothesis"] = (
                f"{scenario['hypothesis']} Controlled follow-up cycle {cycle}: adjust LR/WD only, keeping split, "
                "checkpoint, model, and the main hypothesis fixed."
            )
        else:
            trial["hypothesis"] = scenario["hypothesis"]
        trial["initial_checkpoint_path"] = initial_checkpoint
        trial["decision_context"] = {
            "best": summarize_record(best_record),
            "last_result": last,
            "rule": "precision_recall_adaptive",
        }
        return trial

    def trial_from_record(self, record: dict[str, Any] | None, *, initial_checkpoint: str | None) -> dict[str, Any]:
        train = (record or {}).get("train") or {}
        preprocess = (record or {}).get("preprocess") or {}
        loss_cfg = train.get("loss") or {}
        scheduler_cfg = train.get("scheduler") or {}
        augmentations = train.get("augmentations") or {
            "flips": True,
            "rot90": True,
            "brightness_contrast": True,
            "noise": True,
            "blur": False,
            "gamma": False,
        }
        return {
            "hypothesis": "Fine-tune from the current best checkpoint.",
            "model_name": str(((record or {}).get("config") or {}).get("model", {}).get("name") or "segformer_b2"),
            "patch_size": int(preprocess.get("patch_size") or preprocess.get("tile_size") or 1024),
            "stride": int(preprocess.get("stride") or 512),
            "batch_size": train.get("batch_size") or 2,
            "learning_rate": float(train.get("learning_rate") or 1e-4),
            "weight_decay": float(train.get("weight_decay") or 1e-4),
            "loss": str(loss_cfg.get("name") or loss_cfg.get("type") or "focal_dice"),
            "scheduler": str(scheduler_cfg.get("name") or scheduler_cfg.get("type") or "none"),
            "epochs": int(train.get("epochs") or train.get("max_epochs") or 50),
            "early_stopping_patience": int((train.get("early_stopping") or {}).get("patience") or 14),
            "split_seed": int(preprocess.get("split_seed") or 20260513),
            "initial_checkpoint_path": initial_checkpoint,
            "target_val_fraction": float(preprocess.get("target_val_fraction") or 0.2),
            "split_strategy": str(preprocess.get("split_strategy") or "object_balanced"),
            "augmentations": dict(augmentations),
        }

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
        record = read_run_record(Path(self.args.status_root), run_id)
        metrics = dict((record or {}).get("metrics") or {})
        if record and record.get("checkpoint_path"):
            metrics["checkpoint_path"] = record.get("checkpoint_path")
        mlflow = summary.get("mlflow") or {}
        result = {
            "status": status,
            "run_id": run_id,
            "mlflow_run_id": (record or {}).get("mlflow_run_id") or mlflow.get("run_id"),
            "mlflow_run_url": (record or {}).get("mlflow_run_url")
            or mlflow.get("run_url_external")
            or mlflow.get("external_run_url")
            or mlflow.get("run_url"),
            "metrics": metrics,
            "analysis": self.analyze_result(record),
            "finished_at": utc_now(),
        }
        self.log(f"run done run={run_id} status={status} f1={metrics.get('best_val_pixel_f1')} epoch={metrics.get('best_epoch')}")
        return result

    def analyze_result(self, record: dict[str, Any] | None) -> dict[str, Any]:
        if not record:
            return {"status": "missing_record", "decision": "Do not use this run as a baseline."}
        metrics = record.get("metrics") or {}
        precision = as_float(metrics.get("precision"))
        recall = as_float(metrics.get("recall"))
        f1 = as_float(metrics.get("best_val_pixel_f1"))
        last_f1 = as_float(metrics.get("last_val_pixel_f1"))
        findings: list[str] = []
        next_focus = "balanced"
        if record.get("status") != "success":
            findings.append("Run failed; keep current best checkpoint and avoid treating this config as a valid branch.")
            next_focus = "pipeline_or_config_recovery"
        elif precision is not None and recall is not None and precision > recall + 0.08:
            findings.append("Precision is materially higher than recall; next experiment should target recall and false negatives.")
            next_focus = "increase_recall"
        elif precision is not None and recall is not None and recall > precision + 0.08:
            findings.append("Recall is materially higher than precision; next experiment should reduce false positives.")
            next_focus = "increase_precision"
        if f1 is not None and last_f1 is not None and f1 - last_f1 > 0.03:
            findings.append("Best F1 is much higher than last F1; treat as unstable/overfit and prefer more conservative continuation.")
            next_focus = "stability"
        if not findings:
            findings.append("Run is usable and reasonably balanced; continue with small controlled mutations.")
        return {
            "findings": findings,
            "next_focus": next_focus,
            "summary": summarize_record(record),
        }

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
                row_run_id = str(row.get("run_id") or row.get("dag_run_id") or "")
                if row_run_id == run_id:
                    return str(row.get("state") or "").lower() or None
        except Exception:
            return None
        return None

    def append_ledger(self, trial: dict[str, Any], result: dict[str, Any]) -> None:
        first = not self.ledger_path.exists()
        with self.ledger_path.open("a", encoding="utf-8") as handle:
            if first:
                handle.write(f"# {self.class_name} tuning ledger\n\n")
            metrics = result.get("metrics") or {}
            handle.write(f"## {utc_now()} {trial['run_id']}\n\n")
            handle.write(f"- status: {result.get('status')}\n")
            handle.write(f"- hypothesis: {trial['config']['hypothesis']}\n")
            handle.write(f"- config_hash: `{trial['config_hash']}`\n")
            handle.write(f"- initial_checkpoint: `{trial['config'].get('initial_checkpoint_path')}`\n")
            handle.write(f"- mlflow_run_id: `{result.get('mlflow_run_id')}`\n")
            handle.write(f"- best_val_pixel_f1: {metrics.get('best_val_pixel_f1')}\n")
            handle.write(f"- last_val_pixel_f1: {metrics.get('last_val_pixel_f1')}\n")
            handle.write(f"- precision: {metrics.get('precision')}\n")
            handle.write(f"- recall: {metrics.get('recall')}\n")
            handle.write(f"- iou: {metrics.get('iou')}\n")
            handle.write(f"- best_epoch: {metrics.get('best_epoch')}\n")
            handle.write(f"- checkpoint: `{metrics.get('checkpoint_path')}`\n\n")
            analysis = result.get("analysis") or {}
            findings = analysis.get("findings") or []
            if findings:
                handle.write("Analysis:\n")
                for finding in findings:
                    handle.write(f"- {finding}\n")
                handle.write(f"- next_focus: `{analysis.get('next_focus')}`\n\n")

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
    if args.class_slug not in {"lakes", "desertification"}:
        raise SystemExit("This controller currently supports class_slug values: lakes, desertification.")
    AirflowTuningController(args).run_forever()


if __name__ == "__main__":
    main()
