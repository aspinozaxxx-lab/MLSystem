from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..job_schema import JobSpec
from ..mlflow_adapter import MLFLOW_EXCLUDED_ARTIFACT_NAMES, MLflowJobRun
from ..pipeline_config import load_config
from ..s3_adapter import build_s3_layout_status
from ..storage.local_io import read_json, write_json
from ..storage.s3 import find_layout_files, list_s3_objects, read_s3_text
from ..data.scene_matching import build_scene_matching_report
from .prediction_pipeline import PredictionPipeline
from .training_pipeline import TrainingPipeline


MAIN_DAG_STAGES = [
    "validate_experiment_config",
    "check_s3_layout",
    "match_scenes",
    "validate_scene_matching",
    "inventory_images",
    "prepare_dataset_manifest",
    "prepare_train_tiles_or_windows",
    "validate_dataset",
    "create_mlflow_run",
    "train_model",
    "evaluate_pixel_metrics",
    "predict_validation_scenes",
    "vectorize_validation_predictions",
    "compute_object_f1",
    "predict_pseudolabel_scenes",
    "stitch_probability_maps",
    "vectorize_pseudolabel",
    "postprocess_pseudolabel",
    "export_pseudolabel_artifacts",
    "generate_prediction_examples",
    "log_mlflow_artifacts",
    "write_codex_api_summary",
    "finalize_mlflow_run",
]

CLI_STAGE_ALIASES = {
    "validate-config": "validate_experiment_config",
    "match-scenes": "match_scenes",
    "prepare-dataset": "prepare_dataset_manifest",
    "train": "train_model",
    "evaluate": "evaluate_pixel_metrics",
    "pseudolabel": "predict_pseudolabel_scenes",
    "postprocess": "postprocess_pseudolabel",
    "finalize": "finalize_mlflow_run",
}


class AirflowExperimentConfig(BaseModel):
    experiment_id: str
    class_name: str | None = None
    task: str = "train_predict_pseudolabel"
    smoke: bool = False
    images_uri: str = "s3://mlsystems/images/"
    layout_uri: str = "s3://mlsystems/layouts/deforest/"
    scenes_file: str = "scenes.txt"
    annotation_file: str = "auto"
    model: dict[str, Any] = Field(default_factory=dict)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    train: dict[str, Any] = Field(default_factory=dict)
    pseudolabel: dict[str, Any] = Field(default_factory=dict)
    postprocess: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    predict: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    mlflow: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if not value or not re.match(r"^[A-Za-z0-9._-]+$", value):
            raise ValueError("experiment_id may contain only letters, digits, dot, underscore and dash")
        return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "run"


def load_conf(conf_file: str | None = None, conf_json: str | None = None) -> dict[str, Any]:
    if conf_json:
        return json.loads(conf_json)
    if conf_file:
        return json.loads(Path(conf_file).read_text(encoding="utf-8-sig"))
    return {}


class AirflowRunStore:
    def __init__(self, state_dir: Path, airflow_run_id: str, conf: dict[str, Any]) -> None:
        self.state_dir = state_dir
        self.airflow_run_id = airflow_run_id
        self.conf = conf
        self.experiment_id = str(conf.get("experiment_id") or safe_run_id(airflow_run_id))
        self.run_dir = state_dir / self.experiment_id
        self.stage_dir = self.run_dir / "stages"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    def read_summary(self) -> dict[str, Any]:
        return read_json(self.summary_path, default={}) or {}

    def update_summary(self, **updates: Any) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "airflow_run_id": self.airflow_run_id,
            "updated_at": utc_now(),
            "stages": {},
            "warnings": [],
            "errors": [],
            **self.read_summary(),
        }
        payload.update(updates)
        payload["updated_at"] = utc_now()
        write_json(self.summary_path, payload)
        return payload

    def write_stage(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        stage_payload = {"stage": stage, "finished_at": utc_now(), **payload}
        write_json(self.stage_dir / f"{stage}.json", stage_payload)
        summary = self.read_summary()
        stages = summary.get("stages") or {}
        stages[stage] = {
            "status": stage_payload.get("status"),
            "finished_at": stage_payload["finished_at"],
            "duration_sec": stage_payload.get("duration_sec"),
            "summary": stage_payload.get("summary"),
        }
        summary["stages"] = stages
        summary["current_stage"] = stage
        summary["updated_at"] = utc_now()
        if stage_payload.get("warnings"):
            summary["warnings"] = sorted(set((summary.get("warnings") or []) + stage_payload["warnings"]))
        if stage_payload.get("error"):
            summary["errors"] = (summary.get("errors") or []) + [stage_payload["error"]]
        write_json(self.summary_path, summary)
        return stage_payload


def _stage_result(status: str = "success", **payload: Any) -> dict[str, Any]:
    return {"status": status, **payload}


def _resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {"captured_at": utc_now()}
    try:
        load1, load5, load15 = os.getloadavg()
        cpu_count = os.cpu_count() or 1
        snapshot["cpu"] = {
            "load1": round(load1, 3),
            "load5": round(load5, 3),
            "load15": round(load15, 3),
            "cpu_count": cpu_count,
            "load1_pct_of_cores": round(100.0 * load1 / cpu_count, 2),
        }
    except Exception as exc:
        snapshot["cpu_error"] = f"{type(exc).__name__}: {exc}"
    try:
        meminfo: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            parts = value.strip().split()
            if parts:
                meminfo[key] = int(parts[0])
        total_kb = meminfo.get("MemTotal") or 0
        available_kb = meminfo.get("MemAvailable") or 0
        snapshot["memory"] = {
            "total_mb": round(total_kb / 1024, 1),
            "available_mb": round(available_kb / 1024, 1),
            "used_pct": round(100.0 * (total_kb - available_kb) / total_kb, 2) if total_kb else None,
        }
    except Exception as exc:
        snapshot["memory_error"] = f"{type(exc).__name__}: {exc}"
    try:
        query = (
            "index,name,utilization.gpu,utilization.memory,memory.used,memory.total,"
            "power.draw,temperature.gpu"
        )
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            gpus = []
            for row in proc.stdout.splitlines():
                parts = [part.strip() for part in row.split(",")]
                if len(parts) >= 8:
                    gpus.append(
                        {
                            "index": parts[0],
                            "name": parts[1],
                            "gpu_util_pct": float(parts[2]),
                            "memory_util_pct": float(parts[3]),
                            "memory_used_mb": float(parts[4]),
                            "memory_total_mb": float(parts[5]),
                            "power_w": None if parts[6] in {"[N/A]", "N/A"} else float(parts[6]),
                            "temperature_c": None if parts[7] in {"[N/A]", "N/A"} else float(parts[7]),
                        }
                    )
            snapshot["gpu"] = gpus
        else:
            snapshot["gpu_error"] = proc.stderr.strip() or proc.stdout.strip()
    except Exception as exc:
        snapshot["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


class _StageResourceMonitor:
    def __init__(self, path: Path, *, stage: str, interval_sec: float = 30.0) -> None:
        self.path = path
        self.stage = stage
        self.interval_sec = interval_sec
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"resource-monitor-{stage}", daemon=True)

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=2)
        payload = {"stage": self.stage, "finished_at": utc_now(), "samples": self.samples[-240:]}
        write_json(self.path, payload)
        return payload

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _resource_snapshot()
            self.samples.append(sample)
            if len(self.samples) > 240:
                self.samples = self.samples[-240:]
            payload = {"stage": self.stage, "updated_at": utc_now(), "latest": sample, "samples": self.samples}
            try:
                write_json(self.path, payload)
                gpu_parts = []
                for gpu in sample.get("gpu") or []:
                    gpu_parts.append(
                        f"{gpu.get('name')} util={gpu.get('gpu_util_pct')}% mem={gpu.get('memory_used_mb')}/{gpu.get('memory_total_mb')}MB"
                    )
                cpu = sample.get("cpu") or {}
                memory = sample.get("memory") or {}
                print(
                    "[resource] "
                    f"stage={self.stage} cpu_load1={cpu.get('load1')} "
                    f"cpu_load1_pct={cpu.get('load1_pct_of_cores')} "
                    f"mem_used_pct={memory.get('used_pct')} "
                    f"gpu={' | '.join(gpu_parts) if gpu_parts else sample.get('gpu_error', 'none')}",
                    flush=True,
                )
            except Exception:
                pass
            self._stop.wait(self.interval_sec)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{utc_now()} {message}\n")


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _existing_artifacts(store: AirflowRunStore) -> dict[str, str]:
    names = [
        "train_scenes.txt",
        "val_scenes.txt",
        "pseudolabel_scenes.txt",
        f"{store.experiment_id}.accepted.geojson",
        "object_metrics.json",
        "coverage_report.json",
        "pseudolabel_summary.json",
        "prediction_examples.html",
        "run_summary.json",
        "codex_summary.json",
        "history.json",
        "history.csv",
        "train_dataset_report.json",
    ]
    return {name: _safe_rel(store.run_dir / name, store.run_dir) for name in names if (store.run_dir / name).exists()}


def _forbidden_logged_artifacts(store: AirflowRunStore) -> list[str]:
    return sorted(name for name in MLFLOW_EXCLUDED_ARTIFACT_NAMES if (store.run_dir / name).exists())


def _cleanup_runtime_intermediates(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    cleanup_cfg = conf.pseudolabel.get("cleanup_intermediates", True)
    if cleanup_cfg is False or conf.pseudolabel.get("keep_intermediates"):
        return {"enabled": False, "reason": "disabled_by_config"}
    allowed_names = {"pseudolabel_scene_results", "vectorization_work"}
    deleted_dirs: list[dict[str, Any]] = []
    deleted_bytes = 0
    deleted_files = 0
    run_root = store.run_dir.resolve()
    for name in sorted(allowed_names):
        target = (store.run_dir / name).resolve()
        if not target.exists():
            continue
        if run_root not in target.parents or target.name not in allowed_names:
            continue
        file_count = 0
        total_size = 0
        for path in target.rglob("*"):
            if path.is_file():
                try:
                    total_size += path.stat().st_size
                    file_count += 1
                except FileNotFoundError:
                    pass
        shutil.rmtree(target)
        deleted_files += file_count
        deleted_bytes += total_size
        deleted_dirs.append({"path": _safe_rel(target, store.run_dir), "files": file_count, "bytes": total_size})
    return {
        "enabled": True,
        "deleted_dirs": deleted_dirs,
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "deleted_gb": round(deleted_bytes / 1024**3, 3),
    }


def _read_training_result(store: AirflowRunStore) -> dict[str, Any]:
    summary = store.read_summary()
    return summary.get("training_result") or read_json(store.run_dir / "training_result.json", default={}) or {}


def _build_airflow_job(conf: AirflowExperimentConfig) -> JobSpec:
    train_cfg = dict(conf.train or {})
    preprocess_cfg = dict(conf.preprocess or {})
    model_cfg = dict(conf.model or {})
    pseudolabel_cfg = dict(conf.pseudolabel or {})
    predict_cfg = dict(conf.predict or {})
    params_cfg = dict(conf.params or {})
    if "tile_size" in preprocess_cfg and "patch_size" not in train_cfg:
        train_cfg["patch_size"] = preprocess_cfg["tile_size"]
    if "max_epochs" in train_cfg and "epochs" not in train_cfg:
        train_cfg["epochs"] = train_cfg["max_epochs"]
    if "max_train_tiles" in preprocess_cfg and "max_train_tiles" not in train_cfg:
        train_cfg["max_train_tiles"] = preprocess_cfg["max_train_tiles"]
    if "max_val_tiles" in preprocess_cfg and "max_val_tiles" not in train_cfg:
        train_cfg["max_val_tiles"] = preprocess_cfg["max_val_tiles"]
    if isinstance(train_cfg.get("early_stopping"), bool):
        train_cfg["early_stopping"] = {
            "enabled": bool(train_cfg["early_stopping"]),
            "patience": train_cfg.get("early_stopping_patience", 10),
        }
    train_cfg.setdefault("require_gpu", True)
    train_cfg.setdefault("allow_train_val_sample_fallback", True)
    train_cfg.setdefault("model", model_cfg)
    train_cfg.setdefault("model_name", model_cfg.get("name") or "tiny_unet_4ch")
    if conf.inference:
        params_cfg.setdefault("inference", dict(conf.inference))
        predict_cfg.setdefault("inference", dict(conf.inference))
    params_cfg.update(
        {
            "model": model_cfg,
            "input_bands": model_cfg.get("input_bands") or [1, 2, 3, 4],
            "pseudolabel": pseudolabel_cfg,
        }
    )
    predict_cfg.update({"pseudolabel": pseudolabel_cfg, "model": model_cfg})
    return JobSpec(
        job_id=conf.experiment_id,
        task=conf.task,
        class_name=conf.class_name,
        description="Airflow experiment run",
        params=params_cfg,
        data={
            "images_uri": conf.images_uri,
            "layout_uri": conf.layout_uri,
            "scenes_file": conf.scenes_file,
            "annotation_file": conf.annotation_file,
        },
        preprocess=preprocess_cfg,
        train=train_cfg,
        predict=predict_cfg,
        postprocess=conf.postprocess or {},
        resources={"requires_gpu": bool(train_cfg.get("require_gpu", True))},
        mlflow={"experiment": conf.mlflow.get("experiment")} if conf.mlflow.get("experiment") else {},
    )


def _run_training_pipeline(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    summary = store.read_summary()
    mlflow_info = summary.get("mlflow") or {}
    run_id = mlflow_info.get("run_id")
    pipeline_config = load_config()
    experiment_name = conf.mlflow.get("experiment") or pipeline_config.mlflow_default_experiment
    train_pseudolabel = dict(conf.pseudolabel or {})
    train_pseudolabel["enabled"] = False
    train_only_conf = conf.model_copy(update={"pseudolabel": train_pseudolabel})
    job = _build_airflow_job(train_only_conf)
    job_log = store.run_dir / "airflow_train.log"
    tags = {
        "job_id": conf.experiment_id,
        "airflow_run_id": store.airflow_run_id,
        "orchestrator": "airflow",
        "queue_state": "airflow",
        "task": conf.task,
        "class_name": conf.class_name or "",
        "mlsystem.class_name": conf.class_name or "",
    }
    params = {
        "experiment_id": conf.experiment_id,
        "task": conf.task,
        "model.name": conf.model.get("name"),
        "preprocess.tile_size": conf.preprocess.get("tile_size"),
        "preprocess.max_scenes": conf.preprocess.get("max_scenes"),
        "train.epochs": job.train.get("epochs"),
        "train.time_limit_sec": job.train.get("time_limit_sec"),
    }
    with MLflowJobRun(
        pipeline_config,
        experiment_name=experiment_name,
        run_name=conf.experiment_id,
        params=params,
        tags=tags,
        run_id=run_id,
    ) as mlflow_run:
        result = TrainingPipeline().run(pipeline_config, job, store.run_dir, mlflow_run, job_log, _append_log)
        result["mlflow"] = mlflow_run.result()
        mlflow_run.set_tags({"job_status": "training_completed", "airflow_training_status": "success"})
    write_json(store.run_dir / "training_result.json", result)
    store.update_summary(training_result=result, mlflow=result.get("mlflow") or mlflow_info)
    return result


def _checkpoint_path_for_pseudolabel(conf: AirflowExperimentConfig, store: AirflowRunStore, training_result: dict[str, Any]) -> Path:
    configured_checkpoint = (conf.pseudolabel or {}).get("checkpoint_path") or (conf.predict or {}).get("checkpoint_path") or (conf.params or {}).get("checkpoint_path")
    if configured_checkpoint:
        path = Path(str(configured_checkpoint))
        if path.exists():
            return path
    checkpoint_path = training_result.get("checkpoint_path")
    if checkpoint_path:
        path = Path(str(checkpoint_path))
        if path.exists():
            return path
    model_name = str(conf.model.get("name") or training_result.get("model_name") or "tiny_unet_4ch")
    fallback = store.run_dir / f"{model_name}.pt"
    if fallback.exists():
        return fallback
    raise RuntimeError(f"Training checkpoint is missing: {checkpoint_path or fallback}")


def _run_pseudolabel_pipeline(conf: AirflowExperimentConfig, store: AirflowRunStore, *, stage_mode: str = "full") -> dict[str, Any]:
    training_result = _read_training_result(store)
    checkpoint_path = _checkpoint_path_for_pseudolabel(conf, store, training_result)
    summary = store.read_summary()
    mlflow_info = summary.get("mlflow") or training_result.get("mlflow") or {}
    run_id = mlflow_info.get("run_id")
    pipeline_config = load_config()
    experiment_name = conf.mlflow.get("experiment") or pipeline_config.mlflow_default_experiment
    job = _build_airflow_job(conf)
    pseudolabel_cfg = dict(job.predict.get("pseudolabel") or {})
    pseudolabel_cfg.setdefault("enabled", True)
    pseudolabel_cfg["checkpoint_path"] = str(checkpoint_path)
    pseudolabel_cfg["preserve_train_scenes"] = True
    pseudolabel_cfg["_airflow_stage_mode"] = stage_mode
    job.predict["pseudolabel"] = pseudolabel_cfg
    job.predict["checkpoint_path"] = str(checkpoint_path)
    job.predict["preserve_train_scenes"] = True
    job.params["pseudolabel"] = pseudolabel_cfg
    job.params["checkpoint_path"] = str(checkpoint_path)
    job_log = store.run_dir / "airflow_pseudolabel.log"
    tags = {
        "job_id": conf.experiment_id,
        "airflow_run_id": store.airflow_run_id,
        "orchestrator": "airflow",
        "queue_state": "airflow",
        "task": conf.task,
        "class_name": conf.class_name or "",
        "mlsystem.class_name": conf.class_name or "",
    }
    with MLflowJobRun(
        pipeline_config,
        experiment_name=experiment_name,
        run_name=conf.experiment_id,
        params={"pseudolabel.checkpoint_path": str(checkpoint_path)},
        tags=tags,
        run_id=run_id,
    ) as mlflow_run:
        prediction_result = PredictionPipeline().run_debug_pseudolabel(
            pipeline_config,
            job,
            store.run_dir,
            mlflow_run,
            job_log,
            _append_log,
        )
        prediction_result["mlflow"] = mlflow_run.result()
        status_tag = "pseudolabel_inference_completed" if stage_mode in {"inference", "infer"} else "pseudolabel_completed"
        mlflow_run.set_tags({"job_status": status_tag, "airflow_pseudolabel_status": "success", "airflow_pseudolabel_stage_mode": stage_mode})

    merged = dict(training_result)
    merged["pseudolabel_result"] = prediction_result
    for key in [
        "postprocess_metrics",
        "pseudolabel",
        "matched_scenes_count",
        "total_matched_scenes_count",
        "missing_scenes",
        "ambiguous_scenes",
        "warnings",
    ]:
        if key in prediction_result:
            merged[key] = prediction_result[key]
    timing = dict(merged.get("timing") or {})
    timing.update({key: value for key, value in (prediction_result.get("timing") or {}).items() if key.endswith("pseudolabel_duration_sec") or key.endswith("postprocess_duration_sec")})
    merged["timing"] = timing
    merged["mlflow"] = prediction_result.get("mlflow") or mlflow_info
    write_json(store.run_dir / "training_result.json", merged)
    store.update_summary(training_result=merged, mlflow=merged.get("mlflow") or mlflow_info)
    return prediction_result


def _write_run_summaries(conf: AirflowExperimentConfig, store: AirflowRunStore) -> tuple[Path, Path]:
    summary = store.read_summary()
    result = _read_training_result(store)
    post_metrics = result.get("postprocess_metrics") or {}
    artifacts = _existing_artifacts(store)
    run_summary = {
        "schema_version": 1,
        "experiment_id": conf.experiment_id,
        "airflow_run_id": store.airflow_run_id,
        "status": summary.get("status", "running"),
        "mlflow": summary.get("mlflow") or result.get("mlflow"),
        "training": {
            "mode": result.get("mode"),
            "model_name": result.get("model_name"),
            "device": result.get("device"),
            "epochs_completed": result.get("epochs_completed"),
            "best_val_iou": result.get("best_val_iou"),
            "last_epoch_metrics": result.get("last_epoch_metrics"),
            "train_scene_count": result.get("train_scene_count"),
            "val_scene_count": result.get("val_scene_count"),
            "train_tile_count": result.get("train_tile_count"),
            "val_tile_count": result.get("val_tile_count"),
        },
        "pseudolabel": result.get("pseudolabel"),
        "object_metrics": {
            key: value
            for key, value in post_metrics.items()
            if key.startswith("val/object_") or key in {"val/tp", "val/fp", "val/fn", "val/gt_count", "val/pred_count"}
        },
        "artifacts": artifacts,
        "excluded_local_artifacts_not_logged": _forbidden_logged_artifacts(store),
        "updated_at": utc_now(),
    }
    codex_summary = {
        "experiment_id": conf.experiment_id,
        "airflow_run_id": store.airflow_run_id,
        "mlflow_run_url": (summary.get("mlflow") or result.get("mlflow") or {}).get("run_url_external")
        or (summary.get("mlflow") or result.get("mlflow") or {}).get("external_run_url"),
        "current_stage": summary.get("current_stage"),
        "epochs_completed": result.get("epochs_completed"),
        "best_val_iou": result.get("best_val_iou"),
        "object_f1": post_metrics.get("val/object_f1"),
        "accepted_objects": post_metrics.get("accepted_objects"),
        "coverage_fraction": post_metrics.get("coverage_fraction"),
        "artifacts": artifacts,
        "warnings": summary.get("warnings") or result.get("warnings") or [],
        "updated_at": utc_now(),
    }
    run_summary_path = store.run_dir / "run_summary.json"
    codex_summary_path = store.run_dir / "codex_summary.json"
    write_json(run_summary_path, run_summary)
    write_json(codex_summary_path, codex_summary)
    return run_summary_path, codex_summary_path


def _smoke_or_skip(conf: AirflowExperimentConfig, stage: str) -> dict[str, Any] | None:
    if not conf.smoke:
        return None
    passthrough = {
        "validate_experiment_config",
        "create_mlflow_run",
        "predict_pseudolabel_scenes",
        "stitch_probability_maps",
        "vectorize_pseudolabel",
        "postprocess_pseudolabel",
        "export_pseudolabel_artifacts",
        "generate_prediction_examples",
        "log_mlflow_artifacts",
        "write_codex_api_summary",
        "finalize_mlflow_run",
    }
    if stage in passthrough:
        return None
    return _stage_result("skipped", summary="Synthetic smoke run skips external S3/dataset/training work.")


def _create_mlflow_run(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    try:
        import mlflow

        pipeline_config = load_config()
        mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
        experiment_name = conf.mlflow.get("experiment") or pipeline_config.mlflow_default_experiment
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment and conf.class_name:
            client = mlflow.tracking.MlflowClient()
            client.set_experiment_tag(experiment.experiment_id, "class_name", conf.class_name)
            client.set_experiment_tag(experiment.experiment_id, "mlsystem.class_name", conf.class_name)
            client.set_experiment_tag(experiment.experiment_id, "task", conf.task)
        with mlflow.start_run(run_name=conf.experiment_id) as run:
            mlflow.set_tags(
                {
                    "job_id": conf.experiment_id,
                    "airflow_run_id": store.airflow_run_id,
                    "orchestrator": "airflow",
                    "job_status": "running",
                    "queue_state": "airflow",
                    "task": conf.task,
                    "class_name": conf.class_name or "",
                    "mlsystem.class_name": conf.class_name or "",
                }
            )
            mlflow.log_params(
                {
                    "experiment_id": conf.experiment_id,
                    "task": conf.task,
                    "images_uri": conf.images_uri,
                    "layout_uri": conf.layout_uri,
                    "model.name": conf.model.get("name"),
                    "preprocess.tile_size": conf.preprocess.get("tile_size"),
                    "preprocess.stride": conf.preprocess.get("stride"),
                    "smoke": conf.smoke,
                }
            )
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
        run_url = f"{pipeline_config.mlflow_tracking_uri_external.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
        store.update_summary(
            mlflow={
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "run_url_external": run_url,
                "tracking_uri": pipeline_config.mlflow_tracking_uri_internal,
            }
        )
        return _stage_result("success", mlflow_run_id=run_id, mlflow_run_url=run_url)
    except Exception as exc:
        if conf.smoke:
            fallback = {"run_id": f"smoke-{conf.experiment_id}", "run_url_external": None, "error": f"{type(exc).__name__}: {exc}"}
            store.update_summary(mlflow=fallback, warnings=[f"MLflow smoke fallback: {fallback['error']}"])
            return _stage_result("success", summary="MLflow unavailable in local smoke; stored fallback metadata.", warnings=[fallback["error"]])
        raise


def _finalize_mlflow_run(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    summary = store.read_summary()
    mlflow_info = summary.get("mlflow") or {}
    run_id = mlflow_info.get("run_id")
    cleanup = _cleanup_runtime_intermediates(conf, store)
    if not run_id or str(run_id).startswith("smoke-"):
        store.update_summary(status="success", finished_at=utc_now(), runtime_cleanup=cleanup)
        return _stage_result("success", summary="No real MLflow run to finalize.", cleanup=cleanup)
    try:
        import mlflow

        pipeline_config = load_config()
        mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
        store.update_summary(status="success", finished_at=utc_now(), runtime_cleanup=cleanup)
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags({"job_status": "success", "airflow_status": "success"})
            mlflow.log_artifact(str(store.summary_path))
        return _stage_result("success", mlflow_run_id=run_id, cleanup=cleanup)
    except Exception as exc:
        return _stage_result("failed", error=f"{type(exc).__name__}: {exc}")


def _run_airflow_synthetic_pseudolabel_smoke(store: AirflowRunStore) -> dict[str, Any]:
    smoke_dir = store.run_dir / "synthetic_pseudolabel"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    accepted_geojson = smoke_dir / f"{store.experiment_id}.accepted.geojson"
    prediction_examples = smoke_dir / "prediction_examples.html"
    smoke_summary = {
        "status": "success",
        "mode": "airflow_synthetic",
        "coverage": {"width": 16, "height": 16, "covered_pixels": 256, "missing_pixels": 0},
        "objects": {"accepted": 1, "rejected": 0},
    }
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"score": 0.9, "source": "airflow_synthetic_smoke"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[2, 2], [14, 2], [14, 14], [2, 14], [2, 2]]],
                },
            }
        ],
    }
    accepted_geojson.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_examples.write_text(
        "<!doctype html><html><body><h1>MLSystem Airflow smoke</h1><p>Synthetic pseudolabel smoke completed.</p></body></html>",
        encoding="utf-8",
    )
    write_json(smoke_dir / "summary.json", smoke_summary)
    return {
        "summary": "Synthetic pseudolabel smoke completed without heavy ML dependencies.",
        "accepted_geojson": str(accepted_geojson),
        "prediction_examples_html": str(prediction_examples),
        "smoke_summary": smoke_summary,
    }


def run_stage(stage: str, conf_payload: dict[str, Any], airflow_run_id: str, state_dir: Path) -> dict[str, Any]:
    started = time.time()
    resources_before = _resource_snapshot()
    conf = AirflowExperimentConfig.model_validate(conf_payload)
    store = AirflowRunStore(state_dir, airflow_run_id, conf.model_dump())
    store.update_summary(
        status="running",
        experiment_config=conf.model_dump(),
        airflow={"dag_id": "mlsystem_experiment_pipeline", "run_id": airflow_run_id},
        active_stage=stage,
        active_stage_started_at=utc_now(),
        active_stage_resources=resources_before,
    )
    monitor = _StageResourceMonitor(store.stage_dir / f"{stage}.resources.json", stage=stage)
    monitor.start()

    smoke_result = _smoke_or_skip(conf, stage)
    if smoke_result is not None:
        smoke_result["duration_sec"] = round(time.time() - started, 3)
        monitor_payload = monitor.stop()
        smoke_result["resources"] = {"before": resources_before, "after": _resource_snapshot(), "monitor": _safe_rel(store.stage_dir / f"{stage}.resources.json", store.run_dir), "sample_count": len(monitor_payload.get("samples") or [])}
        return store.write_stage(stage, smoke_result)

    if stage == "validate_experiment_config":
        result = _stage_result("success", summary=f"Config valid for {conf.experiment_id}")
    elif stage == "check_s3_layout":
        result = _stage_result("success", s3_layout=build_s3_layout_status(load_config()))
    elif stage == "match_scenes":
        pipeline_config = load_config()
        images = list_s3_objects(pipeline_config, conf.images_uri, suffixes=(".tif", ".tiff"))
        annotation_uri, scenes_uri = find_layout_files(pipeline_config, conf.layout_uri, conf.scenes_file, conf.annotation_file)
        entries = [line.strip() for line in read_s3_text(pipeline_config, scenes_uri).splitlines() if line.strip() and not line.strip().startswith("#")]
        preferred_prefixes = list(conf.preprocess.get("scene_matching_prefer_prefixes") or [])
        report = {
            **build_scene_matching_report(entries, images, preferred_key_prefixes=preferred_prefixes),
            "images_uri": conf.images_uri,
            "layout_uri": conf.layout_uri,
            "annotation_uri": annotation_uri,
            "scenes_uri": scenes_uri,
        }
        write_json(store.run_dir / "scene_matching_report.json", report)
        store.update_summary(scene_matching=report)
        result = _stage_result("success", matched_count=report.get("matched_count"), missing_count=report.get("missing_count"), ambiguous_count=report.get("ambiguous_count"))
    elif stage == "validate_scene_matching":
        report = store.read_summary().get("scene_matching") or {}
        if report.get("matched_count", 0) <= 0:
            result = _stage_result("failed", error="No scenes matched.")
        else:
            warnings = []
            if report.get("missing_count", 0):
                warnings.append(f"{report['missing_count']} scenes missing")
            if report.get("ambiguous_count", 0):
                warnings.append(f"{report['ambiguous_count']} scenes ambiguous")
            result = _stage_result("success", warnings=warnings)
    elif stage == "inventory_images":
        result = _stage_result("success", summary="Inventory is represented by S3 scene matching inputs in this Airflow step.")
    elif stage == "prepare_dataset_manifest":
        matching = store.read_summary().get("scene_matching") or {}
        matched = matching.get("matched") or []
        max_scenes = conf.preprocess.get("max_scenes")
        if max_scenes is not None:
            matched = matched[: max(1, int(max_scenes))]
        split_idx = max(1, int(len(matched) * 0.75)) if matched else 0
        train_scenes = matched[:split_idx]
        val_scenes = matched[split_idx:] or matched[-1:]
        manifest = {
            "experiment_id": conf.experiment_id,
            "created_at": utc_now(),
            "source": "airflow",
            "scene_matching": matching,
            "selected_scene_count": len(matched),
            "train_scene_count": len(train_scenes),
            "val_scene_count": len(val_scenes),
            "train_scenes": train_scenes,
            "val_scenes": val_scenes,
            "limits": {
                "max_scenes": max_scenes,
                "max_train_tiles": conf.preprocess.get("max_train_tiles"),
                "max_val_tiles": conf.preprocess.get("max_val_tiles"),
            },
        }
        write_json(store.run_dir / "dataset_manifest.json", manifest)
        result = _stage_result(
            "success",
            manifest_path=str(store.run_dir / "dataset_manifest.json"),
            train_scene_count=len(train_scenes),
            val_scene_count=len(val_scenes),
        )
    elif stage == "prepare_train_tiles_or_windows":
        result = _stage_result(
            "success",
            summary="Tile/window preparation will be executed by real_train sampling and SceneInferenceRunner.",
            tile_size=conf.preprocess.get("tile_size"),
            stride=conf.preprocess.get("stride"),
            context=conf.preprocess.get("context"),
        )
    elif stage == "validate_dataset":
        manifest = read_json(store.run_dir / "dataset_manifest.json", default={}) or {}
        if int(manifest.get("train_scene_count") or 0) <= 0 or int(manifest.get("val_scene_count") or 0) <= 0:
            result = _stage_result("failed", error="Dataset manifest has no train or validation scenes.")
        else:
            result = _stage_result(
                "success",
                train_scene_count=manifest.get("train_scene_count"),
                val_scene_count=manifest.get("val_scene_count"),
            )
    elif stage == "create_mlflow_run":
        result = _create_mlflow_run(conf, store)
    elif stage == "train_model":
        if not conf.train.get("enabled", True):
            result = _stage_result("skipped", summary="train.enabled=false")
        else:
            training_result = _run_training_pipeline(conf, store)
            result = _stage_result(
                "success",
                summary="TrainingPipeline completed real MLSystem train-only run; pseudolabel is handled by predict_pseudolabel_scenes.",
                mode=training_result.get("mode"),
                epochs_completed=training_result.get("epochs_completed"),
                best_val_iou=training_result.get("best_val_iou"),
                checkpoint_path=training_result.get("checkpoint_path"),
                artifacts=_existing_artifacts(store),
            )
    elif stage == "evaluate_pixel_metrics":
        if not conf.train.get("enabled", True):
            result = _stage_result("skipped", summary="train.enabled=false")
        else:
            training_result = _read_training_result(store)
            last_metrics = training_result.get("last_epoch_metrics") or {}
            if not last_metrics:
                result = _stage_result("failed", error="No pixel metrics found in training_result.json.")
            else:
                result = _stage_result(
                    "success",
                    train_loss=last_metrics.get("train/loss"),
                    val_pixel_iou=last_metrics.get("val/iou"),
                    val_pixel_dice=last_metrics.get("val/dice"),
                    val_pixel_f1=last_metrics.get("val/pixel_f1"),
                )
    elif stage == "predict_validation_scenes":
        training_result = _read_training_result(store)
        configured_checkpoint = (conf.pseudolabel or {}).get("checkpoint_path") or (conf.predict or {}).get("checkpoint_path") or (conf.params or {}).get("checkpoint_path")
        if not training_result.get("checkpoint_path") and not configured_checkpoint:
            result = _stage_result("failed", error="Training checkpoint is missing; validation prediction cannot start.")
        else:
            result = _stage_result(
                "success",
                summary="Validation prediction is included in the pseudolabel stage for the current compatibility pipeline.",
                checkpoint_path=training_result.get("checkpoint_path"),
            )
    elif stage == "vectorize_validation_predictions":
        summary_path = store.run_dir / "pseudolabel_summary.json"
        if not summary_path.exists():
            result = _stage_result(
                "success",
                summary="Deferred: vectorization is executed in predict_pseudolabel_scenes by the current compatibility pipeline.",
            )
        else:
            summary = read_json(summary_path, default={}) or {}
            result = _stage_result(
                "success",
                accepted_objects=((summary.get("metrics") or {}).get("accepted_objects")),
                threshold_used=((summary.get("metrics") or {}).get("threshold_used")),
            )
    elif stage == "compute_object_f1":
        training_result = _read_training_result(store)
        metrics = training_result.get("postprocess_metrics") or {}
        object_keys = {
            key: metrics.get(key)
            for key in ["val/object_f1", "val/object_precision", "val/object_recall", "val/tp", "val/fp", "val/fn"]
            if key in metrics
        }
        if "val/object_f1" not in object_keys:
            result = _stage_result(
                "success",
                summary="Deferred: object F1 is computed after pseudolabel vectorization by the current compatibility pipeline.",
            )
        else:
            result = _stage_result("success", **object_keys)
    elif stage == "predict_pseudolabel_scenes":
        if conf.smoke:
            smoke = _run_airflow_synthetic_pseudolabel_smoke(store)
            store.update_summary(pseudolabel_smoke=smoke)
            result = _stage_result("success", **smoke)
        elif not conf.pseudolabel.get("enabled", False):
            result = _stage_result("skipped", summary="pseudolabel.enabled=false")
        else:
            prediction_result = _run_pseudolabel_pipeline(conf, store, stage_mode="inference")
            coverage = read_json(store.run_dir / "coverage_report.json", default={}) or {}
            pseudolabel = prediction_result.get("pseudolabel") or {}
            result = _stage_result(
                "success",
                summary="PredictionPipeline completed real GPU pseudolabel inference and wrote per-scene probability maps.",
                scenes_processed=coverage.get("scenes_processed"),
                total_predicted_windows=coverage.get("total_predicted_windows"),
                mean_coverage_fraction=coverage.get("mean_coverage_fraction"),
                accepted_objects=pseudolabel.get("accepted_objects"),
                accepted_geojson_mb=pseudolabel.get("accepted_geojson_mb"),
                scene_results_manifest=str(store.run_dir / "pseudolabel_scene_results_manifest.json"),
            )
    elif stage == "stitch_probability_maps":
        coverage = read_json(store.run_dir / "coverage_report.json", default={}) or {}
        if not coverage:
            result = _stage_result("failed", error="coverage_report.json is missing.")
        else:
            result = _stage_result(
                "success",
                mean_coverage_fraction=coverage.get("mean_coverage_fraction"),
                min_coverage_fraction=coverage.get("min_coverage_fraction"),
                total_expected_windows=coverage.get("total_expected_windows"),
                total_predicted_windows=coverage.get("total_predicted_windows"),
            )
    elif stage == "vectorize_pseudolabel":
        accepted = store.run_dir / f"{conf.experiment_id}.accepted.geojson"
        if not accepted.exists():
            if not conf.pseudolabel.get("enabled", False):
                result = _stage_result("skipped", summary="pseudolabel.enabled=false")
            else:
                prediction_result = _run_pseudolabel_pipeline(conf, store, stage_mode="postprocess")
                coverage = read_json(store.run_dir / "coverage_report.json", default={}) or {}
                pseudolabel = prediction_result.get("pseudolabel") or {}
                result = _stage_result(
                    "success",
                    summary="CPU vectorization/postprocess completed from saved probability maps.",
                    scenes_processed=coverage.get("scenes_processed"),
                    accepted_objects=pseudolabel.get("accepted_objects"),
                    accepted_geojson_mb=pseudolabel.get("accepted_geojson_mb"),
                    accepted_geojson=str(accepted),
                )
        else:
            result = _stage_result("success", accepted_geojson=str(accepted), size_bytes=accepted.stat().st_size)
    elif stage == "postprocess_pseudolabel":
        training_result = _read_training_result(store)
        metrics = training_result.get("postprocess_metrics") or {}
        if not metrics:
            result = _stage_result("failed", error="No postprocess metrics found.")
        else:
            result = _stage_result(
                "success",
                accepted_objects=metrics.get("accepted_objects"),
                max_objects=conf.postprocess.get("max_objects"),
                threshold_used=metrics.get("threshold_used"),
                min_object_area_m2_used=metrics.get("min_object_area_m2_used"),
                simplify_tolerance_m_used=metrics.get("simplify_tolerance_m_used"),
            )
    elif stage == "export_pseudolabel_artifacts":
        artifacts = _existing_artifacts(store)
        required = ["pseudolabel_scenes.txt", f"{conf.experiment_id}.accepted.geojson", "coverage_report.json", "pseudolabel_summary.json"]
        missing_required = [name for name in required if name not in artifacts]
        if missing_required:
            result = _stage_result("failed", error=f"Missing pseudolabel artifacts: {missing_required}")
        else:
            result = _stage_result(
                "success",
                artifacts={name: artifacts[name] for name in required},
                excluded_local_artifacts_not_logged=_forbidden_logged_artifacts(store),
            )
    elif stage == "generate_prediction_examples":
        examples = store.run_dir / "prediction_examples.html"
        if not examples.exists():
            result = _stage_result("failed", error="prediction_examples.html is missing.")
        else:
            result = _stage_result("success", prediction_examples_html=str(examples), size_bytes=examples.stat().st_size)
    elif stage == "log_mlflow_artifacts":
        run_summary_path, codex_summary_path = _write_run_summaries(conf, store)
        mlflow_info = (store.read_summary().get("mlflow") or _read_training_result(store).get("mlflow") or {})
        run_id = mlflow_info.get("run_id")
        if run_id and not str(run_id).startswith("smoke-"):
            import mlflow

            pipeline_config = load_config()
            mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(str(run_summary_path))
                mlflow.log_artifact(str(codex_summary_path))
        result = _stage_result("success", artifacts=_existing_artifacts(store))
    elif stage == "write_codex_api_summary":
        run_summary_path, codex_summary_path = _write_run_summaries(conf, store)
        result = _stage_result("success", run_summary=str(run_summary_path), codex_summary=str(codex_summary_path))
    elif stage == "finalize_mlflow_run":
        result = _finalize_mlflow_run(conf, store)
    else:
        raise ValueError(f"Unknown Airflow MLSystem stage: {stage}")

    result["duration_sec"] = round(time.time() - started, 3)
    monitor_payload = monitor.stop()
    result["resources"] = {"before": resources_before, "after": _resource_snapshot(), "monitor": _safe_rel(store.stage_dir / f"{stage}.resources.json", store.run_dir), "sample_count": len(monitor_payload.get("samples") or [])}
    return store.write_stage(stage, result)


def run_airflow_stage(stage: str, dag_run_conf: dict[str, Any], airflow_run_id: str, state_dir: Path | str) -> dict[str, Any]:
    return run_stage(stage, dag_run_conf, airflow_run_id, Path(state_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="MLSystem Airflow task wrapper")
    sub = parser.add_subparsers(dest="command", required=True)
    command_names = sorted(set([stage.replace("_", "-") for stage in MAIN_DAG_STAGES] + list(CLI_STAGE_ALIASES)))
    for command_name in command_names:
        p = sub.add_parser(command_name)
        p.add_argument("--run-id", required=True)
        p.add_argument("--state-dir", default=os.getenv("MLSYSTEM_AIRFLOW_STATE_DIR", "/data/mlsystem/airflow/status"))
        p.add_argument("--conf-file", default=None)
        p.add_argument("--conf-json", default=None)
    args = parser.parse_args()
    stage = CLI_STAGE_ALIASES.get(args.command, args.command.replace("-", "_"))
    payload = run_stage(stage, load_conf(args.conf_file, args.conf_json), args.run_id, Path(args.state_dir))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
