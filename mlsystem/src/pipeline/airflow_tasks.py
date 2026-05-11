from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..job_schema import JobSpec
from ..mlflow_adapter import MLFLOW_EXCLUDED_ARTIFACT_NAMES, MLflowJobRun
from ..pipeline_config import load_config
from ..storage.local_io import read_json, write_json
from .training_pipeline import TrainingPipeline


MAIN_DAG_STAGES = [
    "inventory_scenes",
    "prepare_dataset",
    "create_mlflow_run",
    "train_model",
    "evaluate_pixel_metrics",
    "predict_validation_scenes",
    "vectorize_validation_predictions",
    "compute_f1",
    "inference_engine_pipeline",
    "generate_prediction_examples",
    "log_mlflow_artifacts",
    "write_codex_api_summary",
    "finalize_mlflow_run",
]

STAGE_POOLS = {
    "inventory_scenes": ("io_light", 1),
    "prepare_dataset": ("cpu_heavy", 2),
    "create_mlflow_run": ("io_light", 1),
    "train_model": ("gpu_training", 1),
    "evaluate_pixel_metrics": ("cpu_light", 1),
    "predict_validation_scenes": ("gpu_inference", 1),
    "vectorize_validation_predictions": ("cpu_heavy", 2),
    "compute_f1": ("cpu_heavy", 1),
    "inference_engine_pipeline": ("io_light", 1),
    "generate_prediction_examples": ("cpu_heavy", 1),
    "log_mlflow_artifacts": ("io_light", 1),
    "write_codex_api_summary": ("io_light", 1),
    "finalize_mlflow_run": ("io_light", 1),
}

GPU_XCOM_STAGES = {"train_model", "predict_validation_scenes"}

STAGE_XCOM_COUNTERS = {
    "inventory_scenes": {"scene_rows", "available_images", "matched_scenes", "missing_scenes", "ambiguous_scenes"},
    "prepare_dataset": {
        "upstream_inventory_matched_scenes",
        "selected_dataset_scenes",
        "excluded_dataset_scenes",
        "total_scenes",
        "total_objects",
        "scenes_without_objects",
        "train_scenes",
        "train_objects",
        "val_scenes",
        "val_objects",
    },
    "predict_validation_scenes": {
        "validation_input_scenes",
        "validation_scenes_processed",
        "validation_scenes_skipped",
        "validation_scenes_failed",
        "validation_prediction_windows",
        "validation_prediction_tiles",
    },
    "evaluate_pixel_metrics": {
        "pixel_tp",
        "pixel_fp",
        "pixel_fn",
        "pixel_tn",
        "scenes_evaluated",
    },
    "vectorize_validation_predictions": {
        "validation_vectorized_scenes",
        "validation_prediction_files",
        "validation_vectorized_objects",
        "validation_empty_scenes",
        "validation_failed_scenes",
    },
    "compute_f1": {
        "reference_objects",
        "predicted_objects",
        "tp_objects",
        "fp_objects",
        "fn_objects",
        "reference_scenes",
        "prediction_scenes",
    },
    "inference_engine_pipeline": {
        "pseudolabel_scenes_processed",
        "pseudolabel_scenes_skipped",
        "pseudolabel_scenes_failed",
        "pseudolabel_prediction_windows",
        "probability_maps",
        "tiles_total",
        "tiles_done",
        "blocks_total",
        "blocks_done",
        "triton_batches",
        "inference_engine_http_submitted",
    },
    "finalize_mlflow_run": {"failed_stages", "warning_stages"},
    "log_mlflow_artifacts": {"artifacts_logged", "artifacts_failed"},
    "generate_prediction_examples": {"examples_requested", "examples_generated", "examples_failed"},
    "write_codex_api_summary": {"summary_sections"},
}

STAGE_XCOM_METRICS = {
    "evaluate_pixel_metrics": {"pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "pixel_accuracy"},
    "compute_f1": {"pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "object_precision", "object_recall", "object_f1"},
    "inference_engine_pipeline": {"streaming_overlap_sec", "triton_batch_fill_ratio", "triton_request_duration_ms"},
}

STAGE_XCOM_DIRECT_COUNTERS = {
    "prepare_dataset": {"split_strategy"},
    "evaluate_pixel_metrics": {"threshold"},
    "inference_engine_pipeline": {"backend", "source", "request_submitted_via_http", "inference_engine_job_id"},
    "finalize_mlflow_run": {"mlflow_run_id", "mlflow_final_status"},
    "log_mlflow_artifacts": {"mlflow_run_id"},
}

CLI_STAGE_ALIASES = {
    "inventory": "inventory_scenes",
    "prepare-dataset": "prepare_dataset",
    "train": "train_model",
    "evaluate": "evaluate_pixel_metrics",
    "compute-f1": "compute_f1",
    "inference-engine": "inference_engine_pipeline",
    "pseudolabel": "inference_engine_pipeline",
    "finalize": "finalize_mlflow_run",
}

DISPATCHER_STAGE_NAMES = {
    "create_mlflow_run",
    "train_model",
    "evaluate_pixel_metrics",
    "predict_validation_scenes",
    "vectorize_validation_predictions",
    "compute_f1",
    "generate_prediction_examples",
    "log_mlflow_artifacts",
    "write_codex_api_summary",
    "finalize_mlflow_run",
}


class AirflowExperimentConfig(BaseModel):
    schema_version: int | None = None
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
    evaluate: dict[str, Any] = Field(default_factory=dict)
    pseudolabel: dict[str, Any] = Field(default_factory=dict)
    postprocess: dict[str, Any] = Field(default_factory=dict)
    inference: dict[str, Any] = Field(default_factory=dict)
    predict: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    mlflow: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id")
    @classmethod
    def validate_experiment_id(cls, value: str) -> str:
        if not value or not re.match(r"^[A-Za-z0-9._-]+$", value):
            raise ValueError("experiment_id may contain only letters, digits, dot, underscore and dash")
        return value

    @model_validator(mode="after")
    def resolve_annotation_source(self) -> "AirflowExperimentConfig":
        annotations = dict(self.annotations or {})
        if _is_mlmarkup_source(annotations):
            resolved = _resolve_mlmarkup_annotation_config(annotations, self.class_name)
            self.annotations = resolved
            self.layout_uri = resolved["layout_uri"]
            self.scenes_file = resolved["scenes_file"]
            self.annotation_file = resolved["annotation_file"]
            self.pseudolabel = {**(self.pseudolabel or {}), "enabled": False}
            self.params = {
                **(self.params or {}),
                "annotations": resolved,
                "pseudolabeling.enabled": False,
            }
        return self


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "run"


def _is_mlmarkup_source(annotations: dict[str, Any]) -> bool:
    return str(annotations.get("source") or "").strip().lower() in {"mlmarkup", "ml_markup", "ml-markup"}


def _resolve_mlmarkup_annotation_config(annotations: dict[str, Any], class_name: str | None) -> dict[str, Any]:
    repo_path = Path(str(annotations.get("repo_path") or os.getenv("MLSYSTEM_MLMARKUP_REPO_PATH") or "/data/mlsystem/MLMarkup"))
    class_dir = str(annotations.get("class_dir") or annotations.get("folder") or _default_mlmarkup_class_dir(class_name))
    scenes_file = str(annotations.get("scenes_file") or "deforestation.txt")
    annotation_file = str(annotations.get("annotation_file") or "deforestation.geojson")
    commit = str(annotations.get("commit") or _git_output(repo_path, "rev-parse", "HEAD") or "")
    branch = str(annotations.get("branch") or _git_output(repo_path, "branch", "--show-current") or "")
    dirty = bool(_git_output(repo_path, "status", "--short"))
    resolved = dict(annotations)
    resolved.update(
        {
            "source": "MLMarkup",
            "repo_path": str(repo_path),
            "class_dir": class_dir,
            "layout_uri": str(repo_path / class_dir),
            "scenes_file": scenes_file,
            "annotation_file": annotation_file,
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
            "use_pseudolabels": False,
        }
    )
    return resolved


def _default_mlmarkup_class_dir(class_name: str | None) -> str:
    normalized = str(class_name or "").strip().lower()
    if normalized in {"deforest", "cuttings", "clearcuts", "clear_cuts", "вырубки"}:
        return "Вырубки"
    return str(class_name or "").strip() or "Вырубки"


def _git_output(repo_path: Path, *args: str) -> str:
    fallback = _git_metadata_without_binary(repo_path, *args)
    if fallback is not None:
        return fallback
    try:
        git_bin = "/usr/bin/git" if Path("/usr/bin/git").exists() else "git"
        result = subprocess.run(
            [git_bin, "-C", str(repo_path), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except Exception:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_metadata_without_binary(repo_path: Path, *args: str) -> str | None:
    if args == ("status", "--short"):
        return ""
    git_dir = repo_path / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.exists():
        return None
    try:
        head = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if args == ("branch", "--show-current"):
        if head.startswith("ref: refs/heads/"):
            return head.rsplit("/", 1)[-1]
        return ""
    if args == ("rev-parse", "HEAD"):
        if not head.startswith("ref: "):
            return head
        ref = head[5:].strip()
        ref_path = git_dir / ref
        if ref_path.exists():
            try:
                return ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                return None
        packed_refs = git_dir / "packed-refs"
        if packed_refs.exists():
            try:
                for line in packed_refs.read_text(encoding="utf-8").splitlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    sha, _, packed_ref = line.partition(" ")
                    if packed_ref.strip() == ref:
                        return sha
            except OSError:
                return None
    return None


def _model_name_from_config(model_cfg: dict[str, Any]) -> str:
    configured = model_cfg.get("name") or model_cfg.get("model_name")
    if configured:
        return str(configured).replace("-", "_")
    architecture = str(model_cfg.get("architecture") or "").strip().lower().replace("-", "_")
    backbone = str(model_cfg.get("backbone") or "").strip().lower().replace("-", "_")
    if architecture == "segformer" and backbone:
        return backbone if backbone.startswith("segformer_") else f"segformer_{backbone.replace('mit_', 'b')}"
    return "tiny_unet_4ch"


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
        stage_json_path = self.stage_dir / f"{stage}.json"
        report_path = self.stage_dir / f"{stage}.report.md"
        stage_payload["stage_json_path"] = str(stage_json_path)
        stage_payload["report_path"] = str(report_path)
        try:
            from .stage_report_formatter import path_views, write_stage_report_file

            stage_payload["path_views"] = {
                "stage_json": path_views(stage_json_path, container_status_root=self.state_dir),
                "stage_report": path_views(report_path, container_status_root=self.state_dir),
            }
            write_json(stage_json_path, stage_payload)
            write_stage_report_file(
                stage_payload,
                path=report_path,
                stage=stage,
                run_id=self.airflow_run_id,
                stage_json_path=stage_json_path,
                container_status_root=self.state_dir,
            )
        except Exception as exc:  # noqa: BLE001 - report formatting must not hide the stage result.
            stage_payload["report_format_error"] = str(exc)
            write_json(stage_json_path, stage_payload)
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


def _attach_stage_runtime_counters(result: dict[str, Any], stage: str, resources_before: dict[str, Any], resources_after: dict[str, Any]) -> None:
    counters = dict(result.get("counters") or {})
    pool_name = (STAGE_POOLS.get(stage) or ("default_pool", 1))[0]
    counters.setdefault("requested_pool", pool_name)
    counters.setdefault("effective_pool", pool_name)
    if stage not in GPU_XCOM_STAGES:
        result["counters"] = counters
        return
    before_gpu = _first_gpu(resources_before)
    after_gpu = _first_gpu(resources_after)
    gpu = after_gpu or before_gpu
    if gpu:
        counters.setdefault("cuda_available", True)
        counters.setdefault("gpu_name", gpu.get("name"))
        counters.setdefault("gpu_memory_total_mb", gpu.get("memory_total_mb"))
        if before_gpu:
            counters.setdefault("gpu_memory_used_mb_before", before_gpu.get("memory_used_mb"))
        if after_gpu:
            counters.setdefault("gpu_memory_used_mb_after", after_gpu.get("memory_used_mb"))
    elif pool_name.startswith("gpu"):
        counters.setdefault("cuda_available", False)
    result["counters"] = counters


def _first_gpu(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    gpus = snapshot.get("gpu")
    if isinstance(gpus, list) and gpus and isinstance(gpus[0], dict):
        return gpus[0]
    return None


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
    cleanup_errors: list[str] = []
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
        def _chmod_and_retry(function: Any, path: str, _exc_info: Any) -> None:
            try:
                path_obj = Path(path)
                path_obj.chmod(0o777 if path_obj.is_dir() else 0o666)
                function(path)
            except Exception as exc:  # noqa: BLE001 - cleanup is best effort.
                cleanup_errors.append(f"{_safe_rel(Path(path), store.run_dir)}: {type(exc).__name__}: {exc}")

        try:
            shutil.rmtree(target, onerror=_chmod_and_retry)
        except Exception as exc:  # noqa: BLE001 - cleanup must not fail the run finalizer.
            cleanup_errors.append(f"{_safe_rel(target, store.run_dir)}: {type(exc).__name__}: {exc}")
            continue
        deleted_files += file_count
        deleted_bytes += total_size
        deleted_dirs.append({"path": _safe_rel(target, store.run_dir), "files": file_count, "bytes": total_size})
    return {
        "enabled": True,
        "deleted_dirs": deleted_dirs,
        "deleted_files": deleted_files,
        "deleted_bytes": deleted_bytes,
        "deleted_gb": round(deleted_bytes / 1024**3, 3),
        "errors": cleanup_errors,
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
    annotations_cfg = dict(conf.annotations or {})
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
    train_cfg.setdefault("model_name", _model_name_from_config(model_cfg))
    if conf.inference:
        params_cfg.setdefault("inference", dict(conf.inference))
        predict_cfg.setdefault("inference", dict(conf.inference))
    params_cfg.update(
        {
            "model": model_cfg,
            "input_bands": model_cfg.get("input_bands") or [1, 2, 3, 4],
            "pseudolabel": pseudolabel_cfg,
            "annotations": annotations_cfg,
            "annotations.source": annotations_cfg.get("source"),
            "mlmarkup.commit": annotations_cfg.get("commit"),
            "mlmarkup.repo_path": annotations_cfg.get("repo_path"),
            "pseudolabeling.enabled": bool(pseudolabel_cfg.get("enabled", False)),
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
            "annotations": annotations_cfg,
        },
        preprocess=preprocess_cfg,
        train=train_cfg,
        evaluate=conf.evaluate or {},
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
    job.params.setdefault(
        "airflow",
        {
            "dag_id": "mlsystem_experiment_pipeline",
            "run_id": store.airflow_run_id,
            "dag_conf": store.conf,
        },
    )
    extra_tags, extra_params = _mlflow_tuning_metadata(conf.params)
    prepared_manifest = store.run_dir / "dataset_manifest.json"
    if prepared_manifest.exists():
        job.preprocess.setdefault("prepared_dataset_manifest", str(prepared_manifest))
        job.preprocess.setdefault("use_prepared_dataset_manifest", True)
    job_log = store.run_dir / "airflow_train.log"
    tags = {
        "job_id": conf.experiment_id,
        "airflow_run_id": store.airflow_run_id,
        "orchestrator": "airflow",
        "execution_path": "airflow_api",
        "task": conf.task,
        "class_name": conf.class_name or "",
        "mlsystem.class_name": conf.class_name or "",
        **extra_tags,
    }
    params = {
        "experiment_id": conf.experiment_id,
        "task": conf.task,
        "model.name": conf.model.get("name"),
        "preprocess.tile_size": conf.preprocess.get("tile_size"),
        "preprocess.max_scenes": conf.preprocess.get("max_scenes"),
        "train.epochs": job.train.get("epochs"),
        "train.time_limit_sec": job.train.get("time_limit_sec"),
        **extra_params,
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
    skip_reason = "synthetic smoke validates Airflow/API orchestration only; external S3, dataset, training, inference, vectorization and MLflow writes are skipped."
    return _stage_result(
        "skipped",
        summary="Synthetic smoke stage skipped: orchestration-only run.",
        skip_reason=skip_reason,
        is_smoke_synthetic=True,
        details={
            "smoke_mode": {
                "synthetic": True,
                "external_s3_dataset_training_inference_skipped": True,
                "purpose": "orchestration_only",
            }
        },
        warnings=[skip_reason],
    )


def _mlflow_url_fields(pipeline_config: Any, experiment_id: str | None, run_id: str | None) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    internal_uri = str(getattr(pipeline_config, "mlflow_tracking_uri_internal", "") or "")
    configured_public = os.getenv("MLSYSTEM_MLFLOW_PUBLIC_URL") or str(getattr(pipeline_config, "mlflow_tracking_uri_external", "") or "")
    base = configured_public or internal_uri
    if not configured_public:
        warnings.append("MLflow public URL is not configured; using tracking URI.")
    fields: dict[str, Any] = {"tracking_uri": internal_uri}
    if base and experiment_id:
        base = base.rstrip("/")
        fields["url_mlflow_experiment"] = f"{base}/#/experiments/{experiment_id}"
        fields["mlflow_experiment_url"] = fields["url_mlflow_experiment"]
    if base and experiment_id and run_id:
        fields["url_mlflow_run"] = f"{base}/#/experiments/{experiment_id}/runs/{run_id}"
        fields["mlflow_run_url"] = fields["url_mlflow_run"]
    return fields, warnings


def _dataset_manifest(store: AirflowRunStore) -> dict[str, Any]:
    return read_json(store.run_dir / "dataset_manifest.json", default={}) or {}


def _manifest_scene_count(manifest: dict[str, Any], split: str) -> int:
    direct_key = f"{split}_scene_count"
    scenes_key = f"{split}_scenes"
    if manifest.get(direct_key) is not None:
        try:
            return int(manifest.get(direct_key) or 0)
        except (TypeError, ValueError):
            return 0
    scenes = manifest.get(scenes_key) or []
    return len(scenes) if isinstance(scenes, list) else 0


def _section_enabled(section: dict[str, Any] | None, default: bool = True) -> bool:
    return bool((section or {}).get("enabled", default))


def _extract_pixel_metrics(training_result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]], list[str]]:
    last_metrics = training_result.get("last_epoch_metrics") or {}
    aliases = [
        ("val/precision", "pixel_precision"),
        ("val/recall", "pixel_recall"),
        ("val/pixel_iou", "pixel_iou"),
        ("val/iou", "pixel_iou"),
        ("val/pixel_accuracy", "pixel_accuracy"),
        ("val/accuracy", "pixel_accuracy"),
    ]
    metrics: dict[str, Any] = {}
    alias_rows: list[dict[str, str]] = []
    for original, normalized in aliases:
        if original in last_metrics and metrics.get(normalized) is None:
            metrics[normalized] = last_metrics.get(original)
            alias_rows.append({"original_metric_name": original, "normalized_metric_name": normalized})
    warnings: list[str] = []
    raw_f1 = last_metrics.get("val/pixel_f1")
    precision = _finite_float(metrics.get("pixel_precision"))
    recall = _finite_float(metrics.get("pixel_recall"))
    if precision is not None and recall is not None:
        standard_f1 = _standard_f1(precision, recall)
        metrics["pixel_f1"] = standard_f1
        if raw_f1 is not None:
            raw_f1_float = _finite_float(raw_f1)
            alias_rows.append({"original_metric_name": "val/pixel_f1", "normalized_metric_name": "pixel_f1_source_value_not_used"})
            if raw_f1_float is None or not _close_metric(raw_f1_float, standard_f1):
                warnings.append(
                    "val/pixel_f1 is inconsistent with val/precision and val/recall; "
                    f"metric_pixel_f1 was derived as standard F1={standard_f1} from precision/recall, "
                    f"original val/pixel_f1={raw_f1}."
                )
    elif raw_f1 is not None:
        metrics["pixel_f1"] = None
        alias_rows.append({"original_metric_name": "val/pixel_f1", "normalized_metric_name": "pixel_f1_source_value_not_used"})
        warnings.append("val/pixel_f1 exists but precision/recall are unavailable, so metric_pixel_f1 is not published as standard F1.")

    iou = _finite_float(metrics.get("pixel_iou"))
    if precision is not None and recall is not None:
        standard_iou = _standard_iou_from_precision_recall(precision, recall)
        if iou is not None and standard_iou is not None and not _close_metric(iou, standard_iou):
            warnings.append(
                "pixel_iou is inconsistent with val/precision and val/recall; "
                f"metric_pixel_iou was derived as standard IoU={standard_iou}, original pixel_iou={metrics.get('pixel_iou')}."
            )
            metrics["pixel_iou"] = standard_iou
    for key in ("pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "pixel_accuracy"):
        if key not in metrics:
            metrics[key] = None
            warnings.append(f"{key} is not available in training_result.last_epoch_metrics.")
    counters = {
        "scenes_evaluated": training_result.get("val_scene_count"),
        "threshold": last_metrics.get("val/threshold") or training_result.get("metrics_threshold"),
        "pixel_tp": last_metrics.get("val/pixel_tp"),
        "pixel_fp": last_metrics.get("val/pixel_fp"),
        "pixel_fn": last_metrics.get("val/pixel_fn"),
        "pixel_tn": last_metrics.get("val/pixel_tn"),
    }
    missing_counters = [key for key in ("pixel_tp", "pixel_fp", "pixel_fn", "pixel_tn") if counters.get(key) is None]
    if missing_counters:
        warnings.append(
            "pixel confusion-matrix counters are not available in training_result.last_epoch_metrics: "
            + ", ".join(missing_counters)
        )
    if counters.get("threshold") is None:
        warnings.append("pixel threshold is not available in training_result.last_epoch_metrics.")
    return metrics, counters, alias_rows, warnings


def _standard_f1(precision: float, recall: float) -> float | None:
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return 2.0 * precision * recall / denominator


def _standard_iou_from_precision_recall(precision: float, recall: float) -> float | None:
    if precision <= 0 or recall <= 0:
        return 0.0
    denominator = (1.0 / precision) + (1.0 / recall) - 1.0
    if denominator <= 0:
        return None
    return 1.0 / denominator


def _close_metric(left: float, right: float | None, *, rel_tol: float = 1e-3, abs_tol: float = 1e-8) -> bool:
    if right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def _write_pixel_metrics_artifacts(store: AirflowRunStore, metrics: dict[str, Any], counters: dict[str, Any], aliases: list[dict[str, str]], warnings: list[str]) -> dict[str, str]:
    payload = {
        "metrics": metrics,
        "counters": counters,
        "aliases": aliases,
        "warnings": warnings,
        "metrics_source": str(store.run_dir / "training_result.json"),
    }
    json_path = store.run_dir / "pixel_metrics.json"
    txt_path = store.run_dir / "pixel_metrics.txt"
    write_json(json_path, payload)
    lines = [
        "Pixel metrics",
        f"metrics_source={payload['metrics_source']}",
        f"scenes_evaluated={counters.get('scenes_evaluated')}",
    ]
    for key in sorted(metrics):
        lines.append(f"{key}={metrics.get(key)}")
    for key in ("pixel_tp", "pixel_fp", "pixel_fn", "pixel_tn", "threshold"):
        lines.append(f"{key}={counters.get(key)}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"pixel_metrics.json": str(json_path), "pixel_metrics.txt": str(txt_path)}


def _device_counters(training_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "device": training_result.get("device"),
        "cuda_available": training_result.get("cuda_available"),
        "gpu_name": training_result.get("gpu_name"),
    }


def _object_metric_summary(training_result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    metrics = training_result.get("postprocess_metrics") or {}
    normalized = {
        "object_precision": metrics.get("val/object_precision"),
        "object_recall": metrics.get("val/object_recall"),
        "object_f1": metrics.get("val/object_f1"),
    }
    counters = {
        "reference_objects": metrics.get("val/gt_count"),
        "predicted_objects": metrics.get("val/pred_count"),
        "tp_objects": metrics.get("val/tp"),
        "fp_objects": metrics.get("val/fp"),
        "fn_objects": metrics.get("val/fn"),
        "reference_scenes": training_result.get("val_scene_count"),
        "prediction_scenes": None,
    }
    warnings: list[str] = []
    if normalized["object_f1"] is None:
        warnings.append("object metrics are not available at compute_f1 stage; current compatibility pipeline computes them after pseudolabel vectorization when postprocess metrics exist.")
    return normalized, counters, warnings


def _log_mlflow_metrics_if_available(store: AirflowRunStore, metrics: dict[str, Any], counters: dict[str, Any] | None = None) -> list[str]:
    summary = store.read_summary()
    mlflow_info = summary.get("mlflow") or _read_training_result(store).get("mlflow") or {}
    run_id = mlflow_info.get("run_id")
    if not run_id or str(run_id).startswith("smoke-"):
        return []
    warnings: list[str] = []
    try:
        import mlflow

        mlflow.set_tracking_uri(load_config().mlflow_tracking_uri_internal)
        with mlflow.start_run(run_id=run_id):
            for key, value in metrics.items():
                number = _finite_float(value)
                if number is not None:
                    mlflow.log_metric(key, number)
            for key, value in (counters or {}).items():
                number = _finite_float(value)
                if number is not None:
                    mlflow.log_metric(key, number)
    except Exception as exc:  # noqa: BLE001 - report warning without hiding stage metrics.
        warnings.append(f"Failed to log F1 metrics to MLflow: {type(exc).__name__}: {exc}")
    return warnings


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _write_simple_key_value_report(path: Path, title: str, sections: dict[str, dict[str, Any]]) -> None:
    lines = [title]
    for section, values in sections.items():
        lines.extend(["", section])
        for key in sorted(values):
            lines.append(f"{key}={values[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_set_mlflow_experiment_tag(client: Any, experiment_id: str, key: str, value: str) -> str | None:
    try:
        client.set_experiment_tag(experiment_id, key, value)
        return None
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        if "experiment_tag_pk" in lowered or ("duplicate key" in lowered and "experiment_tags" in lowered):
            return f"Skipped concurrent MLflow experiment tag write: {key}"
        raise


def _is_mlflow_duplicate_experiment_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return "resource_already_exists" in message or ("already exists" in message and "experiment" in message)


def _get_or_create_mlflow_experiment_id(client: Any, experiment_name: str, *, attempts: int = 5) -> str:
    import time

    for attempt in range(max(1, attempts)):
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        try:
            return str(client.create_experiment(experiment_name))
        except Exception as exc:
            if not _is_mlflow_duplicate_experiment_error(exc) or attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"MLflow experiment was not created: {experiment_name}")


def _mlflow_tuning_metadata(params: dict[str, Any] | None) -> tuple[dict[str, str], dict[str, str]]:
    params = params or {}
    prefixes = ("tuning.", "dataset.", "validation.")
    tags: dict[str, str] = {}
    mlflow_params: dict[str, str] = {}
    for key, value in params.items():
        key_str = str(key)
        if not key_str.startswith(prefixes):
            continue
        if isinstance(value, (dict, list, tuple, set)):
            value_str = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            value_str = "" if value is None else str(value)
        tags[key_str] = value_str[:5000]
        mlflow_params[key_str] = value_str[:500]
    return tags, mlflow_params


def _create_mlflow_run(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    try:
        import mlflow

        pipeline_config = load_config()
        mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
        experiment_name = conf.mlflow.get("experiment") or pipeline_config.mlflow_default_experiment
        client = mlflow.tracking.MlflowClient()
        experiment_id = _get_or_create_mlflow_experiment_id(client, experiment_name)
        extra_tags, extra_params = _mlflow_tuning_metadata(conf.params)
        tag_warnings: list[str] = []
        if conf.class_name:
            for key, value in (
                ("class_name", conf.class_name),
                ("mlsystem.class_name", conf.class_name),
                ("task", conf.task),
            ):
                warning = _safe_set_mlflow_experiment_tag(client, experiment_id, key, str(value))
                if warning:
                    tag_warnings.append(warning)
        with mlflow.start_run(experiment_id=experiment_id, run_name=conf.experiment_id) as run:
            mlflow.set_tags(
                {
                    "job_id": conf.experiment_id,
                    "airflow_run_id": store.airflow_run_id,
                    "orchestrator": "airflow",
                    "job_status": "running",
                    "execution_path": "airflow_api",
                    "task": conf.task,
                    "class_name": conf.class_name or "",
                    "mlsystem.class_name": conf.class_name or "",
                    **extra_tags,
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
                    **extra_params,
                }
            )
            run_id = run.info.run_id
            artifact_uri = run.info.artifact_uri
        url_fields, url_warnings = _mlflow_url_fields(pipeline_config, experiment_id, run_id)
        run_url = url_fields.get("url_mlflow_run")
        experiment_url = url_fields.get("url_mlflow_experiment")
        store.update_summary(
            mlflow={
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "run_url_external": run_url,
                "experiment_url_external": experiment_url,
                "tracking_uri": pipeline_config.mlflow_tracking_uri_internal,
                "artifact_uri": artifact_uri,
            }
        )
        return _stage_result(
            "success",
            summary=f"MLflow run created: {run_id}",
            mlflow_run_id=run_id,
            mlflow_experiment_id=experiment_id,
            mlflow_experiment_name=experiment_name,
            artifact_uri=artifact_uri,
            counters={"mlflow_experiment_id": experiment_id, "mlflow_run_id": run_id},
            warnings=[*url_warnings, *tag_warnings],
            details={
                "mlflow": {
                    "experiment_name": experiment_name,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "run_name": conf.experiment_id,
                    "tracking_uri": pipeline_config.mlflow_tracking_uri_internal,
                    "artifact_uri": artifact_uri,
                    "run_url": run_url,
                    "experiment_url": experiment_url,
                }
            },
            **url_fields,
        )
    except Exception as exc:
        if conf.smoke:
            fallback = {"run_id": f"smoke-{conf.experiment_id}", "run_url_external": None, "error": f"{type(exc).__name__}: {exc}"}
            store.update_summary(mlflow=fallback, warnings=[f"MLflow smoke fallback: {fallback['error']}"])
            return _stage_result(
                "success",
                summary="MLflow unavailable in local smoke; stored fallback metadata.",
                warnings=[fallback["error"]],
                counters={"mlflow_run_id": fallback["run_id"]},
                mlflow_run_id=fallback["run_id"],
            )
        raise


def _finalize_mlflow_run(conf: AirflowExperimentConfig, store: AirflowRunStore) -> dict[str, Any]:
    summary = store.read_summary()
    mlflow_info = summary.get("mlflow") or {}
    run_id = mlflow_info.get("run_id")
    cleanup = _cleanup_runtime_intermediates(conf, store)
    stages = summary.get("stages") or {}
    failed_stages = [name for name, payload in stages.items() if (payload or {}).get("status") == "failed"]
    warning_count = len(summary.get("warnings") or [])
    final_counters = {
        "failed_stages": len(failed_stages),
        "warning_stages": warning_count,
        "runtime_cleanup_deleted_files": cleanup.get("deleted_files"),
        "runtime_cleanup_deleted_bytes": cleanup.get("deleted_bytes"),
        "mlflow_final_status": "success",
    }
    if not run_id or str(run_id).startswith("smoke-"):
        store.update_summary(status="success", finished_at=utc_now(), runtime_cleanup=cleanup)
        return _stage_result(
            "success",
            summary="No real MLflow run to finalize.",
            counters={**final_counters, "mlflow_run_id": run_id},
            mlflow_run_id=run_id,
            mlflow_final_status="success",
            details={"cleanup": cleanup, "failed_stages": failed_stages},
        )
    try:
        import mlflow

        pipeline_config = load_config()
        mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
        url_fields, url_warnings = _mlflow_url_fields(pipeline_config, mlflow_info.get("experiment_id"), run_id)
        store.update_summary(status="success", finished_at=utc_now(), runtime_cleanup=cleanup)
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags({"job_status": "success", "airflow_status": "success"})
            mlflow.log_artifact(str(store.summary_path))
        return _stage_result(
            "success",
            summary=f"MLflow run finalized: {run_id}",
            counters={**final_counters, "mlflow_run_id": run_id},
            warnings=url_warnings,
            mlflow_run_id=run_id,
            mlflow_final_status="success",
            details={"cleanup": cleanup, "failed_stages": failed_stages, "mlflow": {**mlflow_info, **url_fields}},
            **url_fields,
        )
    except Exception as exc:
        return _stage_result("failed", error=f"{type(exc).__name__}: {exc}", counters={**final_counters, "mlflow_run_id": run_id, "mlflow_final_status": "failed"})


def _run_airflow_synthetic_pseudolabel_smoke(store: AirflowRunStore) -> dict[str, Any]:
    smoke_dir = store.run_dir / "synthetic_pseudolabel"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    accepted_geojson = smoke_dir / f"{store.experiment_id}.accepted.geojson"
    root_accepted_geojson = store.run_dir / f"{store.experiment_id}.accepted.geojson"
    prediction_examples = smoke_dir / "prediction_examples.html"
    root_prediction_examples = store.run_dir / "prediction_examples.html"
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
    geojson_text = json.dumps(geojson, ensure_ascii=False, indent=2)
    accepted_geojson.write_text(geojson_text, encoding="utf-8")
    root_accepted_geojson.write_text(geojson_text, encoding="utf-8")
    prediction_html = "<!doctype html><html><body><h1>MLSystem Airflow smoke</h1><p>Synthetic pseudolabel smoke completed.</p></body></html>"
    prediction_examples.write_text(prediction_html, encoding="utf-8")
    root_prediction_examples.write_text(prediction_html, encoding="utf-8")
    coverage_report = {
        "mode": "airflow_synthetic",
        "scenes_processed": 1,
        "total_expected_windows": 1,
        "total_predicted_windows": 1,
        "mean_coverage_fraction": 1.0,
        "min_coverage_fraction": 1.0,
    }
    pseudolabel_summary = {
        "status": "success",
        "mode": "airflow_synthetic",
        "accepted_geojson": str(root_accepted_geojson),
        "prediction_examples_html": str(root_prediction_examples),
        "metrics": {
            "accepted_objects": 1,
            "threshold_used": 0.5,
            "min_object_area_m2_used": 1,
            "simplify_tolerance_m_used": 0,
        },
    }
    write_json(store.run_dir / "coverage_report.json", coverage_report)
    write_json(store.run_dir / "pseudolabel_summary.json", pseudolabel_summary)
    (store.run_dir / "pseudolabel_scenes.txt").write_text("synthetic\n", encoding="utf-8")
    write_json(store.run_dir / "pseudolabel_scene_results_manifest.json", {"mode": "airflow_synthetic", "scenes": ["synthetic"]})
    write_json(smoke_dir / "summary.json", smoke_summary)
    return {
        "summary": "Synthetic pseudolabel smoke completed without heavy ML dependencies.",
        "accepted_geojson": str(root_accepted_geojson),
        "prediction_examples_html": str(root_prediction_examples),
        "smoke_summary": smoke_summary,
        "coverage_report": str(store.run_dir / "coverage_report.json"),
        "pseudolabel_summary": str(store.run_dir / "pseudolabel_summary.json"),
    }


def _run_dispatcher_stage(stage: str, conf_payload: dict[str, Any], airflow_run_id: str, state_dir: Path) -> dict[str, Any]:
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
        monitor.stop()
        return store.write_stage(stage, smoke_result)

    if stage == "create_mlflow_run":
        result = _create_mlflow_run(conf, store)
    elif stage == "train_model":
        if not conf.train.get("enabled", True):
            result = _stage_result("skipped", summary="train.enabled=false")
        else:
            training_result = _run_training_pipeline(conf, store)
            result = _stage_result(
                "success",
                summary="TrainingPipeline completed real MLSystem train-only run; pseudolabel is handled by inference_engine_pipeline.",
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
            if not (training_result.get("last_epoch_metrics") or {}):
                result = _stage_result("failed", error="No pixel metrics found in training_result.json.")
            else:
                metrics, counters, aliases, warnings = _extract_pixel_metrics(training_result)
                artifacts = _write_pixel_metrics_artifacts(store, metrics, counters, aliases, warnings)
                result = _stage_result(
                    "success",
                    summary="Pixel metrics extracted from training_result.json.",
                    metrics=metrics,
                    counters=counters,
                    warnings=warnings,
                    artifacts=artifacts,
                    details={
                        "metrics_source": str(store.run_dir / "training_result.json"),
                        "aliases": aliases,
                        "metrics": metrics,
                    },
                )
    elif stage == "predict_validation_scenes":
        training_result = _read_training_result(store)
        configured_checkpoint = (conf.pseudolabel or {}).get("checkpoint_path") or (conf.predict or {}).get("checkpoint_path") or (conf.params or {}).get("checkpoint_path")
        if not _section_enabled(conf.train) or not _section_enabled(conf.predict):
            manifest = _dataset_manifest(store)
            validation_input_scenes = _manifest_scene_count(manifest, "val")
            prediction_summary = {
                "input_validation_scenes": validation_input_scenes,
                "processed_scenes": 0,
                "skipped_scenes": validation_input_scenes,
                "failed_scenes": 0,
                "predicted_tiles": 0,
                "predicted_windows": 0,
                "status": "skipped",
                "reason": "validation prediction disabled; InferenceEngine pipeline uses the configured MLflow/Triton model.",
            }
            json_path = store.run_dir / "validation_prediction_summary.json"
            txt_path = store.run_dir / "validation_prediction_summary.txt"
            write_json(json_path, prediction_summary)
            _write_simple_key_value_report(txt_path, "Validation prediction", {"summary": prediction_summary})
            result = _stage_result(
                "skipped",
                summary="Validation prediction skipped because train/predict is disabled for this InferenceEngine-only DAG run.",
                skip_reason=prediction_summary["reason"],
                counters={
                    "validation_input_scenes": validation_input_scenes,
                    "validation_scenes_processed": 0,
                    "validation_scenes_skipped": validation_input_scenes,
                    "validation_scenes_failed": 0,
                    "validation_prediction_tiles": 0,
                    "validation_prediction_windows": 0,
                },
                artifacts={
                    "validation_prediction_summary.json": str(json_path),
                    "validation_prediction_summary.txt": str(txt_path),
                },
            )
        elif not training_result.get("checkpoint_path") and not configured_checkpoint:
            result = _stage_result("failed", error="Training checkpoint is missing; validation prediction cannot start.")
        else:
            manifest = _dataset_manifest(store)
            validation_input_scenes = _manifest_scene_count(manifest, "val")
            prediction_summary = {
                "input_validation_scenes": validation_input_scenes,
                "processed_scenes": 0,
                "skipped_scenes": validation_input_scenes,
                "failed_scenes": 0,
                "predicted_tiles": None,
                "predicted_windows": None,
                "checkpoint_path": training_result.get("checkpoint_path") or configured_checkpoint,
                "note": "Validation prediction is not a separate production computation in the current compatibility pipeline.",
            }
            json_path = store.run_dir / "validation_prediction_summary.json"
            txt_path = store.run_dir / "validation_prediction_summary.txt"
            write_json(json_path, prediction_summary)
            _write_simple_key_value_report(
                txt_path,
                "Validation prediction",
                {
                    "counts": prediction_summary,
                    "device": _device_counters(training_result),
                },
            )
            result = _stage_result(
                "skipped",
                summary="Compatibility placeholder: predict_validation_scenes does not run distinct validation inference yet.",
                skip_reason="validation prediction is currently deferred; no prediction windows were produced by this stage.",
                warnings=["validation scenes are marked skipped here because current compatibility code does not run a distinct validation prediction stage."],
                counters={
                    "validation_input_scenes": validation_input_scenes,
                    "validation_scenes_processed": 0,
                    "validation_scenes_skipped": validation_input_scenes,
                    "validation_scenes_failed": 0,
                    "validation_prediction_tiles": None,
                    "validation_prediction_windows": None,
                    **_device_counters(training_result),
                },
                artifacts={
                    "validation_prediction_summary.json": str(json_path),
                    "validation_prediction_summary.txt": str(txt_path),
                    "dataset_manifest.json": str(store.run_dir / "dataset_manifest.json"),
                },
                checkpoint_path=training_result.get("checkpoint_path") or configured_checkpoint,
            )
    elif stage == "vectorize_validation_predictions":
        summary_path = store.run_dir / "pseudolabel_summary.json"
        if not summary_path.exists():
            manifest = _dataset_manifest(store)
            validation_input_scenes = _manifest_scene_count(manifest, "val")
            summary = {
                "input_scenes": validation_input_scenes,
                "processed_scenes": 0,
                "prediction_files_read": 0,
                "objects_total": None,
                "empty_scenes": None,
                "failed_scenes": 0,
                "status": "deferred",
                "reason": "validation vectorization is not a distinct computation in the current compatibility pipeline.",
            }
            json_path = store.run_dir / "validation_vectorization_summary.json"
            txt_path = store.run_dir / "validation_vectorization_summary.txt"
            by_scene_path = store.run_dir / "validation_objects_by_scene.txt"
            write_json(json_path, summary)
            _write_simple_key_value_report(txt_path, "Validation vectorization", {"summary": summary})
            by_scene_path.write_text("", encoding="utf-8")
            result = _stage_result(
                "skipped",
                summary="Compatibility placeholder: validation vectorization is not implemented as a distinct stage yet.",
                skip_reason="validation vectorization did not run because validation prediction artifacts are not produced by a distinct stage yet.",
                warnings=["validation vectorization counters are not available before pseudolabel compatibility postprocess runs."],
                counters={
                    "validation_vectorized_scenes": 0,
                    "validation_prediction_files": 0,
                    "validation_vectorized_objects": None,
                    "validation_empty_scenes": None,
                    "validation_failed_scenes": 0,
                },
                artifacts={
                    "validation_vectorization_summary.json": str(json_path),
                    "validation_vectorization_summary.txt": str(txt_path),
                    "validation_objects_by_scene.txt": str(by_scene_path),
                },
            )
        else:
            summary = read_json(summary_path, default={}) or {}
            metrics = summary.get("metrics") or {}
            objects_total = metrics.get("accepted_objects")
            vector_summary = {
                "processed_scenes": (read_json(store.run_dir / "coverage_report.json", default={}) or {}).get("scenes_processed"),
                "prediction_files_read": None,
                "objects_total": objects_total,
                "empty_scenes": None,
                "failed_scenes": None,
                "source": str(summary_path),
            }
            json_path = store.run_dir / "validation_vectorization_summary.json"
            txt_path = store.run_dir / "validation_vectorization_summary.txt"
            by_scene_path = store.run_dir / "validation_objects_by_scene.txt"
            write_json(json_path, vector_summary)
            _write_simple_key_value_report(txt_path, "Validation vectorization", {"summary": vector_summary})
            by_scene_path.write_text("", encoding="utf-8")
            result = _stage_result(
                "success",
                counters={
                    "validation_vectorized_scenes": vector_summary.get("processed_scenes"),
                    "validation_prediction_files": vector_summary.get("prediction_files_read"),
                    "validation_vectorized_objects": objects_total,
                    "validation_empty_scenes": vector_summary.get("empty_scenes"),
                    "validation_failed_scenes": vector_summary.get("failed_scenes"),
                },
                artifacts={
                    "validation_vectorization_summary.json": str(json_path),
                    "validation_vectorization_summary.txt": str(txt_path),
                    "validation_objects_by_scene.txt": str(by_scene_path),
                },
                accepted_objects=objects_total,
                threshold_used=metrics.get("threshold_used"),
            )
    elif stage == "compute_f1":
        if not _section_enabled(conf.train):
            summary_payload = {
                "status": "skipped",
                "reason": "training and validation metrics are disabled for this InferenceEngine-only DAG run.",
                "pixel_metrics": {},
                "object_metrics": {},
            }
            json_path = store.run_dir / "f1_summary.json"
            txt_path = store.run_dir / "f1_summary.txt"
            object_matching_json = store.run_dir / "object_matching_report.json"
            object_matching_txt = store.run_dir / "object_matching_report.txt"
            write_json(json_path, summary_payload)
            _write_simple_key_value_report(txt_path, "F1 summary", {"summary": summary_payload})
            write_json(object_matching_json, {"status": "skipped", "reason": summary_payload["reason"]})
            _write_simple_key_value_report(object_matching_txt, "Object matching", {"summary": summary_payload})
            result = _stage_result(
                "skipped",
                summary="F1 computation skipped because training is disabled for this InferenceEngine-only DAG run.",
                skip_reason=summary_payload["reason"],
                counters={
                    "reference_objects": 0,
                    "predicted_objects": 0,
                    "tp_objects": 0,
                    "fp_objects": 0,
                    "fn_objects": 0,
                    "reference_scenes": 0,
                    "prediction_scenes": 0,
                },
                artifacts={
                    "f1_summary.json": str(json_path),
                    "f1_summary.txt": str(txt_path),
                    "object_matching_report.json": str(object_matching_json),
                    "object_matching_report.txt": str(object_matching_txt),
                },
                details=summary_payload,
            )
        else:
            training_result = _read_training_result(store)
            pixel_metrics, pixel_counters, aliases, pixel_warnings = _extract_pixel_metrics(training_result)
            object_metrics, object_counters, object_warnings = _object_metric_summary(training_result)
            summary_payload = {
                "pixel_metrics": pixel_metrics,
                "pixel_counters": pixel_counters,
                "object_metrics": object_metrics,
                "object_counters": object_counters,
                "aliases": aliases,
                "warnings": pixel_warnings + object_warnings,
            }
            json_path = store.run_dir / "f1_summary.json"
            txt_path = store.run_dir / "f1_summary.txt"
            object_matching_json = store.run_dir / "object_matching_report.json"
            object_matching_txt = store.run_dir / "object_matching_report.txt"
            write_json(json_path, summary_payload)
            _write_simple_key_value_report(
                txt_path,
                "F1 summary",
                {"pixel_metrics": pixel_metrics, "pixel_counts": pixel_counters, "object_metrics": object_metrics, "object_counts": object_counters},
            )
            write_json(object_matching_json, {"status": "not_available" if object_metrics.get("object_f1") is None else "available", "source": str(store.run_dir / "training_result.json"), "object_counters": object_counters})
            _write_simple_key_value_report(object_matching_txt, "Object matching", {"object_counts": object_counters, "object_metrics": object_metrics})
            all_metrics = {**pixel_metrics, **object_metrics}
            all_counters = {**pixel_counters, **object_counters}
            mlflow_warnings = _log_mlflow_metrics_if_available(store, all_metrics, all_counters)
            compute_status = "success_with_warning" if object_metrics.get("object_f1") is None else "success"
            compute_summary = (
                "Pixel metrics summarized; object metrics are not available because validation vectorization is not implemented as a distinct stage yet."
                if object_metrics.get("object_f1") is None
                else "F1 metrics summarized from available pixel and object artifacts."
            )
            result = _stage_result(
                compute_status,
                summary=compute_summary,
                metrics=all_metrics,
                counters=all_counters,
                warnings=pixel_warnings + object_warnings + mlflow_warnings,
                artifacts={
                    "f1_summary.json": str(json_path),
                    "f1_summary.txt": str(txt_path),
                    "object_matching_report.json": str(object_matching_json),
                    "object_matching_report.txt": str(object_matching_txt),
                },
                details=summary_payload,
            )
    elif stage == "generate_prediction_examples":
        examples = store.run_dir / "prediction_examples.html"
        if not examples.exists():
            result = _stage_result("failed", error="prediction_examples.html is missing.")
        else:
            result = _stage_result(
                "success",
                summary="Prediction examples artifact is available.",
                counters={
                    "examples_requested": 1,
                    "examples_generated": 1,
                    "examples_failed": 0,
                    "prediction_examples_size_bytes": examples.stat().st_size,
                },
                artifacts={"prediction_examples.html": str(examples)},
                prediction_examples_html=str(examples),
                size_bytes=examples.stat().st_size,
            )
    elif stage == "log_mlflow_artifacts":
        run_summary_path, codex_summary_path = _write_run_summaries(conf, store)
        mlflow_info = (store.read_summary().get("mlflow") or _read_training_result(store).get("mlflow") or {})
        run_id = mlflow_info.get("run_id")
        artifacts_failed = 0
        artifact_errors: list[str] = []
        logged_paths = [str(run_summary_path), str(codex_summary_path)]
        if run_id and not str(run_id).startswith("smoke-"):
            import mlflow

            pipeline_config = load_config()
            mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
            with mlflow.start_run(run_id=run_id):
                for artifact_path in logged_paths:
                    try:
                        mlflow.log_artifact(artifact_path)
                    except Exception as exc:  # noqa: BLE001 - keep logging stage report explicit.
                        artifacts_failed += 1
                        artifact_errors.append(f"{Path(artifact_path).name}: {type(exc).__name__}: {exc}")
        pipeline_config = load_config()
        url_fields, url_warnings = _mlflow_url_fields(pipeline_config, mlflow_info.get("experiment_id"), run_id)
        result = _stage_result(
            "failed" if artifacts_failed else "success",
            summary=f"Logged {len(logged_paths) - artifacts_failed} summary artifacts to MLflow.",
            counters={
                "artifacts_logged": len(logged_paths) - artifacts_failed,
                "artifacts_failed": artifacts_failed,
                "mlflow_run_id": run_id,
            },
            warnings=url_warnings,
            error="; ".join(artifact_errors) if artifact_errors else None,
            artifacts={"run_summary.json": str(run_summary_path), "codex_summary.json": str(codex_summary_path)},
            mlflow_run_id=run_id,
            **url_fields,
        )
    elif stage == "write_codex_api_summary":
        run_summary_path, codex_summary_path = _write_run_summaries(conf, store)
        result = _stage_result(
            "success",
            summary="Run and Codex API summaries written.",
            counters={"summary_sections": 2},
            artifacts={"run_summary.json": str(run_summary_path), "codex_summary.json": str(codex_summary_path)},
            summary_path=str(codex_summary_path),
            run_summary=str(run_summary_path),
            codex_summary=str(codex_summary_path),
        )
    elif stage == "finalize_mlflow_run":
        result = _finalize_mlflow_run(conf, store)
    else:
        raise ValueError(f"Unknown Airflow MLSystem stage: {stage}")

    result["duration_sec"] = round(time.time() - started, 3)
    monitor_payload = monitor.stop()
    resources_after = _resource_snapshot()
    if not result.get("is_smoke_synthetic"):
        _attach_stage_runtime_counters(result, stage, resources_before, resources_after)
        result["resources"] = {"before": resources_before, "after": resources_after, "monitor": _safe_rel(store.stage_dir / f"{stage}.resources.json", store.run_dir), "sample_count": len(monitor_payload.get("samples") or [])}
    return store.write_stage(stage, result)


def run_stage(stage: str, conf_payload: dict[str, Any], airflow_run_id: str, state_dir: Path) -> dict[str, Any]:
    from .stages.context import StageContext
    from .stages.registry import get_stage_entrypoint
    from .stages.report import StageFailure, StageReport

    try:
        entrypoint = get_stage_entrypoint(stage)
    except KeyError:
        if stage not in DISPATCHER_STAGE_NAMES:
            known = ", ".join(MAIN_DAG_STAGES + sorted(DISPATCHER_STAGE_NAMES))
            raise ValueError(f"Unknown Airflow MLSystem stage: {stage}. Known stages: {known}")
        return _run_dispatcher_stage(stage, conf_payload, airflow_run_id, state_dir)

    started = time.time()
    resources_before = _resource_snapshot()
    conf = AirflowExperimentConfig.model_validate(conf_payload)
    store = AirflowRunStore(state_dir, airflow_run_id, conf.model_dump())
    logger = logging.getLogger(f"mlsystem.airflow.{stage}")
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

    failure_to_raise: Exception | None = None
    try:
        smoke_result = _smoke_or_skip(conf, stage)
        if smoke_result is not None:
            result = smoke_result
        else:
            context = StageContext(
                stage_id=stage,
                run_id=airflow_run_id,
                config=conf,
                raw_conf=conf_payload,
                status_dir=state_dir,
                store=store,
                logger=logger,
            )
            report = entrypoint(context)
            result = report.to_stage_payload()
            log_text = report.to_airflow_log()
            if report.status == "failed":
                logger.error(log_text)
                failure_to_raise = RuntimeError(result.get("error") or report.summary or f"{stage} failed")
            else:
                logger.info(log_text)
    except StageFailure as exc:
        report = exc.report
        result = report.to_stage_payload()
        logger.error(report.to_airflow_log())
        failure_to_raise = exc
    except Exception as exc:  # noqa: BLE001
        report = StageReport(stage_id=stage, status="failed", errors=[str(exc)])
        result = report.to_stage_payload()
        logger.error(report.to_airflow_log())
        failure_to_raise = exc
    finally:
        monitor_payload = monitor.stop()

    result["duration_sec"] = round(time.time() - started, 3)
    resources_after = _resource_snapshot()
    if not result.get("is_smoke_synthetic"):
        _attach_stage_runtime_counters(result, stage, resources_before, resources_after)
        result["resources"] = {
            "before": resources_before,
            "after": resources_after,
            "monitor": _safe_rel(store.stage_dir / f"{stage}.resources.json", store.run_dir),
            "sample_count": len(monitor_payload.get("samples") or []),
        }
    written = store.write_stage(stage, result)
    if failure_to_raise is not None:
        raise RuntimeError(result.get("error") or f"{stage} failed") from failure_to_raise
    return written


def run_airflow_stage(stage: str, dag_run_conf: dict[str, Any], airflow_run_id: str, state_dir: Path | str) -> dict[str, Any]:
    from ..orchestration.airflow_api_client import run_stage_via_api

    return run_stage_via_api(stage, dag_run_conf, airflow_run_id, Path(state_dir))


def push_stage_xcom(summary: dict[str, Any], task_instance: Any) -> None:
    """Push compact stage result as readable Airflow XCom key/value pairs."""
    if task_instance is None:
        return
    enriched = dict(summary)
    stage = _canonical_xcom_stage(str(enriched.get("stage") or ""))
    if stage:
        enriched.setdefault("pool", (STAGE_POOLS.get(str(stage)) or ("default_pool", 1))[0])
    enriched.setdefault("execution_mode", str(os.getenv("MLSYSTEM_AIRFLOW_EXECUTION_MODE") or "api").lower())
    for key in (
        "stage",
        "status",
        "run_id",
        "job_id",
        "summary",
        "report_path",
        "stage_json_path",
        "warnings_count",
        "errors_count",
        "duration_sec",
        "pool",
        "execution_mode",
        "skip_reason",
        "is_smoke_synthetic",
        "mlflow_run_id",
        "mlflow_final_status",
        "summary_path",
        "url_mlflow_run",
        "url_mlflow_experiment",
    ):
        if key in enriched:
            safe_xcom_push(task_instance, key, enriched.get(key))
    counters = enriched.get("key_counters") or {}
    requested_pool = counters.get("requested_pool") or enriched.get("pool")
    effective_pool = counters.get("effective_pool") or requested_pool
    safe_xcom_push(task_instance, "requested_pool", requested_pool)
    safe_xcom_push(task_instance, "effective_pool", effective_pool)
    if enriched.get("is_smoke_synthetic"):
        return

    metrics = enriched.get("key_metrics") or {}
    object_metrics_available: bool | None = None
    if stage == "compute_f1":
        object_metric_keys = {"object_precision", "object_recall", "object_f1"}
        pixel_metric_keys = {"pixel_precision", "pixel_recall", "pixel_f1", "pixel_iou", "pixel_accuracy"}
        object_available = any(_is_numeric_xcom_value(metrics.get(name)) for name in object_metric_keys)
        object_metrics_available = object_available
        pixel_available = any(_is_numeric_xcom_value(metrics.get(name)) for name in pixel_metric_keys)
        safe_xcom_push(task_instance, "object_metrics_available", object_available)
        if not object_available:
            safe_xcom_push(task_instance, "object_metrics_reason", "validation vectorization is not implemented as a distinct stage yet")
        safe_xcom_push(task_instance, "pixel_metrics_available", pixel_available)
        if not pixel_available:
            safe_xcom_push(task_instance, "pixel_metrics_reason", "pixel metrics are not available in current training_result.json")

    for name in sorted(STAGE_XCOM_DIRECT_COUNTERS.get(stage, set())):
        if name in counters:
            safe_xcom_push(task_instance, _xcom_key(name), counters.get(name))
    if stage in GPU_XCOM_STAGES:
        for name in ("device", "cuda_available", "gpu_name"):
            if name in counters:
                safe_xcom_push(task_instance, name, counters.get(name))
    for name in sorted(STAGE_XCOM_COUNTERS.get(stage, set())):
        if stage == "compute_f1" and object_metrics_available is False:
            continue
        if name in counters:
            safe_xcom_push(task_instance, f"counter_{_xcom_key(name)}", counters.get(name))
    for name in sorted(STAGE_XCOM_METRICS.get(stage, set())):
        if name in metrics:
            safe_xcom_push(task_instance, f"metric_{_xcom_key(name)}", metrics.get(name))


def stage_return_message(summary: dict[str, Any]) -> str:
    message = f"{summary.get('status')}: {summary.get('stage')}, report={summary.get('report_path')}"
    return str(xcom_safe_value(message) or "")[:512]


XCOM_MAX_VALUE_CHARS = 2000
_XCOM_BASE_KEYS = {
    "stage",
    "status",
    "run_id",
    "job_id",
    "summary",
    "report_path",
    "stage_json_path",
    "warnings_count",
    "errors_count",
    "duration_sec",
    "pool",
    "execution_mode",
    "requested_pool",
    "effective_pool",
    "skip_reason",
    "is_smoke_synthetic",
    "mlflow_run_id",
    "mlflow_final_status",
    "summary_path",
    "url_mlflow_run",
    "url_mlflow_experiment",
}

_XCOM_DIRECT_KEYS = {
    "backend",
    "device",
    "cuda_available",
    "gpu_name",
    "split_strategy",
    "vectorization_mode",
    "threshold",
    "limit_source",
    "limit_reason",
    "object_metrics_available",
    "object_metrics_reason",
    "pixel_metrics_available",
    "pixel_metrics_reason",
}


def safe_xcom_push(task_instance: Any, key: str, value: Any) -> None:
    """Push one XCom key after scalar-only normalization."""
    normalized_key = _xcom_key(key)
    logger = logging.getLogger("mlsystem.airflow.xcom")
    if not _is_allowed_xcom_key(normalized_key):
        logger.warning("Skipping unsupported XCom key %s", normalized_key)
        return
    safe_value, converted, skip_reason = _coerce_xcom_value(value, key=normalized_key)
    if skip_reason:
        logger.info(
            "Skipping unsafe/unavailable XCom key=%s value_type=%s reason=%s",
            normalized_key,
            type(value).__name__,
            skip_reason,
        )
        return
    if converted:
        logger.warning("Converted XCom value for key %s to safe scalar", normalized_key)
    task_instance.xcom_push(key=normalized_key, value=safe_value)


def xcom_safe_value(value: Any) -> str | int | float | bool | None:
    """Return an Airflow metadata-safe scalar value."""
    return _coerce_xcom_value(value)[0]


def _is_allowed_xcom_key(key: str) -> bool:
    return key in _XCOM_BASE_KEYS or key in _XCOM_DIRECT_KEYS or key.startswith("counter_") or key.startswith("metric_") or key.startswith("url_")


def _canonical_xcom_stage(stage: str) -> str:
    return stage


def _coerce_xcom_value(value: Any, *, key: str | None = None) -> tuple[str | int | float | bool | None, bool, str | None]:
    original_type = type(value)
    normalized_key = _xcom_key(key or "")
    is_metric = normalized_key.startswith("metric_")
    is_counter = normalized_key.startswith("counter_")
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.generic):
            value = value.item()
    except Exception:  # noqa: BLE001 - numpy is optional for the wrapper.
        pass

    if value is None:
        return None, False, "unavailable None value"
    if isinstance(value, bool):
        if is_metric or is_counter:
            return None, False, "boolean is not a numeric metric/counter"
        return value, original_type is not bool, None
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value), original_type is not int, None
    if isinstance(value, float):
        if not math.isfinite(float(value)):
            return None, False, "non-finite float"
        return float(value), original_type is not float, None
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None, False, "non-finite Decimal"
        return float(value), True, None
    if isinstance(value, datetime):
        if is_metric or is_counter:
            return None, False, "datetime is not a numeric metric/counter"
        return value.isoformat(), True, None
    if isinstance(value, date):
        if is_metric or is_counter:
            return None, False, "date is not a numeric metric/counter"
        return value.isoformat(), True, None
    if isinstance(value, Path):
        if is_metric or is_counter:
            return None, False, "Path is not a numeric metric/counter"
        return _truncate_xcom_text(str(value)), True, None
    if isinstance(value, str):
        if is_metric or is_counter:
            if value.strip().lower() in {"", "none", "null", "nan", "inf", "+inf", "-inf", "not_available", "not_computed", "unknown"}:
                return None, False, "unavailable string value"
            return None, False, "string is not a numeric metric/counter"
        return _truncate_xcom_text(value), False, None
    if isinstance(value, bytes):
        return None, False, "bytes are not allowed in XCom"
    if isinstance(value, (dict, list, tuple, set)):
        return None, False, "container values are not allowed in scalar XCom"
    return None, False, "custom object is not allowed in scalar XCom"


def _is_numeric_xcom_value(value: Any) -> bool:
    safe_value, _converted, skip_reason = _coerce_xcom_value(value, key="metric_value")
    return skip_reason is None and isinstance(safe_value, (int, float)) and not isinstance(safe_value, bool)


def _truncate_xcom_text(value: str, max_chars: int = XCOM_MAX_VALUE_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _xcom_key(value: Any) -> str:
    key = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return key[:180] or "value"


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
