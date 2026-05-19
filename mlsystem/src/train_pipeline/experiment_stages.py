from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ..dataset_preparing.contracts import DatasetIdentity
from ..mlflow_adapter.api import (
    MLFLOW_EXCLUDED_ARTIFACT_NAMES,
    create_run as mlflow_create_run,
    log_artifacts_to_run as mlflow_log_artifacts_to_run,
    log_dataset_input_to_run as mlflow_log_dataset_input_to_run,
    log_metrics_to_run as mlflow_log_metrics_to_run,
    log_params_to_run as mlflow_log_params_to_run,
    mlflow_tuning_metadata,
    mlflow_url_fields,
    run_with_active_mlflow_run as mlflow_run_with_active_mlflow_run,
    search_runs as mlflow_search_runs,
    set_run_tags as mlflow_set_run_tags,
)
from ..settings.api import load_config
from ..train_pipeline.contracts import JobSpec, PipelineRunConfig
from ..storage.api import raster_path_for_s3_key, read_json, write_json
from ..train.api import train_model
from ..train.contracts import TrainConfig, TrainRequest

STAGE_POOLS = {
    "inventory_scenes": ("io_light", 1),
    "prepare_dataset": ("cpu_heavy", 2),
    "create_mlflow_run": ("io_light", 1),
    "train_model": ("gpu_training", 1),
    "evaluate_pixel_metrics": ("cpu_light", 1),
    "predict_validation_scenes": ("gpu_inference", 1),
    "vectorize_validation_predictions": ("cpu_heavy", 2),
    "compute_f1": ("cpu_heavy", 1),
    "generate_prediction_examples": ("cpu_heavy", 1),
    "log_mlflow_artifacts": ("io_light", 1),
    "write_codex_api_summary": ("io_light", 1),
    "finalize_mlflow_run": ("io_light", 1),
}

GPU_RESOURCE_STAGES = {"train_model", "predict_validation_scenes"}
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
MODEL_METRIC_KEYS = {
    "f1_pixel",
    "epochs_total",
    "epoch_time_sec",
    "training_time_sec",
    "val/best_threshold",
    "val/pixel_f1_best_threshold",
    "val/precision_best_threshold",
    "val/recall_best_threshold",
    "val/pixel_iou_best_threshold",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _amsterdam_today() -> date:
    try:
        return datetime.now(ZoneInfo("Europe/Amsterdam")).date()
    except Exception:
        return datetime.now().date()


def _slug_for_run_name(value: str | None) -> str:
    text = str(value or "").strip().casefold()
    aliases = {
        "deforestation": "deforest",
        "cuttings": "deforest",
        "deforest": "deforest",
        "вырубки": "deforest",
        "lakes": "lakes",
        "озера": "lakes",
        "abrasion": "abrasion",
        "абразия": "abrasion",
        "wind_erosion": "wind_erosion",
        "ветровая эрозия": "wind_erosion",
        "water_erosion": "water_erosion",
        "водная эрозия": "water_erosion",
        "burnt_forests": "burnt_forests",
        "burnt": "burnt_forests",
        "burned": "burnt_forests",
        "гари": "burnt_forests",
        "forest": "forest_boundaries",
        "forest_boundary": "forest_boundaries",
        "forest_boundaries": "forest_boundaries",
        "границы леса": "forest_boundaries",
        "salty": "salty",
        "salty_soils": "salty",
        "salinization": "salty",
        "засоления": "salty",
        "landslide": "obval_opolz_osyp",
        "obval_opolz_osyp": "obval_opolz_osyp",
        "оползневые": "obval_opolz_osyp",
        "desertification": "desertification",
        "desert": "desertification",
        "опустынивание": "desertification",
        "arable": "arable_land",
        "arable_land": "arable_land",
        "pashni": "arable_land",
        "пашни": "arable_land",
        "quarries": "quarries",
        "карьеры": "quarries",
    }
    for candidate, slug in aliases.items():
        if candidate in text:
            return slug
    chars = []
    for ch in text.replace("-", "_"):
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "run"


def _select_short_mlflow_run_name(class_slug: str, started_on: date, existing_run_names: list[str]) -> str:
    base = f"{class_slug}_{started_on.strftime('%m%d')}"
    suffixes = [1 if name == base else None for name in existing_run_names]
    prefix = f"{base}_"
    for name in existing_run_names:
        if not name.startswith(prefix):
            continue
        tail = name[len(prefix):]
        if tail.isdigit():
            suffixes.append(int(tail))
    used = [item for item in suffixes if item is not None]
    if not used:
        return base
    return f"{base}_{max(used) + 1}"


def _existing_mlflow_run_names(pipeline_config: Any, experiment_name: str) -> list[str]:
    try:
        runs = mlflow_search_runs(pipeline_config, experiment_name=experiment_name, max_results=1000)
    except Exception:
        return []
    names: list[str] = []
    for run in runs:
        tags = run.get("tags") if isinstance(run.get("tags"), dict) else {}
        name = tags.get("mlflow.runName") or run.get("run_name")
        if name:
            names.append(str(name))
    return names


def _model_name_from_config(model_cfg: dict[str, Any]) -> str:
    configured = model_cfg.get("name") or model_cfg.get("model_name")
    if configured:
        return str(configured).replace("-", "_")
    architecture = str(model_cfg.get("architecture") or "").strip().lower().replace("-", "_")
    backbone = str(model_cfg.get("backbone") or "").strip().lower().replace("-", "_")
    if architecture == "segformer" and backbone:
        return backbone if backbone.startswith("segformer_") else f"segformer_{backbone.replace('mit_', 'b')}"
    return "tiny_unet_4ch"


def _dataset_identity_from_summary(payload: Any) -> DatasetIdentity | None:
    if isinstance(payload, DatasetIdentity):
        return payload
    if not isinstance(payload, dict):
        return None
    fields = DatasetIdentity.__dataclass_fields__
    clean = {key: value for key, value in payload.items() if key in fields}
    required = {"fingerprint", "version", "version_source", "objects_count", "scenes_count"}
    if not required.issubset(clean):
        return None
    return DatasetIdentity(**clean)


def load_conf(conf_file: str | None = None, conf_json: str | None = None) -> dict[str, Any]:
    if conf_json:
        return json.loads(conf_json)
    if conf_file:
        return json.loads(Path(conf_file).read_text(encoding="utf-8-sig"))
    return {}


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
    if stage not in GPU_RESOURCE_STAGES:
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


def _existing_artifacts(store: Any) -> dict[str, str]:
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


def _forbidden_logged_artifacts(store: Any) -> list[str]:
    return sorted(name for name in MLFLOW_EXCLUDED_ARTIFACT_NAMES if (store.run_dir / name).exists())


def _cleanup_runtime_intermediates(conf: PipelineRunConfig, store: Any) -> dict[str, Any]:
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


def _read_training_result(store: Any) -> dict[str, Any]:
    summary = store.read_summary()
    return summary.get("training_result") or read_json(store.run_dir / "training_result.json", default={}) or {}


def _build_pipeline_job(conf: PipelineRunConfig) -> JobSpec:
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
        description="Pipeline experiment run",
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


def _requires_explicit_training_epochs(conf: PipelineRunConfig) -> bool:
    if conf.smoke or conf.pipeline.dry_run:
        return False
    if str(conf.task or "").casefold() in {"annotation_check", "status_check", "inventory", "prepare"}:
        return False
    train_cfg = dict(conf.train or {})
    params_cfg = dict(conf.params or {})
    return not bool(train_cfg.get("debug") or params_cfg.get("debug") or params_cfg.get("smoke"))


def _validate_explicit_training_epochs(conf: PipelineRunConfig) -> None:
    # Production runs must not inherit short debug epoch counts from historical tuning notes.
    if not _requires_explicit_training_epochs(conf):
        return
    train_cfg = dict(conf.train or {})
    if train_cfg.get("epochs") is None and train_cfg.get("max_epochs") is None:
        raise ValueError("train.epochs or train.max_epochs is required for non-smoke training runs.")


def _history_mlflow_metric_rows(result: Any) -> list[tuple[int, dict[str, float | int]]]:
    rows: list[tuple[int, dict[str, float | int]]] = []
    cumulative_sec = 0.0
    for epoch in result.history or []:
        metrics = epoch.metrics or {}
        duration = _finite_float(metrics.get("epoch_duration_sec")) or 0.0
        cumulative_sec += duration
        payload: dict[str, float | int] = {
            "epochs_total": int(epoch.epoch),
            "training_time_sec": round(cumulative_sec, 4),
        }
        f1 = _finite_float(metrics.get("val/pixel_f1"))
        if f1 is not None:
            payload["f1_pixel"] = f1
        if duration > 0:
            payload["epoch_time_sec"] = round(duration, 4)
        for key, value in metrics.items():
            if _is_training_threshold_metric(key):
                finite = _finite_float(value)
                if finite is not None:
                    payload[key] = finite
        rows.append((int(epoch.epoch), payload))
    return rows


def _log_training_history_metrics(pipeline_config: Any, run_id: str, result: Any) -> None:
    for epoch, metrics in _history_mlflow_metric_rows(result):
        mlflow_log_metrics_to_run(pipeline_config, run_id, _allowed_training_metric_payload(metrics), step=epoch)


def _run_training_pipeline(conf: PipelineRunConfig, store: Any) -> dict[str, Any]:
    summary = store.read_summary()
    mlflow_info = summary.get("mlflow") or {}
    run_id = mlflow_info.get("run_id")
    pipeline_config = load_config()
    train_pseudolabel = dict(conf.pseudolabel or {})
    train_pseudolabel["enabled"] = False
    train_only_conf = conf.model_copy(update={"pseudolabel": train_pseudolabel})
    job = _build_pipeline_job(train_only_conf)
    job.params.setdefault(
        "pipeline",
        {
            "pipeline_id": "mlsystem_experiment_pipeline",
            "run_id": store.pipeline_run_id,
            "pipeline_trace": store.conf,
        },
    )
    extra_tags, extra_params = mlflow_tuning_metadata(conf.params)
    prepared_manifest = store.run_dir / "dataset_manifest.json"
    if prepared_manifest.exists():
        job.preprocess.setdefault("prepared_dataset_manifest", str(prepared_manifest))
        job.preprocess.setdefault("use_prepared_dataset_manifest", True)
    manifest = read_json(prepared_manifest, default={}) or {}
    summary = store.read_summary()
    dataset_identity = _dataset_identity_from_summary(summary.get("dataset_identity"))
    dataset_artifacts = [
        str(path)
        for path in [
            store.run_dir / "dataset_manifest.json",
            store.run_dir / "inventory_scenes.json",
            store.run_dir / "scene_matching_report.json",
            store.run_dir / "split_summary.json",
        ]
        if path.exists() and path.is_file()
    ]
    if run_id and not str(run_id).startswith("smoke-"):
        if dataset_identity is not None:
            mlflow_log_dataset_input_to_run(pipeline_config, str(run_id), dataset_identity, context="training")
        mlflow_log_artifacts_to_run(pipeline_config, str(run_id), dataset_artifacts)
    train_loader = None
    val_loader = None
    tile_bundle = None
    if prepared_manifest.exists():
        from ..tile_preparation.api import build_datasets, train_dataloader, val_dataloader
        from ..tile_preparation.contracts import SceneInputContract, TileDatasetRequest

        annotation_path = store.run_dir / "dataset_annotation.geojson"
        image_paths_by_name = {
            str(row.get("scene_name")): row.get("image_path")
            for row in manifest.get("scene_object_counts") or []
            if row.get("scene_name") and row.get("image_path")
        }

        def _scene_contracts(rows: list[dict[str, Any]]) -> list[SceneInputContract]:
            contracts: list[SceneInputContract] = []
            for row in rows:
                name = str(row.get("entry") or row.get("name") or "")
                image_path = image_paths_by_name.get(name)
                if not image_path and row.get("key"):
                    image_path = raster_path_for_s3_key(pipeline_config, str(row.get("key")))
                if image_path:
                    contracts.append(SceneInputContract(image_path, name or Path(str(image_path)).stem))
            return contracts

        tile_size = int(job.preprocess.get("tile_size") or job.train.get("patch_size") or 512)
        stride = int(job.preprocess.get("stride") or tile_size)
        augmentation_level = int(job.preprocess.get("augmentation_level") or job.train.get("augmentation_level") or 2)
        tile_bundle = build_datasets(
            TileDatasetRequest(
                train_scenes=_scene_contracts(list(manifest.get("train_scenes") or [])),
                val_scenes=_scene_contracts(list(manifest.get("val_scenes") or [])),
                annotation_path=annotation_path,
                tile_size=tile_size,
                stride=stride,
                augmentation_level=augmentation_level,
                mosaic_enabled=job.preprocess.get("mosaic_enabled"),
                normalization_mode=str(job.preprocess.get("normalization_mode") or "uint8_255"),
                max_empty_tile_share=_optional_float(job.preprocess.get("max_empty_tile_share", job.train.get("max_empty_tile_share"))),
                max_tiles_per_scene=_optional_int(job.preprocess.get("max_tiles_per_scene", job.train.get("max_tiles_per_scene"))),
                max_train_tiles=_optional_int(job.preprocess.get("max_train_tiles", job.train.get("max_train_tiles"))),
                max_val_tiles=_optional_int(job.preprocess.get("max_val_tiles", job.train.get("max_val_tiles"))),
                augmentations=job.train.get("augmentations"),
                seed=_optional_int(job.preprocess.get("split_seed", job.train.get("seed"))),
            )
        )
        batch_size = int(job.train.get("batch_size") or 1)
        dataloader_workers = int(job.train.get("dataloader_workers", job.preprocess.get("dataloader_workers", 0)) or 0)
        train_loader = train_dataloader(tile_bundle, batch_size=batch_size, workers=dataloader_workers)
        val_loader = val_dataloader(tile_bundle, batch_size=batch_size, workers=dataloader_workers)
    _validate_explicit_training_epochs(conf)
    train_cfg = TrainConfig(
        model_name=str(job.train.get("model_name") or _model_name_from_config(conf.model)),
        input_channels=len(job.params.get("input_bands") or [1, 2, 3, 4]),
        output_channels=int(conf.model.get("out_channels") or 1),
        base_channels=int(job.train.get("base_channels") or 8),
        epochs=int(job.train.get("epochs") or job.train.get("max_epochs") or 1),
        learning_rate=float(job.train.get("learning_rate") if job.train.get("learning_rate") is not None else 5e-4),
        weight_decay=float(job.train.get("weight_decay") if job.train.get("weight_decay") is not None else 0.0),
        optimizer=str(job.train.get("optimizer") or job.train.get("optimizer_name") or "adamw"),
        scheduler=job.train.get("scheduler") or job.train.get("scheduler_name"),
        loss=dict(job.train.get("loss") or {}),
        metric_threshold=float(job.train.get("metric_threshold") or job.evaluate.get("threshold") or 0.5),
        metric_thresholds=_normalized_metric_thresholds(job.train.get("metric_thresholds") or job.evaluate.get("metric_thresholds")),
        require_gpu=bool(job.train.get("require_gpu", True)),
        max_train_batches=job.train.get("max_train_batches"),
        max_val_batches=job.train.get("max_val_batches"),
        early_stopping_patience=(job.train.get("early_stopping") or {}).get("patience")
        if isinstance(job.train.get("early_stopping"), dict) and (job.train.get("early_stopping") or {}).get("enabled", True)
        else job.train.get("early_stopping_patience"),
        initial_checkpoint_path=job.train.get("initial_checkpoint_path") or job.train.get("checkpoint_path"),
        initial_checkpoint_strict=bool(job.train.get("initial_checkpoint_strict", True)),
    )
    request = TrainRequest(
        config=train_cfg,
        train_dataloader=train_loader or job.params.get("train_dataloader"),
        val_dataloader=val_loader or job.params.get("val_dataloader"),
        output_dir=store.run_dir,
        experiment_id=conf.experiment_id,
        metadata={
            "pipeline_run_id": store.pipeline_run_id,
            "task": conf.task,
            "class_name": conf.class_name,
            "pseudolabeling.enabled": False,
        },
    )
    try:
        def _call_train_model() -> Any:
            return train_model(request)

        if run_id and not str(run_id).startswith("smoke-"):
            result_obj = mlflow_run_with_active_mlflow_run(
                pipeline_config,
                str(run_id),
                _call_train_model,
                log_system_metrics=True,
            )
        else:
            result_obj = _call_train_model()
    finally:
        if tile_bundle is not None:
            tile_bundle.train_dataset.close()
            tile_bundle.val_dataset.close()
    result = asdict(result_obj)
    result["checkpoint_path"] = result_obj.checkpoint.path if result_obj.checkpoint else None
    result["last_epoch_metrics"] = result_obj.history[-1].metrics if result_obj.history else {}
    result["mode"] = "train"
    result["mlflow"] = mlflow_info
    if dataset_identity is not None:
        result["dataset_identity"] = asdict(dataset_identity)
    params = {
        "experiment_id": conf.experiment_id,
        "task": conf.task,
        "model.name": conf.model.get("name"),
        "preprocess.tile_size": conf.preprocess.get("tile_size"),
        "preprocess.max_scenes": conf.preprocess.get("max_scenes"),
        **result_obj.mlflow_params,
        **extra_params,
    }
    if run_id and not str(run_id).startswith("smoke-"):
        mlflow_log_params_to_run(pipeline_config, str(run_id), params)
        _log_training_history_metrics(pipeline_config, str(run_id), result_obj)
        final_metrics = _allowed_training_metric_payload(result_obj.mlflow_metrics)
        final_f1 = {"f1_pixel": final_metrics["f1_pixel"]} if "f1_pixel" in final_metrics else {}
        final_rest = {key: value for key, value in final_metrics.items() if key != "f1_pixel"}
        if final_f1:
            mlflow_log_metrics_to_run(pipeline_config, str(run_id), final_f1, step=result_obj.best_epoch or result_obj.epochs_completed)
        if final_rest:
            mlflow_log_metrics_to_run(pipeline_config, str(run_id), final_rest, step=result_obj.epochs_completed)
        artifact_errors = mlflow_log_artifacts_to_run(pipeline_config, str(run_id), [*result_obj.mlflow_artifacts, *dataset_artifacts])
        if artifact_errors:
            result["warnings"] = list(result.get("warnings") or []) + artifact_errors
        mlflow_set_run_tags(pipeline_config, str(run_id), {"job_status": "training_completed", "pipeline_training_status": "success", **extra_tags})
    write_json(store.run_dir / "training_result.json", result)
    store.update_summary(training_result=result, mlflow=result.get("mlflow") or mlflow_info)
    return result


def _write_run_summaries(conf: PipelineRunConfig, store: Any) -> tuple[Path, Path]:
    summary = store.read_summary()
    result = _read_training_result(store)
    post_metrics = result.get("postprocess_metrics") or {}
    artifacts = _existing_artifacts(store)
    run_summary = {
        "schema_version": 1,
        "experiment_id": conf.experiment_id,
        "pipeline_run_id": store.pipeline_run_id,
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
        "pipeline_run_id": store.pipeline_run_id,
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


def _smoke_or_skip(conf: PipelineRunConfig, stage: str) -> dict[str, Any] | None:
    if not conf.smoke:
        return None
    skip_reason = "synthetic smoke validates Pipeline/API orchestration only; external S3, dataset, training, inference, vectorization and MLflow writes are skipped."
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


def _dataset_manifest(store: Any) -> dict[str, Any]:
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


def _write_pixel_metrics_artifacts(store: Any, metrics: dict[str, Any], counters: dict[str, Any], aliases: list[dict[str, str]], warnings: list[str]) -> dict[str, str]:
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


def _log_mlflow_metrics_if_available(store: Any, metrics: dict[str, Any], counters: dict[str, Any] | None = None) -> list[str]:
    return []


def _diagnostic_metric_payload(metrics: dict[str, Any], counters: dict[str, Any]) -> dict[str, float]:
    return {}


def _allowed_training_metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key in MODEL_METRIC_KEYS or _is_training_threshold_metric(key)}


def _is_training_threshold_metric(key: str) -> bool:
    return (
        key.startswith("val/pixel_f1_at_threshold_")
        or key.startswith("val/pixel_precision_at_threshold_")
        or key.startswith("val/pixel_recall_at_threshold_")
        or key.startswith("val/pixel_iou_at_threshold_")
        or key.startswith("val/pixel_accuracy_at_threshold_")
        or key.startswith("val/pixel_tp_at_threshold_")
        or key.startswith("val/pixel_fp_at_threshold_")
        or key.startswith("val/pixel_fn_at_threshold_")
        or key.startswith("val/pixel_tn_at_threshold_")
    )


def _normalized_metric_thresholds(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    values = raw if isinstance(raw, list | tuple) else str(raw).split(",")
    thresholds: list[float] = []
    for item in values:
        value = _finite_float(item)
        if value is None:
            continue
        value = max(0.0, min(1.0, value))
        if value not in thresholds:
            thresholds.append(value)
    return thresholds or None


def _optional_float(value: Any) -> float | None:
    return _finite_float(value) if value is not None else None


def _optional_int(value: Any) -> int | None:
    number = _finite_float(value)
    return None if number is None else int(number)


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


def _create_mlflow_run(conf: PipelineRunConfig, store: Any) -> dict[str, Any]:
    try:
        pipeline_config = load_config()
        experiment_name = conf.mlflow.get("experiment") or pipeline_config.mlflow_default_experiment
        extra_tags, extra_params = mlflow_tuning_metadata(conf.params)
        class_slug = _slug_for_run_name(conf.class_name or conf.experiment_id)
        run_name = _select_short_mlflow_run_name(
            class_slug,
            _amsterdam_today(),
            _existing_mlflow_run_names(pipeline_config, experiment_name),
        )
        experiment_tags = {}
        if conf.class_name:
            experiment_tags = {
                "class_name": conf.class_name,
                "mlsystem.class_name": conf.class_name,
                "task": conf.task,
            }
        created = mlflow_create_run(
            pipeline_config,
            experiment_name=experiment_name,
            run_name=run_name,
            tags={
                "job_id": conf.experiment_id,
                "pipeline_run_id": store.pipeline_run_id,
                "mlsystem.pipeline_run_id": store.pipeline_run_id,
                "mlsystem.full_experiment_id": conf.experiment_id,
                "mlsystem.class_slug": class_slug,
                "orchestrator": "pipeline",
                "job_status": "running",
                "execution_path": "pipeline_api",
                "task": conf.task,
                "class_name": conf.class_name or "",
                "mlsystem.class_name": conf.class_name or "",
                **extra_tags,
            },
            params={
                "experiment_id": conf.experiment_id,
                "task": conf.task,
                "images_uri": conf.images_uri,
                "layout_uri": conf.layout_uri,
                "model.name": conf.model.get("name"),
                "preprocess.tile_size": conf.preprocess.get("tile_size"),
                "preprocess.stride": conf.preprocess.get("stride"),
                "smoke": conf.smoke,
                **extra_params,
            },
            experiment_tags=experiment_tags,
        )
        experiment_id = str(created["experiment_id"])
        run_id = str(created["run_id"])
        artifact_uri = str(created.get("artifact_uri") or "")
        url_fields = {
            key: value
            for key, value in {
                "url_mlflow_run": created.get("url_mlflow_run"),
                "url_mlflow_experiment": created.get("url_mlflow_experiment"),
            }.items()
            if value
        }
        url_warnings: list[str] = []
        run_url = url_fields.get("url_mlflow_run")
        experiment_url = url_fields.get("url_mlflow_experiment")
        store.update_summary(
            mlflow={
                "experiment_name": experiment_name,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "run_url_external": run_url,
                "experiment_url_external": experiment_url,
                "run_name": run_name,
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
            warnings=[*url_warnings, *list(created.get("warnings") or [])],
            details={
                "mlflow": {
                    "experiment_name": experiment_name,
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "run_name": run_name,
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


def _finalize_mlflow_run(conf: PipelineRunConfig, store: Any) -> dict[str, Any]:
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
        pipeline_config = load_config()
        url_fields = mlflow_url_fields(pipeline_config, mlflow_info.get("experiment_id"), run_id)
        url_warnings: list[str] = []
        store.update_summary(status="success", finished_at=utc_now(), runtime_cleanup=cleanup)
        mlflow_set_run_tags(pipeline_config, str(run_id), {"job_status": "success", "pipeline_status": "success"})
        artifact_errors = mlflow_log_artifacts_to_run(pipeline_config, str(run_id), [store.summary_path])
        if artifact_errors:
            return _stage_result(
                "failed",
                error="; ".join(artifact_errors),
                counters={**final_counters, "mlflow_run_id": run_id, "mlflow_final_status": "failed"},
            )
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


def _run_dispatcher_stage(stage: str, conf: PipelineRunConfig, store: Any) -> dict[str, Any]:
    started = time.time()
    resources_before = _resource_snapshot()
    store.update_summary(
        status="running",
        experiment_config=conf.model_dump(),
        pipeline={"pipeline_id": "mlsystem_experiment_pipeline", "run_id": store.pipeline_run_id},
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
                summary="train.api completed MLSystem train-only run; pseudolabel is handled by inference_pipeline.",
                mode=training_result.get("mode"),
                epochs_completed=training_result.get("epochs_completed"),
                best_val_iou=training_result.get("best_val_iou"),
                checkpoint_path=training_result.get("checkpoint_path"),
                mlflow=training_result.get("mlflow") or {},
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
                "reason": "validation prediction disabled for this training run.",
            }
            json_path = store.run_dir / "validation_prediction_summary.json"
            txt_path = store.run_dir / "validation_prediction_summary.txt"
            write_json(json_path, prediction_summary)
            _write_simple_key_value_report(txt_path, "Validation prediction", {"summary": prediction_summary})
            result = _stage_result(
                "skipped",
                summary="Validation prediction skipped because train/predict is disabled for this training run.",
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
                "reason": "training and validation metrics are disabled for this training run.",
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
                summary="F1 computation skipped because training is disabled for this training run.",
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
            pipeline_config = load_config()
            artifact_errors = mlflow_log_artifacts_to_run(pipeline_config, str(run_id), logged_paths)
            artifacts_failed = len(artifact_errors)
        pipeline_config = load_config()
        url_fields = mlflow_url_fields(pipeline_config, mlflow_info.get("experiment_id"), run_id)
        url_warnings: list[str] = []
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
        raise ValueError(f"Unknown Pipeline MLSystem stage: {stage}")

    result["duration_sec"] = round(time.time() - started, 3)
    monitor_payload = monitor.stop()
    resources_after = _resource_snapshot()
    if not result.get("is_smoke_synthetic"):
        _attach_stage_runtime_counters(result, stage, resources_before, resources_after)
        result["resources"] = {"before": resources_before, "after": resources_after, "monitor": _safe_rel(store.stage_dir / f"{stage}.resources.json", store.run_dir), "sample_count": len(monitor_payload.get("samples") or [])}
    return store.write_stage(stage, result)


def run_stage(stage: str, config: PipelineRunConfig, store: Any) -> dict[str, Any]:
    from .stages.context import StageContext
    from .stages.registry import get_stage_entrypoint
    from .stages.report import StageFailure, StageReport

    if stage in DISPATCHER_STAGE_NAMES:
        return _run_dispatcher_stage(stage, config, store)

    try:
        entrypoint = get_stage_entrypoint(stage)
    except KeyError:
        known = ", ".join(sorted(DISPATCHER_STAGE_NAMES))
        raise ValueError(f"Unknown Pipeline MLSystem stage: {stage}. Known stages: {known}")

    started = time.time()
    resources_before = _resource_snapshot()
    conf = config
    logger = logging.getLogger(f"mlsystem.pipeline.{stage}")
    store.update_summary(
        status="running",
        experiment_config=conf.model_dump(),
        pipeline={"pipeline_id": "mlsystem_experiment_pipeline", "run_id": store.pipeline_run_id},
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
                run_id=store.pipeline_run_id,
                config=conf,
                raw_conf=conf.model_dump(mode="json"),
                status_dir=store.state_dir,
                store=store,
                logger=logger,
            )
            report = entrypoint(context)
            result = report.to_stage_payload()
            log_text = report.to_pipeline_log()
            if report.status == "failed":
                logger.error(log_text)
                failure_to_raise = RuntimeError(result.get("error") or report.summary or f"{stage} failed")
            else:
                logger.info(log_text)
    except StageFailure as exc:
        report = exc.report
        result = report.to_stage_payload()
        logger.error(report.to_pipeline_log())
        failure_to_raise = exc
    except Exception as exc:  # noqa: BLE001
        report = StageReport(stage_id=stage, status="failed", errors=[str(exc)])
        result = report.to_stage_payload()
        logger.error(report.to_pipeline_log())
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


def main() -> None:
    raise SystemExit("Use python -m mlsystem.src.train_pipeline.cli for pipeline lifecycle execution.")


if __name__ == "__main__":
    main()
