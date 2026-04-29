from __future__ import annotations
import json
import platform
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .pipeline_config import PipelineConfig

MAX_ARTIFACT_BYTES = 20_000_000
MLFLOW_NOTE_TAG = "mlflow.note.content"
MLFLOW_EXCLUDED_ARTIFACT_NAMES = {
    "accepted.geojson.gz",
    "accepted.gpkg",
    "accepted_debug.geojson",
    "windows_preview.geojson",
}

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _resource_snapshot() -> dict[str, Any]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": round(vm.used / (1024 ** 3), 3),
            "ram_total_gb": round(vm.total / (1024 ** 3), 3),
            "ram_percent": vm.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3), 3),
            "disk_percent": disk.percent,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

@contextmanager
def trace_stage(name: str, attributes: dict[str, Any] | None = None):
    """MLflow trace/span wrapper that records only compact scalar attributes."""
    attributes = attributes or {}
    try:
        import mlflow
        if hasattr(mlflow, "start_span"):
            with mlflow.start_span(name=name) as span:
                if hasattr(span, "set_attributes"):
                    span.set_attributes({key: str(value) for key, value in attributes.items() if value is not None})
                yield
            return
    except Exception:
        pass
    yield

def compact_run_label(job_id: str, model_name: str | None = None, tile_size: int | str | None = None) -> str:
    """Return a short MLflow run label that stays useful when UI columns are narrow."""
    lower = job_id.lower()
    exp_no = None
    if "exp" in lower:
        suffix = lower.split("exp", 1)[1]
        digits = "".join(ch for ch in suffix[:3] if ch.isdigit())
        if digits:
            exp_no = f"E{int(digits):02d}"
    elif "smoke" in lower or "smk" in lower:
        exp_no = "SMK"
    prefix = exp_no or job_id[:12]
    model = (model_name or "").lower()
    if "resnet18" in model or "_r18" in lower or "r18" in lower:
        model_part = "r18"
    elif "resnet34" in model or "_r34" in lower or "r34" in lower:
        model_part = "r34"
    elif "resnet50" in model or "_r50" in lower or "r50" in lower:
        model_part = "r50"
    elif "tiny" in model or "tiny" in lower:
        model_part = "tiny"
    else:
        model_part = model.replace("unet_", "").replace("resnet", "r")[:8] or "run"
    tile = str(tile_size or "")
    if not tile:
        for token in lower.replace("-", "_").split("_"):
            if token.startswith("t") and token[1:].isdigit():
                tile = token[1:]
                break
    return f"{prefix}-{model_part}-t{tile}" if tile else f"{prefix}-{model_part}"

def build_run_note(
    *,
    job_id: str,
    run_label: str,
    model_name: str | None,
    tile_size: int | str | None,
    stride: int | str | None,
    data_uri: str | None = None,
    layout_uri: str | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    counts: dict[str, Any] | None = None,
) -> str:
    metrics = metrics or {}
    artifacts = artifacts or {}
    warnings = warnings or []
    counts = counts or {}
    lines = [
        f"# {run_label}",
        "",
        f"- job_id: `{job_id}`",
        f"- model: `{model_name or 'unknown'}`",
        f"- tile/stride: `{tile_size or 'n/a'}` / `{stride or 'n/a'}`",
    ]
    if data_uri:
        lines.append(f"- images: `{data_uri}`")
    if layout_uri:
        lines.append(f"- layout: `{layout_uri}`")
    if counts:
        lines.extend(["", "## Dataset"])
        for key in ("train_scene_count", "val_scene_count", "test_scene_count", "positive_scene_count", "negative_scene_count"):
            if counts.get(key) is not None:
                lines.append(f"- `{key}`: `{counts[key]}`")
    metric_keys = [
        "best_val_object_f1",
        "val/object_f1",
        "val/object_precision",
        "val/object_recall",
        "test/object_f1",
        "test/object_precision",
        "test/object_recall",
        "pseudolabel/object_f1",
        "pseudolabel/object_precision",
        "pseudolabel/object_recall",
        "best_val_iou",
        "val/iou",
        "val/dice",
        "val/pixel_f1",
        "test/pixel_f1",
        "test/pixel_iou",
        "test/pixel_dice",
        "accepted_objects",
        "accepted_geojson_mb",
        "top500_applied",
    ]
    visible_metrics = {key: metrics.get(key) for key in metric_keys if metrics.get(key) is not None}
    if visible_metrics:
        lines.extend(["", "## Metrics"])
        for key, value in visible_metrics.items():
            lines.append(f"- `{key}`: `{value}`")
    if artifacts:
        lines.extend(["", "## Artifacts"])
        for key, value in artifacts.items():
            if value:
                lines.append(f"- `{key}`: `{value}`")
    if warnings:
        lines.extend(["", "## Warnings"])
        for item in warnings[:10]:
            lines.append(f"- {item}")
    return "\n".join(lines)

def _run_url(base_uri: str, experiment_id: str, run_id: str) -> str:
    return f"{base_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"

def _stringify_param(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

def flatten_params(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_params(value, name))
        elif value is not None:
            result[name] = _stringify_param(value)
    return result

def check_mlflow(config: PipelineConfig) -> dict[str, Any]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
        client = MlflowClient()
        experiments = client.search_experiments(max_results=50)
        return {
            "ok": True,
            "tracking_uri_internal": config.mlflow_tracking_uri_internal,
            "tracking_uri_external": config.mlflow_tracking_uri_external,
            "default_experiment": config.mlflow_default_experiment,
            "experiment_count": len(experiments),
            "experiments": [
                {"id": e.experiment_id, "name": e.name, "artifact_location": e.artifact_location}
                for e in experiments
            ],
        }
    except Exception as exc:
        return {
            "ok": False,
            "tracking_uri_internal": config.mlflow_tracking_uri_internal,
            "tracking_uri_external": config.mlflow_tracking_uri_external,
            "error": f"{type(exc).__name__}: {exc}",
        }

def setup_deforest_experiment(config: PipelineConfig, experiment_name: str = "mlsystem-deforest") -> dict[str, Any]:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    experiment = mlflow.set_experiment(experiment_name)
    client = MlflowClient()
    note = "\n".join(
        [
            "# MLSystem deforest experiments",
            "",
            "Binary semantic segmentation of forest clear-cuts from 4-channel Kanopus imagery.",
            "",
            "- class_name: `deforest`",
            "- task: `binary semantic segmentation`",
            "- input_bands: `[1, 2, 3, 4]`",
            "- preview_bands: `[4, 1, 2]`",
            "- storage: `s3://mlsystems`",
            "- images: `s3://mlsystems/images/`",
            "- layouts: `s3://mlsystems/layouts/deforest/`",
            "- reports: `results/reports/deforest_experiments_summary.md`",
            "- acceptance_metric: `object_f1`",
            "- object_iou_threshold: `0.5`",
            "",
            "Acceptance metric follows CHTZ Appendix G: object-level F1 over polygons with one-to-one matching and IoU > 0.5.",
            "Pixel metrics are auxiliary diagnostics only. Object F1 is computed after vectorization/postprocess.",
        ]
    )
    tags = {
        MLFLOW_NOTE_TAG: note,
        "class_name": "deforest",
        "task": "binary semantic segmentation",
        "input_bands": "[1,2,3,4]",
        "preview_bands": "[4,1,2]",
        "storage": "s3://mlsystems",
        "images_uri": "s3://mlsystems/images/",
        "layout_uri": "s3://mlsystems/layouts/deforest/",
        "acceptance_metric": "object_f1",
        "object_iou_threshold": "0.5",
        "metric_method": "CHTZ Appendix G polygon object matching",
        "pixel_metrics_role": "auxiliary",
        "data_status": "full_or_partial_scene_matching_depends_on_job",
        "results_summary": "results/reports/deforest_experiments_summary.md",
    }
    for key, value in tags.items():
        client.set_experiment_tag(experiment.experiment_id, key, value)
    return {
        "ok": True,
        "experiment_name": experiment_name,
        "experiment_id": experiment.experiment_id,
        "tracking_uri_internal": config.mlflow_tracking_uri_internal,
        "tracking_uri_external": config.mlflow_tracking_uri_external,
    }

class MLflowJobRun:
    def __init__(
        self,
        config: PipelineConfig,
        experiment_name: str,
        run_name: str,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.experiment_name = experiment_name or config.mlflow_default_experiment
        self.run_name = run_name
        self.params = params or {}
        self.tags = tags or {}
        self.existing_run_id = run_id
        self._mlflow = None
        self._run = None
        self.experiment_id: str | None = None
        self.run_id: str | None = None
        self.started_at: str | None = None
        self.resource_start: dict[str, Any] | None = None

    def __enter__(self) -> "MLflowJobRun":
        import mlflow
        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri_internal)
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        self.started_at = utc_now()
        self.resource_start = _resource_snapshot()
        try:
            if self.existing_run_id:
                self._run = mlflow.start_run(run_id=self.existing_run_id, log_system_metrics=True)
            else:
                self._run = mlflow.start_run(run_name=self.run_name, log_system_metrics=True)
        except TypeError:
            if self.existing_run_id:
                self._run = mlflow.start_run(run_id=self.existing_run_id)
            else:
                self._run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self._run.info.run_id
        mlflow.set_tags(
            {
                "mlflow.runName": self.run_name,
                "mlsystem.host": socket.gethostname(),
                "mlsystem.cpu_only": str(self.config.cpu_only).lower(),
                "mlsystem.tracking_uri_internal": self.config.mlflow_tracking_uri_internal,
                "mlsystem.tracking_uri_external": self.config.mlflow_tracking_uri_external,
                "mlsystem.started_at": self.started_at,
                **self.tags,
            }
        )
        self.log_params(
            {
                "system.hostname": socket.gethostname(),
                "system.platform": platform.platform(),
                "system.python": sys.version.split()[0],
                "system.cpu_only": self.config.cpu_only,
                **self.params,
            }
        )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if not self._mlflow:
            return
        finished_at = utc_now()
        resource_finish = _resource_snapshot()
        self.set_tags(
            {
                "mlsystem.finished_at": finished_at,
                "mlsystem.run_exception": "" if exc_type is None else str(exc_type),
            }
        )
        final_metrics: dict[str, float] = {}
        for key, value in resource_finish.items():
            if isinstance(value, (int, float)):
                final_metrics[f"resource/final_{key}"] = float(value)
        if final_metrics:
            self.log_metrics(final_metrics)
        self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._mlflow:
            return
        for key, value in flatten_params(params).items():
            self._mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict[str, float | int | None], step: int | None = None) -> None:
        if not self._mlflow:
            return
        clean = {key: float(value) for key, value in metrics.items() if value is not None}
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, Any]) -> None:
        if not self._mlflow:
            return
        self._mlflow.set_tags({key: "" if value is None else str(value) for key, value in tags.items()})

    def log_artifacts(self, artifacts: list[Path]) -> None:
        if not self._mlflow:
            return
        for artifact in artifacts:
            if artifact.name in MLFLOW_EXCLUDED_ARTIFACT_NAMES:
                continue
            if artifact.exists() and artifact.is_file() and artifact.stat().st_size < MAX_ARTIFACT_BYTES:
                self._mlflow.log_artifact(str(artifact))

    def resource_summary(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "start": self.resource_start,
            "finish": _resource_snapshot(),
            "system_metrics_source": "mlflow.start_run(log_system_metrics=True); final resource snapshot stored in summaries",
        }

    def log_table(self, data: Any, artifact_file: str) -> bool:
        if not self._mlflow or not hasattr(self._mlflow, "log_table"):
            return False
        try:
            self._mlflow.log_table(data=data, artifact_file=artifact_file)
            return True
        except Exception:
            return False

    def result(self) -> dict[str, Any]:
        if not self.experiment_id or not self.run_id:
            return {"ok": False, "error": "MLflow run has not started"}
        return {
            "ok": True,
            "tracking_uri_internal": self.config.mlflow_tracking_uri_internal,
            "tracking_uri_external": self.config.mlflow_tracking_uri_external,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "internal_run_url": _run_url(self.config.mlflow_tracking_uri_internal, self.experiment_id, self.run_id),
            "external_run_url": _run_url(self.config.mlflow_tracking_uri_external, self.experiment_id, self.run_id),
            "run_url_internal": _run_url(self.config.mlflow_tracking_uri_internal, self.experiment_id, self.run_id),
            "run_url_external": _run_url(self.config.mlflow_tracking_uri_external, self.experiment_id, self.run_id),
        }

def start_job_run(
    config: PipelineConfig,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    run_id: str | None = None,
) -> MLflowJobRun:
    return MLflowJobRun(config, experiment_name, run_name, params=params, tags=tags, run_id=run_id)

def set_run_tags(config: PipelineConfig, run_id: str, tags: dict[str, Any]) -> None:
    import mlflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({key: "" if value is None else str(value) for key, value in tags.items()})

def create_queued_job_run(
    config: PipelineConfig,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any],
    tags: dict[str, Any],
    artifacts: list[Path] | None = None,
    queue_position: int | None = None,
) -> dict[str, Any]:
    queued_tags = {
        "job_status": "queued",
        "queue_state": "pending",
        **{key: "" if value is None else str(value) for key, value in tags.items()},
    }
    with start_job_run(config, experiment_name, run_name, params=params, tags=queued_tags) as run:
        if queue_position is not None:
            run.log_metrics({"queue/position": queue_position}, step=0)
        run.log_artifacts(artifacts or [])
        return run.result()

def log_lightweight_run(
    config: PipelineConfig,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    artifacts: list[Path] | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    try:
        with start_job_run(config, experiment_name, run_name, params=params, tags=tags) as run:
            run.log_metrics(metrics or {})
            run.log_artifacts(artifacts or [])
            return run.result()
    except Exception as exc:
        return {
            "ok": False,
            "tracking_uri_internal": config.mlflow_tracking_uri_internal,
            "tracking_uri_external": config.mlflow_tracking_uri_external,
            "error": f"{type(exc).__name__}: {exc}",
        }
