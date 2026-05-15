from __future__ import annotations
import json
import os
import platform
import socket
import sys
import time
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
MLFLOW_EXCLUDED_METRIC_PREFIXES = (
    "recource/",
    "recource.",
    "recource_",
    "resource/",
    "resource.",
    "resource_",
    "resourse/",
    "resourse.",
    "resourse_",
    "resources/",
    "resources.",
    "resources_",
)

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
    start_span = None
    try:
        import mlflow
        start_span = getattr(mlflow, "start_span", None)
    except Exception:
        start_span = None
    if start_span:
        with start_span(name=name) as span:
            if hasattr(span, "set_attributes"):
                span.set_attributes({key: str(value) for key, value in attributes.items() if value is not None})
            yield
        return
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


def mlflow_url_fields(config: PipelineConfig, experiment_id: str | None, run_id: str | None) -> dict[str, Any]:
    if not experiment_id or not run_id:
        return {}
    return {
        "url_mlflow_run": _run_url(config.mlflow_tracking_uri_external, str(experiment_id), str(run_id)),
        "url_mlflow_experiment": f"{config.mlflow_tracking_uri_external.rstrip('/')}/#/experiments/{experiment_id}",
    }

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


def get_or_create_experiment_id(config: PipelineConfig, experiment_name: str, *, attempts: int = 5) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    client = MlflowClient()
    for attempt in range(max(1, attempts)):
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        try:
            return str(client.create_experiment(experiment_name))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}".lower()
            duplicate = "resource_already_exists" in message or ("already exists" in message and "experiment" in message)
            if not duplicate or attempt == attempts - 1:
                raise
            time.sleep(0.2 * (attempt + 1))
    raise RuntimeError(f"MLflow experiment was not created: {experiment_name}")


def set_experiment_tags(config: PipelineConfig, experiment_id: str, tags: dict[str, Any]) -> list[str]:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    client = MlflowClient()
    warnings: list[str] = []
    for key, value in tags.items():
        try:
            client.set_experiment_tag(experiment_id, str(key), "" if value is None else str(value))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            lowered = message.lower()
            if "experiment_tag_pk" in lowered or ("duplicate key" in lowered and "experiment_tags" in lowered):
                warnings.append(f"Skipped concurrent MLflow experiment tag write: {key}")
            else:
                raise
    return warnings


def create_run(
    config: PipelineConfig,
    *,
    experiment_name: str,
    run_name: str,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    experiment_tags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    experiment_id = get_or_create_experiment_id(config, experiment_name)
    tag_warnings = set_experiment_tags(config, experiment_id, experiment_tags or {}) if experiment_tags else []
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.set_tags({key: "" if value is None else str(value) for key, value in (tags or {}).items()})
        clean_params = flatten_params(params or {})
        if clean_params:
            mlflow.log_params(clean_params)
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
    return {
        "ok": True,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "run_name": run_name,
        "artifact_uri": artifact_uri,
        "tracking_uri": config.mlflow_tracking_uri_internal,
        "tracking_uri_internal": config.mlflow_tracking_uri_internal,
        "tracking_uri_external": config.mlflow_tracking_uri_external,
        "warnings": tag_warnings,
        **mlflow_url_fields(config, experiment_id, run_id),
    }


def log_metrics_to_run(config: PipelineConfig, run_id: str, metrics: dict[str, Any], *, step: int | None = None) -> None:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    clean = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None and _is_number(value) and not _is_excluded_metric_key(str(key))
    }
    if not clean:
        return
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(clean, step=step)


def log_params_to_run(config: PipelineConfig, run_id: str, params: dict[str, Any]) -> None:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    clean = flatten_params(params)
    if not clean:
        return
    with mlflow.start_run(run_id=run_id):
        for key, value in clean.items():
            try:
                mlflow.log_param(key, value)
            except Exception:
                continue


def log_artifacts_to_run(config: PipelineConfig, run_id: str, artifacts: list[str | Path]) -> list[str]:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    errors: list[str] = []
    with mlflow.start_run(run_id=run_id):
        for artifact in artifacts:
            path = Path(artifact)
            try:
                if path.exists() and path.is_file() and path.name not in MLFLOW_EXCLUDED_ARTIFACT_NAMES:
                    mlflow.log_artifact(str(path))
            except Exception as exc:
                errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
    return errors


def set_run_tags(config: PipelineConfig, run_id: str, tags: dict[str, Any]) -> None:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({key: "" if value is None else str(value) for key, value in tags.items()})


def download_run_artifacts(config: PipelineConfig | None, run_id: str) -> Path:
    import mlflow

    if config is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    return Path(mlflow.artifacts.download_artifacts(run_id=run_id))


def get_run(config: PipelineConfig | None, run_id: str) -> Any:
    import mlflow

    if config is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    return mlflow.tracking.MlflowClient().get_run(run_id)


def list_artifact_paths(config: PipelineConfig | None, run_id: str) -> list[str]:
    import mlflow

    if config is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    return [item.path for item in mlflow.tracking.MlflowClient().list_artifacts(run_id)]


def search_child_runs(config: PipelineConfig | None, experiment_id: str, parent_run_id: str, *, max_results: int = 200) -> list[Any]:
    import mlflow

    if config is not None:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    client = mlflow.tracking.MlflowClient()
    return list(
        client.search_runs(
            [experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            max_results=max_results,
            order_by=["attributes.start_time DESC"],
        )
    )


def search_runs(
    config: PipelineConfig,
    *,
    experiment_name: str,
    filter_string: str = "",
    max_results: int = 1000,
    order_by: list[str] | None = None,
) -> list[dict[str, Any]]:
    import mlflow

    mlflow.set_tracking_uri(config.mlflow_tracking_uri_internal)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by or ["attributes.start_time DESC"],
    )
    result: list[dict[str, Any]] = []
    for run in runs:
        result.append(
            {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "artifact_uri": run.info.artifact_uri,
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
                "tags": dict(run.data.tags),
                **mlflow_url_fields(config, run.info.experiment_id, run.info.run_id),
            }
        )
    return result


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in {float("inf"), float("-inf")}

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
        self._existing_param_keys: set[str] = set()

    def __enter__(self) -> "MLflowJobRun":
        import mlflow
        from mlflow.tracking import MlflowClient

        self._mlflow = mlflow
        os.environ.pop("MLFLOW_RUN_ID", None)
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri_internal)
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        self.started_at = utc_now()
        self.resource_start = _resource_snapshot()
        if self.existing_run_id:
            self._run = mlflow.start_run(run_id=self.existing_run_id)
        else:
            self._run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self._run.info.run_id
        if self.existing_run_id:
            try:
                self._existing_param_keys = set(MlflowClient().get_run(self.run_id).data.params)
            except Exception:
                self._existing_param_keys = set()
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
        self._mlflow.end_run(status="FAILED" if exc_type else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._mlflow:
            return
        for key, value in flatten_params(params).items():
            if key in self._existing_param_keys:
                continue
            self._mlflow.log_param(key, value)
            self._existing_param_keys.add(key)

    def log_metrics(self, metrics: dict[str, float | int | None], step: int | None = None) -> None:
        if not self._mlflow:
            return
        clean = {
            key: float(value)
            for key, value in metrics.items()
            if value is not None and not _is_excluded_metric_key(str(key))
        }
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
            always_log = artifact.name.endswith(".accepted.geojson")
            if artifact.exists() and artifact.is_file() and (always_log or artifact.stat().st_size < MAX_ARTIFACT_BYTES):
                self._mlflow.log_artifact(str(artifact))

    def resource_summary(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "start": self.resource_start,
            "finish": _resource_snapshot(),
            "system_metrics_source": "local resource snapshot stored in summaries; MLflow resource/system metrics disabled",
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


def _is_excluded_metric_key(key: str) -> bool:
    lowered = key.lower()
    return any(lowered.startswith(prefix) for prefix in MLFLOW_EXCLUDED_METRIC_PREFIXES)

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
