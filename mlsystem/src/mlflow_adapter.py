from __future__ import annotations
import json
import platform
import socket
import sys
from pathlib import Path
from typing import Any
from .pipeline_config import PipelineConfig

MAX_ARTIFACT_BYTES = 5_000_000

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

    def __enter__(self) -> "MLflowJobRun":
        import mlflow
        self._mlflow = mlflow
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri_internal)
        experiment = mlflow.set_experiment(self.experiment_name)
        self.experiment_id = experiment.experiment_id
        if self.existing_run_id:
            self._run = mlflow.start_run(run_id=self.existing_run_id)
        else:
            self._run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self._run.info.run_id
        mlflow.set_tags(
            {
                "mlsystem.host": socket.gethostname(),
                "mlsystem.cpu_only": str(self.config.cpu_only).lower(),
                "mlsystem.tracking_uri_internal": self.config.mlflow_tracking_uri_internal,
                "mlsystem.tracking_uri_external": self.config.mlflow_tracking_uri_external,
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
            if artifact.exists() and artifact.is_file() and artifact.stat().st_size < MAX_ARTIFACT_BYTES:
                self._mlflow.log_artifact(str(artifact))

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
