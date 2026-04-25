from __future__ import annotations
from pathlib import Path
from typing import Any
from .pipeline_config import PipelineConfig

def check_mlflow(config: PipelineConfig) -> dict[str, Any]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        client = MlflowClient()
        experiments = client.search_experiments(max_results=20)
        return {"ok": True, "tracking_uri": config.mlflow_tracking_uri, "experiment_count": len(experiments), "experiments": [{"id": e.experiment_id, "name": e.name, "artifact_location": e.artifact_location} for e in experiments]}
    except Exception as exc:
        return {"ok": False, "tracking_uri": config.mlflow_tracking_uri, "error": f"{type(exc).__name__}: {exc}"}

def log_lightweight_run(config: PipelineConfig, experiment_name: str, run_name: str, params: dict[str, Any] | None = None, metrics: dict[str, float] | None = None, artifacts: list[Path] | None = None, tags: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            for key, value in (params or {}).items():
                if value is not None:
                    mlflow.log_param(key, value)
            for key, value in (metrics or {}).items():
                mlflow.log_metric(key, float(value))
            for artifact in artifacts or []:
                if artifact.exists() and artifact.is_file() and artifact.stat().st_size < 5_000_000:
                    mlflow.log_artifact(str(artifact))
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id
        return {"ok": True, "tracking_uri": config.mlflow_tracking_uri, "experiment_name": experiment_name, "experiment_id": experiment_id, "run_id": run_id, "run_url": f"{config.mlflow_tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"}
    except Exception as exc:
        return {"ok": False, "tracking_uri": config.mlflow_tracking_uri, "error": f"{type(exc).__name__}: {exc}"}
