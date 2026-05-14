from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineRunConfig
from .run_store import PipelineRunStore


def log_pipeline_metadata(config: PipelineRunConfig, store: PipelineRunStore, run_id: str) -> list[str]:
    if not config.pipeline.log_mlflow:
        return []
    warnings: list[str] = []
    bound = store.bind(run_id, config)
    summary = bound.read_summary()
    mlflow_info = summary.get("mlflow") or bound.read_run().get("mlflow") or {}
    mlflow_run_id = mlflow_info.get("run_id")
    if not mlflow_run_id or str(mlflow_run_id).startswith("smoke-"):
        return warnings
    try:
        import mlflow

        from ..pipeline_config import load_config

        pipeline_config = load_config()
        mlflow.set_tracking_uri(pipeline_config.mlflow_tracking_uri_internal)
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.set_tags({"pipeline.run_id": run_id, "pipeline.final_status": bound.read_run().get("state")})
            mlflow.log_param("pipeline.stages", ",".join(config.pipeline.stages)[:500])
            mlflow.log_param("pipeline.trace", json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)[:500])
            for stage_report in sorted((bound.stage_dir).glob("*.json")):
                _log_artifact_if_reasonable(mlflow, stage_report)
            _log_artifact_if_reasonable(mlflow, bound.summary_path)
            _log_artifact_if_reasonable(mlflow, bound.log_dir / "pipeline.log")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"MLflow pipeline metadata logging skipped: {type(exc).__name__}: {exc}")
    return warnings


def _log_artifact_if_reasonable(mlflow: Any, path: Path, *, max_bytes: int = 20 * 1024 * 1024) -> None:
    if not path.exists() or not path.is_file():
        return
    if path.stat().st_size > max_bytes:
        return
    mlflow.log_artifact(str(path))
