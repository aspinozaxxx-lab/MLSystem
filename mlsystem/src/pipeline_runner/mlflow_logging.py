from __future__ import annotations

import json

from .config import PipelineRunConfig
from .run_store import PipelineRunStore
from ..mlflow_adapter import log_artifacts_to_run, log_params_to_run, set_run_tags
from ..pipeline_config import load_config


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
        pipeline_config = load_config()
        set_run_tags(pipeline_config, str(mlflow_run_id), {"pipeline.run_id": run_id, "pipeline.final_status": bound.read_run().get("state")})
        log_params_to_run(
            pipeline_config,
            str(mlflow_run_id),
            {
                "pipeline.stages": ",".join(config.pipeline.stages)[:500],
                "pipeline.trace": json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)[:500],
            },
        )
        artifacts = [
            path
            for path in [*sorted((bound.stage_dir).glob("*.json")), bound.summary_path, bound.log_dir / "pipeline.log"]
            if path.exists() and path.is_file() and path.stat().st_size <= 20 * 1024 * 1024
        ]
        artifact_errors = log_artifacts_to_run(pipeline_config, str(mlflow_run_id), artifacts)
        warnings.extend(f"MLflow artifact logging skipped: {item}" for item in artifact_errors)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"MLflow pipeline metadata logging skipped: {type(exc).__name__}: {exc}")
    return warnings
