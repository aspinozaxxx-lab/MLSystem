from __future__ import annotations

from pathlib import Path
from typing import Any

from ..job_schema import JobSpec
from ..mlflow_adapter.api import MLflowJobRun
from ..pipeline_config import PipelineConfig


class PredictionPipeline:
    def run_debug_pseudolabel(
        self,
        config: PipelineConfig,
        job: JobSpec,
        experiment_dir: Path,
        mlflow_run: MLflowJobRun,
        job_log: Path,
        log_fn: Any,
    ) -> dict[str, Any]:
        from ..real_train import run_debug_pseudolabel

        return run_debug_pseudolabel(config, job, experiment_dir, mlflow_run, job_log, log_fn)
