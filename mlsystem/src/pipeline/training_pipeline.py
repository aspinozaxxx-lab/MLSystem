from __future__ import annotations

from pathlib import Path
from typing import Any

from ..job_schema import JobSpec
from ..mlflow_adapter import MLflowJobRun
from ..pipeline_config import PipelineConfig


class TrainingPipeline:
    """Compatibility pipeline wrapper.

    The orchestration layer talks to this class instead of importing the legacy
    training function directly. The heavy implementation is still delegated to
    the real_train facade until all internals are extracted.
    """

    def run(
        self,
        config: PipelineConfig,
        job: JobSpec,
        experiment_dir: Path,
        mlflow_run: MLflowJobRun,
        job_log: Path,
        log_fn: Any,
    ) -> dict[str, Any]:
        from ..real_train import run_real_train

        return run_real_train(config, job, experiment_dir, mlflow_run, job_log, log_fn)
