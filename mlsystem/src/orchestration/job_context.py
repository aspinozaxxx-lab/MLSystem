from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..job_schema import JobSpec
from ..pipeline_config import PipelineConfig


@dataclass
class JobContext:
    config: PipelineConfig
    job: JobSpec
    claim_id: str
    running_dir: Path
    experiment_dir: Path
    job_log: Path
    queue_metadata: dict[str, Any]
