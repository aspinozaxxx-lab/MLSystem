from __future__ import annotations

import json
import shutil
from pathlib import Path

from mlsystem.src.api.job_runner import JobRunner
from mlsystem.src.api.job_store import JobStore
from mlsystem.src.api.models import StageStartRequest
from mlsystem.src.api.stage_routes import stages_payload, start_stage


def main() -> int:
    test_dir = Path(__file__).resolve().parent
    output_dir = test_dir / "output"
    job_root = output_dir / "jobs"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages = stages_payload()
    (output_dir / "stages.json").write_text(json.dumps(stages, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    runner = JobRunner(JobStore(job_root))
    response = start_stage(
        "api_stage_smoke",
        "inventory_scenes",
        StageStartRequest(experiment_config={"experiment_id": "api_stage_smoke"}, dry_run=True, status_root=str(output_dir / "status")),
        runner,
    )
    job = runner.store.read_job(response.job_id)
    (output_dir / "dry_run_job.json").write_text(json.dumps(job.model_dump(), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"stages={output_dir / 'stages.json'}")
    print(f"dry_run_job={output_dir / 'dry_run_job.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
