from __future__ import annotations

import argparse
import io
import json
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from .job_store import JobStore, finish_job
from .models import JobError, StageStartRequest, utc_now
from .security import mask_secrets, mask_text


def run_job(job_id: str, job_root: Path | None = None) -> int:
    store = JobStore(job_root)
    job = store.update_job(job_id, state="running", started_at=utc_now())
    request = StageStartRequest.model_validate(store.read_request(job_id))
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        from ..pipeline_runner.config import PipelineRunConfig
        from ..pipeline_runner.run_store import PipelineRunStore
        from ..pipeline_runner.stages import run_stage

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            config = PipelineRunConfig.model_validate(request.experiment_config)
            pipeline_store = PipelineRunStore(Path(request.status_root), job.pipeline_run_id, config)
            result = run_stage(job.stage, config, pipeline_store, job.pipeline_run_id)
        report = mask_secrets(result)
        finish_job(store, job, state="succeeded", report=report)
        _write_tails(store, job_id, stdout_buffer.getvalue(), stderr_buffer.getvalue())
        return 0
    except Exception as exc:  # noqa: BLE001 - worker must persist diagnostic details.
        tb = traceback.format_exc()
        error = JobError(
            type=type(exc).__name__,
            message=mask_text(str(exc)),
            traceback_tail=mask_text("\n".join(tb.splitlines()[-40:])),
        )
        report = _load_written_stage_report(request, job.stage, job.pipeline_run_id)
        if not report:
            report = {"stage": job.stage, "status": "failed"}
        report["status"] = "failed"
        report["error"] = error.model_dump()
        report.setdefault("errors", [error.message])
        finish_job(store, job, state="failed", report=mask_secrets(report), error=error)
        _write_tails(store, job_id, mask_text(stdout_buffer.getvalue()) or "", mask_text(stderr_buffer.getvalue() + "\n" + tb) or "")
        return 1


def _write_tails(store: JobStore, job_id: str, stdout_text: str, stderr_text: str, max_chars: int = 20000) -> None:
    job_dir = store.job_dir(job_id)
    (job_dir / "stdout_tail.txt").write_text(stdout_text[-max_chars:], encoding="utf-8")
    (job_dir / "stderr_tail.txt").write_text(stderr_text[-max_chars:], encoding="utf-8")


def _load_written_stage_report(request: StageStartRequest, stage: str, pipeline_run_id: str) -> dict | None:
    experiment_id = request.experiment_config.get("experiment_id") if isinstance(request.experiment_config, dict) else None
    run_dir_name = _safe_run_id(pipeline_run_id or str(experiment_id or "run"))
    path = Path(request.status_root) / run_dir_name / "stages" / f"{stage}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "run"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one persisted MLSystem API stage job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-root", type=Path, default=None)
    args = parser.parse_args()
    return run_job(args.job_id, args.job_root)


if __name__ == "__main__":
    raise SystemExit(main())
