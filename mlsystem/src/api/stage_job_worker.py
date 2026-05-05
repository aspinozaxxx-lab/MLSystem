from __future__ import annotations

import argparse
import io
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
        from ..pipeline.airflow_tasks import run_stage

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = run_stage(job.stage, request.experiment_config, job.airflow_run_id, Path(request.status_root))
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
        finish_job(store, job, state="failed", report={"status": "failed", "error": error.model_dump()}, error=error)
        _write_tails(store, job_id, mask_text(stdout_buffer.getvalue()) or "", mask_text(stderr_buffer.getvalue() + "\n" + tb) or "")
        return 1


def _write_tails(store: JobStore, job_id: str, stdout_text: str, stderr_text: str, max_chars: int = 20000) -> None:
    job_dir = store.job_dir(job_id)
    (job_dir / "stdout_tail.txt").write_text(stdout_text[-max_chars:], encoding="utf-8")
    (job_dir / "stderr_tail.txt").write_text(stderr_text[-max_chars:], encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one persisted MLSystem API stage job.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-root", type=Path, default=None)
    args = parser.parse_args()
    return run_job(args.job_id, args.job_root)


if __name__ == "__main__":
    raise SystemExit(main())
