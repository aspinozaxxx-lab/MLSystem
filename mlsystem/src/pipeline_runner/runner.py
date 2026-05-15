from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..api.security import mask_text
from .config import PipelineRunConfig
from .run_store import PipelineRunStore, utc_now


class PipelineRun(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str
    state: str
    progress_percent: int = 0
    current_stage: str | None = None
    current_stage_index: int = 0
    total_stages: int = 0
    created_at: str
    started_at: str | None = None
    updated_at: str
    finished_at: str | None = None
    duration_sec: float | None = None
    pid: int | None = None
    mlflow: dict[str, Any] = Field(default_factory=dict)
    stages: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    error: Any = None


def worker_module_name() -> str:
    configured = os.getenv("MLSYSTEM_PIPELINE_WORKER_MODULE")
    if configured:
        return configured
    if Path("/opt/mlsystem/src").exists():
        return "src.pipeline_runner.worker"
    return "mlsystem.src.pipeline_runner.worker"


class PipelineRunner:
    def __init__(self, store: PipelineRunStore | None = None) -> None:
        self.store = store or PipelineRunStore()

    def start_run(self, config: PipelineRunConfig | dict[str, Any], *, source: str = "api") -> PipelineRun:
        parsed = config if isinstance(config, PipelineRunConfig) else PipelineRunConfig.model_validate(config)
        created = self.store.create_run(parsed)
        run_id = str(created["run_id"])
        self.store.update_run(run_id, source=source)
        bound_store = self.store.bind(run_id, parsed)
        worker_stdout_path = bound_store.log_dir / "worker_stdout.log"
        worker_stderr_path = bound_store.log_dir / "worker_stderr.log"
        cmd = [
            sys.executable,
            "-m",
            worker_module_name(),
            "--run-id",
            run_id,
            "--run-root",
            str(self.store.root),
        ]
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        env = os.environ.copy()
        env.setdefault("PYTHONFAULTHANDLER", "1")
        env.setdefault("PYTHONUNBUFFERED", "1")
        try:
            with worker_stdout_path.open("ab") as stdout_fp, worker_stderr_path.open("ab") as stderr_fp:
                process = subprocess.Popen(  # noqa: S603
                    cmd,
                    cwd=os.getenv("MLSYSTEM_API_WORKDIR") or None,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_fp,
                    stderr=stderr_fp,
                    creationflags=creationflags,
                    env=env,
                )
        except Exception as exc:  # noqa: BLE001
            failed = self.store.update_run(
                run_id,
                state="failed",
                finished_at=utc_now(),
                error={"type": type(exc).__name__, "message": mask_text(f"Failed to start pipeline worker: {exc}")},
            )
            return PipelineRun.model_validate(failed)
        return PipelineRun.model_validate(self.store.update_run(run_id, pid=process.pid))

    def refresh_run(self, run_id: str) -> PipelineRun:
        payload = self.store.read_run(run_id)
        if payload.get("state") in {"queued", "running"} and payload.get("pid") and not _pid_running(int(payload["pid"])):
            latest = self.store.read_run(run_id)
            if latest.get("state") in {"queued", "running"}:
                diagnostic = _worker_exit_diagnostic(self.store, run_id, latest)
                message = "Pipeline worker exited before writing a terminal state."
                self.store.append_log(run_id, f"\n[pipeline_runner] {message} diagnostic={diagnostic}\n")
                payload = self.store.update_run(
                    run_id,
                    state="failed",
                    finished_at=utc_now(),
                    error={"type": "WorkerExited", "message": message, **diagnostic},
                )
                try:
                    summary = self.store.bind(run_id).read_summary()
                    summary["worker_exit_diagnostic"] = diagnostic
                    summary["errors"] = (summary.get("errors") or []) + [{"type": "WorkerExited", "message": message}]
                    self.store.bind(run_id).update_summary(
                        worker_exit_diagnostic=diagnostic,
                        errors=summary["errors"],
                    )
                except Exception:
                    pass
            else:
                payload = latest
        return PipelineRun.model_validate(payload)

    def cancel_run(self, run_id: str) -> PipelineRun:
        self.store.mark_cancel_requested(run_id)
        payload = self.store.read_run(run_id)
        if payload.get("state") == "queued":
            payload = self.store.update_run(run_id, state="cancelled", finished_at=utc_now(), current_stage=None)
        return self.refresh_run(run_id)


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if _proc_state_for_pid(pid) == "Z":
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _proc_state_for_pid(pid: int) -> str | None:
    if os.name == "nt":
        return None
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("State:"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else None
    return None


def _worker_exit_diagnostic(store: PipelineRunStore, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    bound = store.bind(run_id)
    current_stage = str(payload.get("current_stage") or "")
    stage_log = bound.log_dir / f"{current_stage}.log" if current_stage else None
    worker_stderr = bound.log_dir / "worker_stderr.log"
    worker_stdout = bound.log_dir / "worker_stdout.log"
    pipeline_log = bound.log_dir / "pipeline.log"
    text_for_hints = "\n".join(
        tail
        for tail in (
            _tail_text(worker_stderr, 4000),
            _tail_text(stage_log, 4000) if stage_log else "",
            _tail_text(pipeline_log, 4000),
        )
        if tail
    )
    lower = text_for_hints.lower()
    oom_suspected = any(marker in lower for marker in ("out of memory", "oom", "killed"))
    return {
        "pid": payload.get("pid"),
        "last_stage": current_stage or None,
        "exit_code": None,
        "signal": None,
        "exit_code_available": False,
        "reason": "worker process is no longer running; exit code is unavailable because status was refreshed after the child process was detached",
        "oom_suspected_from_logs": oom_suspected,
        "worker_stderr_tail": mask_text(_tail_text(worker_stderr, 2000)),
        "worker_stdout_tail": mask_text(_tail_text(worker_stdout, 2000)),
        "stage_log_tail": mask_text(_tail_text(stage_log, 2000)) if stage_log else "",
    }


def _tail_text(path: Path | None, max_chars: int = 2000) -> str:
    if path is None or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError:
        return ""
