from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ._run_store import PseudolabelRunStore
from ._worker import run_worker
from .contracts import PseudolabelRun, PseudolabelRunRequest


ClientFactory = Callable[[], Any]


def worker_module_name() -> str:
    configured = os.getenv("MLSYSTEM_INFERENCE_PIPELINE_WORKER_MODULE")
    if configured:
        return configured
    if Path("/opt/mlsystem/src").exists():
        return "src.inference_pipeline._worker"
    return "mlsystem.src.inference_pipeline._worker"


class InferencePipelineRunner:
    def __init__(
        self,
        store: PseudolabelRunStore | None = None,
        *,
        client_factory: ClientFactory | None = None,
        run_worker_inline: bool | None = None,
    ) -> None:
        self.store = store or PseudolabelRunStore()
        self.client_factory = client_factory
        self.run_worker_inline = _env_inline_worker() if run_worker_inline is None else run_worker_inline

    def start_run(self, request: PseudolabelRunRequest | dict[str, Any]) -> PseudolabelRun:
        parsed = request if isinstance(request, PseudolabelRunRequest) else PseudolabelRunRequest.model_validate(request)
        run = self.store.create_run(parsed)
        if parsed.dry_run or self.run_worker_inline:
            run_worker(run.run_id, self.store.root, client_factory=self.client_factory)
            return self.store.read_run(run.run_id)
        cmd = [
            sys.executable,
            "-m",
            worker_module_name(),
            "--run-id",
            run.run_id,
            "--run-root",
            str(self.store.root),
        ]
        process = subprocess.Popen(cmd, cwd=os.getenv("MLSYSTEM_API_WORKDIR") or None)  # noqa: S603
        return self.store.update_run(run.run_id, pid=process.pid)

    def get_run(self, run_id: str) -> PseudolabelRun:
        return self.store.read_run(run_id)

    def cancel_run(self, run_id: str) -> PseudolabelRun:
        run = self.store.read_run(run_id)
        if run.pid and run.state in {"queued", "running"}:
            try:
                os.kill(run.pid, signal.SIGTERM)
            except OSError:
                pass
        self.store.append_log(run_id, "cancel requested")
        return self.store.cancel_run(run_id)

    def tail_log(self, run_id: str, max_chars: int = 20000) -> str:
        return self.store.tail_log(run_id, max_chars=max_chars)


def _env_inline_worker() -> bool:
    return str(os.getenv("MLSYSTEM_INFERENCE_PIPELINE_INLINE") or "").strip().lower() in {"1", "true", "yes", "on"}
