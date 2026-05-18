from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .contracts import PseudolabelRun, PseudolabelRunRequest


RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
TERMINAL_STATES = {"succeeded", "failed", "cancelled", "timed_out"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_root() -> Path:
    return Path(os.getenv("MLSYSTEM_PSEUDOLABEL_RUN_ROOT") or "/data/mlsystem/pseudolabel-runs")


def safe_run_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:180] or "pseudolabel"


def validate_run_id(value: str) -> str:
    if not value or not RUN_ID_RE.match(value):
        raise ValueError("run_id may contain only letters, digits, dot, underscore and dash")
    return value


class PseudolabelRunStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_run_root()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, request: PseudolabelRunRequest) -> PseudolabelRun:
        run_id = validate_run_id(request.run_id) if request.run_id else safe_run_id(f"{request.experiment_id}-{utc_now()}-{uuid4().hex[:8]}")
        run_dir = self.run_dir(run_id)
        if run_dir.exists():
            raise FileExistsError(f"Pseudolabel run already exists: {run_id}")
        run_dir.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        run = PseudolabelRun(
            run_id=run_id,
            experiment_id=request.experiment_id,
            state="queued",
            progress_percent=0,
            created_at=now,
            updated_at=now,
        )
        self.write_json(run_dir / "request.json", request.model_dump(mode="json"))
        self.write_json(run_dir / "trace.json", request.model_dump(mode="json"))
        self.write_run(run)
        self.append_log(run_id, f"created pseudolabel run experiment_id={request.experiment_id}")
        return run

    def run_dir(self, run_id: str) -> Path:
        return self.root / validate_run_id(run_id)

    def status_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "status.json"

    def summary_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "summary.json"

    def log_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "logs" / "pseudolabel.log"

    def read_request(self, run_id: str) -> PseudolabelRunRequest:
        return PseudolabelRunRequest.model_validate(self.read_json(self.run_dir(run_id) / "request.json"))

    def read_run(self, run_id: str) -> PseudolabelRun:
        return PseudolabelRun.model_validate(self.read_json(self.status_path(run_id)))

    def write_run(self, run: PseudolabelRun) -> None:
        self.write_json(self.status_path(run.run_id), run.model_dump(mode="json"))

    def update_run(self, run_id: str, **updates: Any) -> PseudolabelRun:
        current = self.read_run(run_id)
        payload = current.model_dump(mode="json")
        payload.update(updates)
        payload["updated_at"] = utc_now()
        updated = PseudolabelRun.model_validate(payload)
        self.write_run(updated)
        return updated

    def cancel_run(self, run_id: str) -> PseudolabelRun:
        run = self.read_run(run_id)
        if run.state in TERMINAL_STATES:
            return run
        return self.update_run(run_id, state="cancelled", progress_percent=100, finished_at=utc_now())

    def write_summary(self, run_id: str, payload: dict[str, Any]) -> None:
        self.write_json(self.summary_path(run_id), payload)

    def append_log(self, run_id: str, message: str) -> None:
        path = self.log_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(f"{utc_now()} {message}\n")

    def tail_log(self, run_id: str, max_chars: int = 20000) -> str:
        path = self.log_path(run_id)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max(1, int(max_chars)) :]

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
