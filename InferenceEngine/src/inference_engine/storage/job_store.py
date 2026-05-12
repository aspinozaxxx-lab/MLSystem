from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .local_io import read_json, write_json


TERMINAL_STATUSES = {"success", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_job_id(experiment_id: str) -> str:
    prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in experiment_id)[:80] or "job"
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class JobStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "state.json"

    def request_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "request.json"

    def events_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "events.jsonl"

    def create(self, request: dict[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
        job_id = job_id or new_job_id(str(request.get("experiment_id") or "job"))
        job_dir = self.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        created_at = utc_now()
        state = {
            "schema_version": 1,
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "updated_at": created_at,
            "experiment_id": request.get("experiment_id"),
            "run_id": request.get("run_id"),
            "metrics": {},
            "counters": {},
            "artifacts": {},
            "error": None,
        }
        write_json(self.request_path(job_id), request)
        write_json(self.state_path(job_id), state)
        self.add_event(job_id, "job.queued", {"experiment_id": request.get("experiment_id")})
        return state

    def read(self, job_id: str) -> dict[str, Any]:
        state = read_json(self.state_path(job_id), default=None)
        if state is None:
            raise KeyError(f"Unknown InferenceEngine job: {job_id}")
        return state

    def read_request(self, job_id: str) -> dict[str, Any]:
        payload = read_json(self.request_path(job_id), default=None)
        if payload is None:
            raise KeyError(f"Unknown InferenceEngine job request: {job_id}")
        return payload

    def update(self, job_id: str, **updates: Any) -> dict[str, Any]:
        state = self.read(job_id)
        for key, value in updates.items():
            if key in {"metrics", "counters", "artifacts"}:
                merged = dict(state.get(key) or {})
                merged.update(value or {})
                state[key] = merged
            else:
                state[key] = value
        state["updated_at"] = utc_now()
        write_json(self.state_path(job_id), state)
        return state

    def replace_artifacts(self, job_id: str, artifacts: dict[str, str]) -> dict[str, Any]:
        state = self.read(job_id)
        state["artifacts"] = dict(artifacts or {})
        state["updated_at"] = utc_now()
        write_json(self.state_path(job_id), state)
        return state

    def add_event(self, job_id: str, event_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "ts": utc_now(),
            "ts_unix": time.time(),
            "job_id": job_id,
            "type": event_type,
            "payload": payload or {},
        }
        path = self.events_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def events(self, job_id: str) -> list[dict[str, Any]]:
        path = self.events_path(job_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def artifacts(self, job_id: str) -> dict[str, str]:
        return dict(self.read(job_id).get("artifacts") or {})

    def cancel(self, job_id: str) -> dict[str, Any]:
        state = self.read(job_id)
        if state.get("status") not in TERMINAL_STATUSES:
            state = self.update(job_id, status="cancelled")
            self.add_event(job_id, "job.cancelled", {})
        return state
