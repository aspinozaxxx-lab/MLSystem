from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..api.security import mask_secrets, mask_text
from .stage_report_formatter import write_stage_report_file
from .config import config_with_run_id, dump_trace_yaml, parse_pipeline_run_config, safe_run_id
from .contracts import PipelineRunConfig

RUN_STATES = {"queued", "running", "succeeded", "failed", "cancelled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_root() -> Path:
    return Path(os.getenv("MLSYSTEM_RUN_ROOT", "/data/mlsystem/runs"))


class TrainPipelineRunStore:
    def __init__(self, run_root: str | Path | None = None, run_id: str | None = None, config: PipelineRunConfig | dict[str, Any] | None = None) -> None:
        self.root = Path(run_root) if run_root else default_run_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = safe_run_id(run_id) if run_id else None
        self.conf = _config_payload(config)
        self.experiment_id = str(self.conf.get("experiment_id") or self.run_id or "run")
        if self.run_id:
            self.run_dir = self.root / self.run_id
            self.log_dir = self.run_dir / "logs"
            self.stage_dir = self.run_dir / "stages"
            self.artifact_dir = self.run_dir / "artifacts"
            self.progress_dir = self.run_dir / "progress"
            for path in [self.log_dir, self.stage_dir, self.artifact_dir, self.progress_dir]:
                path.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = self.root
            self.log_dir = self.root / "logs"
            self.stage_dir = self.root / "stages"
            self.artifact_dir = self.root / "artifacts"
            self.progress_dir = self.root / "progress"

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def state_dir(self) -> Path:
        return self.root

    @property
    def pipeline_run_id(self) -> str:
        return str(self.run_id or self.experiment_id)

    def bind(self, run_id: str, config: PipelineRunConfig | dict[str, Any] | None = None) -> "TrainPipelineRunStore":
        return TrainPipelineRunStore(self.root, run_id, config)

    def create_run(self, config: PipelineRunConfig | dict[str, Any]) -> dict[str, Any]:
        parsed = parse_pipeline_run_config(config)
        run_id = parsed.run_id or _new_run_id(parsed.experiment_id)
        parsed = config_with_run_id(parsed, run_id)
        run_dir = self.root / run_id
        if run_dir.exists() and parsed.run_id:
            raise FileExistsError(f"Pipeline run already exists: {run_id}")
        bound = self.bind(run_id, parsed)
        for path in [bound.log_dir, bound.stage_dir, bound.artifact_dir, bound.progress_dir]:
            path.mkdir(parents=True, exist_ok=True)
        payload = _initial_run_payload(parsed)
        _write_json(bound.run_dir / "trace.json", parsed.model_dump(mode="json"))
        (bound.run_dir / "trace.yaml").write_text(dump_trace_yaml(parsed), encoding="utf-8")
        _write_json(bound.run_dir / "run.json", payload)
        _write_json(bound.run_dir / "status.json", payload)
        _write_json(bound.summary_path, {"schema_version": 1, "run_id": run_id, "experiment_id": parsed.experiment_id, "stages": {}, "warnings": [], "errors": []})
        return payload

    def read_config(self, run_id: str | None = None) -> PipelineRunConfig:
        bound = self._bound(run_id)
        return parse_pipeline_run_config(_read_json(bound.run_dir / "trace.json"))

    def read_run(self, run_id: str | None = None) -> dict[str, Any]:
        bound = self._bound(run_id)
        return _read_json(bound.run_dir / "status.json")

    def update_run(self, run_id: str | None = None, **updates: Any) -> dict[str, Any]:
        bound = self._bound(run_id)
        current = bound.read_run()
        current.update(updates)
        current["updated_at"] = utc_now()
        if current.get("finished_at") and current.get("started_at") and current.get("duration_sec") is None:
            current["duration_sec"] = _duration_sec(current.get("started_at"), current.get("finished_at"))
        _write_json(bound.run_dir / "status.json", current)
        _write_json(bound.run_dir / "run.json", current)
        return current

    def read_summary(self, run_id: str | None = None) -> dict[str, Any]:
        bound = self._bound(run_id) if run_id else self
        return _read_json(bound.summary_path, default={}) or {}

    def update_summary(self, **updates: Any) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "updated_at": utc_now(),
            "stages": {},
            "warnings": [],
            "errors": [],
            **self.read_summary(),
        }
        payload.update(updates)
        payload["updated_at"] = utc_now()
        _write_json(self.summary_path, payload)
        if "mlflow" in updates and self.run_id:
            try:
                self.update_run(self.run_id, mlflow=updates["mlflow"])
            except FileNotFoundError:
                pass
        return payload

    def append_log(self, run_id: str | None, text: str) -> None:
        bound = self._bound(run_id)
        masked = mask_text(text) or ""
        _append_text(bound.log_dir / "pipeline.log", masked)

    def write_stage(self, stage: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.run_id:
            raise ValueError("write_stage requires a bound run store")
        return self.write_stage_report(self.run_id, stage, payload)

    def write_stage_report(self, run_id: str | None, stage: str, report: dict[str, Any]) -> dict[str, Any]:
        bound = self._bound(run_id)
        stage_payload = {"stage": stage, "finished_at": utc_now(), **report}
        stage_json_path = bound.stage_dir / f"{stage}.json"
        report_path = bound.stage_dir / f"{stage}.report.md"
        stage_payload["stage_json_path"] = str(stage_json_path)
        stage_payload["report_path"] = str(report_path)
        _write_json(stage_json_path, stage_payload)
        try:
            write_stage_report_file(
                stage_payload,
                path=report_path,
                stage=stage,
                run_id=bound.run_id,
                stage_json_path=stage_json_path,
                container_status_root=bound.root,
                host_status_root=bound.root,
            )
        except Exception as exc:  # noqa: BLE001
            stage_payload["report_format_error"] = str(exc)
            _write_json(stage_json_path, stage_payload)
        bound._update_summary_for_stage(stage, stage_payload)
        bound._update_run_for_stage(stage, stage_payload)
        return mask_secrets(stage_payload)

    def tail_log(self, run_id: str | None = None, max_chars: int = 20000) -> str:
        bound = self._bound(run_id)
        path = bound.log_dir / "pipeline.log"
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]

    def cancel_requested(self, run_id: str | None = None) -> bool:
        bound = self._bound(run_id)
        return (bound.run_dir / "cancel.requested").exists()

    def mark_cancel_requested(self, run_id: str | None = None) -> None:
        bound = self._bound(run_id)
        (bound.run_dir / "cancel.requested").write_text(utc_now() + "\n", encoding="utf-8")

    def _bound(self, run_id: str | None = None) -> "TrainPipelineRunStore":
        effective = run_id or self.run_id
        if not effective:
            raise ValueError("run_id is required")
        if effective == self.run_id:
            return self
        return self.bind(effective)

    def _update_summary_for_stage(self, stage: str, stage_payload: dict[str, Any]) -> None:
        summary = self.read_summary()
        stages = summary.get("stages") or {}
        stages[stage] = {
            "status": stage_payload.get("status"),
            "finished_at": stage_payload.get("finished_at"),
            "duration_sec": stage_payload.get("duration_sec"),
            "summary": stage_payload.get("summary"),
            "report_path": stage_payload.get("report_path"),
            "stage_json_path": stage_payload.get("stage_json_path"),
        }
        summary["stages"] = stages
        summary["current_stage"] = stage
        summary["updated_at"] = utc_now()
        if stage_payload.get("warnings"):
            summary["warnings"] = sorted(set((summary.get("warnings") or []) + stage_payload["warnings"]))
        if stage_payload.get("error"):
            summary["errors"] = (summary.get("errors") or []) + [stage_payload["error"]]
        if stage_payload.get("mlflow") or stage_payload.get("details", {}).get("mlflow"):
            summary["mlflow"] = stage_payload.get("mlflow") or stage_payload.get("details", {}).get("mlflow")
        _write_json(self.summary_path, summary)

    def _update_run_for_stage(self, stage: str, stage_payload: dict[str, Any]) -> None:
        try:
            current = self.read_run()
        except FileNotFoundError:
            return
        stages = _stage_rows(current)
        row = _find_stage_row(stages, stage)
        row.update(
            {
                "name": stage,
                "state": _stage_state(stage_payload.get("status")),
                "status": stage_payload.get("status"),
                "duration_sec": stage_payload.get("duration_sec"),
                "finished_at": stage_payload.get("finished_at"),
                "summary": stage_payload.get("summary"),
                "report_path": stage_payload.get("report_path"),
                "stage_json_path": stage_payload.get("stage_json_path"),
            }
        )
        artifacts = dict(current.get("artifacts") or {})
        artifacts.update(stage_payload.get("artifacts") or {})
        mlflow = current.get("mlflow") or {}
        stage_mlflow = stage_payload.get("mlflow")
        if isinstance(stage_mlflow, dict):
            mlflow = {**mlflow, **stage_mlflow}
        details_mlflow = (stage_payload.get("details") or {}).get("mlflow")
        if isinstance(details_mlflow, dict):
            mlflow = {**mlflow, **details_mlflow}
        current.update({"stages": stages, "artifacts": artifacts, "mlflow": mlflow, "updated_at": utc_now()})
        _write_json(self.run_dir / "status.json", current)
        _write_json(self.run_dir / "run.json", current)


def _config_payload(config: PipelineRunConfig | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(config, PipelineRunConfig):
        return config.model_dump(mode="json")
    return dict(config or {})


def _new_run_id(experiment_id: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return safe_run_id(f"{experiment_id}_{stamp}_{uuid.uuid4().hex[:8]}")


def _initial_run_payload(config: PipelineRunConfig) -> dict[str, Any]:
    now = utc_now()
    stages = [{"name": stage, "state": "queued", "progress_percent": 0, "started_at": None, "finished_at": None} for stage in config.pipeline.stages]
    return {
        "run_id": config.run_id,
        "state": "queued",
        "progress_percent": 0,
        "current_stage": None,
        "current_stage_index": 0,
        "total_stages": len(config.pipeline.stages),
        "created_at": now,
        "started_at": None,
        "updated_at": now,
        "finished_at": None,
        "duration_sec": None,
        "pid": None,
        "mlflow": {},
        "stages": stages,
        "artifacts": {},
        "error": None,
    }


def _stage_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("stages")
    return list(rows) if isinstance(rows, list) else []


def _find_stage_row(stages: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    for row in stages:
        if row.get("name") == stage:
            return row
    row: dict[str, Any] = {"name": stage, "state": "queued"}
    stages.append(row)
    return row


def _stage_state(status: Any) -> str:
    if status in {"success", "success_with_warning", "skipped"}:
        return "succeeded"
    if status == "failed":
        return "failed"
    if status == "cancelled":
        return "cancelled"
    return str(status or "unknown")


def _duration_sec(started_at: Any, finished_at: Any) -> float | None:
    try:
        return round((datetime.fromisoformat(str(finished_at)) - datetime.fromisoformat(str(started_at))).total_seconds(), 3)
    except Exception:
        return None


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mask_secrets(payload), ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(text)
