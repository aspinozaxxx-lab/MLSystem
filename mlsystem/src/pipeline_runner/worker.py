from __future__ import annotations

import argparse
import faulthandler
import json
import os
import traceback
from pathlib import Path
from typing import Any

from ..api.security import mask_text
from . import stages
from .config import PipelineRunConfig
from .logging import tee_stage_logs
from .mlflow_logging import log_pipeline_metadata
from .progress import progress_percent
from .run_store import PipelineRunStore, utc_now


def run_worker(run_id: str, run_root: str | Path | None = None) -> int:
    faulthandler.enable(all_threads=True)
    store = PipelineRunStore(run_root, run_id)
    config = store.read_config()
    started_at = utc_now()
    store.update_run(
        run_id,
        state="running",
        started_at=started_at,
        updated_at=started_at,
        pid=os.getpid(),
        total_stages=len(config.pipeline.stages),
        error=None,
    )
    completed: set[str] = set()
    failed = False
    try:
        for index, stage in enumerate(config.pipeline.stages):
            if store.cancel_requested(run_id):
                _mark_cancelled(store, run_id, completed, config)
                return 2
            _mark_stage_running(store, run_id, config, stage, index, completed)
            try:
                with tee_stage_logs(store.run_dir, stage):
                    print(f"[{utc_now()}] starting stage {stage}", flush=True)
                    if config.pipeline.dry_run:
                        report = _dry_run_report(stage)
                    else:
                        report = stages.run_stage(stage, config, store, run_id)
                    report = store.write_stage_report(run_id, stage, report)
                    print(f"[{utc_now()}] finished stage {stage}: {report.get('status')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed = True
                _mark_failed(store, run_id, stage, exc)
                if config.pipeline.stop_on_failure:
                    return 1
                continue
            status = str((report or {}).get("status") or "")
            if status == "failed":
                failed = True
                if config.pipeline.stop_on_failure:
                    _finish_failed_status(store, run_id, f"Stage failed: {stage}")
                    return 1
            else:
                completed.add(stage)
            store.update_run(
                run_id,
                progress_percent=progress_percent(config.pipeline.stages, completed),
                current_stage=stage,
                current_stage_index=index,
            )
        final_state = "failed" if failed else "succeeded"
        finished_at = utc_now()
        warnings = log_pipeline_metadata(config, store, run_id)
        final = store.update_run(
            run_id,
            state=final_state,
            progress_percent=100 if final_state == "succeeded" else progress_percent(config.pipeline.stages, completed),
            current_stage=None,
            current_stage_index=len(config.pipeline.stages),
            finished_at=finished_at,
            duration_sec=None,
            error=None if final_state == "succeeded" else "One or more stages failed.",
        )
        if warnings:
            summary = store.read_summary()
            summary["warnings"] = sorted(set((summary.get("warnings") or []) + warnings))
            store.update_summary(warnings=summary["warnings"])
        _write_final_summary(store, run_id, final, config)
        return 0 if final_state == "succeeded" else 1
    except Exception as exc:  # noqa: BLE001
        _finish_failed_status(store, run_id, f"{type(exc).__name__}: {exc}", traceback.format_exc())
        return 1


def _mark_stage_running(store: PipelineRunStore, run_id: str, config: PipelineRunConfig, stage: str, index: int, completed: set[str]) -> None:
    payload = store.read_run(run_id)
    rows = list(payload.get("stages") or [])
    for row in rows:
        if row.get("name") == stage:
            row.update({"state": "running", "started_at": utc_now(), "finished_at": None})
            break
    store.update_run(
        run_id,
        state="running",
        current_stage=stage,
        current_stage_index=index,
        total_stages=len(config.pipeline.stages),
        progress_percent=progress_percent(config.pipeline.stages, completed),
        stages=rows,
    )


def _mark_failed(store: PipelineRunStore, run_id: str, stage: str, exc: Exception) -> None:
    tb = traceback.format_exc()
    error = {
        "type": type(exc).__name__,
        "message": mask_text(str(exc)),
        "traceback_tail": mask_text("\n".join(tb.splitlines()[-40:])),
    }
    try:
        report = _read_stage_report(store, stage) or {"stage": stage, "status": "failed"}
        report["status"] = "failed"
        report["error"] = error
        report.setdefault("errors", [error["message"]])
        store.write_stage_report(run_id, stage, report)
    except Exception:
        pass
    _finish_failed_status(store, run_id, str(error["message"]), tb)


def _finish_failed_status(store: PipelineRunStore, run_id: str, message: str, tb: str | None = None) -> None:
    current = store.read_run(run_id)
    finished_at = utc_now()
    error: dict[str, Any] = {
        "message": mask_text(message),
    }
    if tb:
        error["traceback_tail"] = mask_text("\n".join(tb.splitlines()[-40:]))
    store.append_log(run_id, "\n" + str(error.get("message") or "") + "\n")
    store.update_run(
        run_id,
        state="failed",
        finished_at=finished_at,
        duration_sec=None,
        error=error,
        progress_percent=int(current.get("progress_percent") or 0),
    )


def _mark_cancelled(store: PipelineRunStore, run_id: str, completed: set[str], config: PipelineRunConfig) -> None:
    finished_at = utc_now()
    store.append_log(run_id, f"[{finished_at}] cancellation requested; stopping before next stage\n")
    store.update_run(
        run_id,
        state="cancelled",
        finished_at=finished_at,
        duration_sec=None,
        progress_percent=progress_percent(config.pipeline.stages, completed),
        current_stage=None,
    )


def _dry_run_report(stage: str) -> dict[str, Any]:
    return {
        "status": "success",
        "summary": "dry_run=true; stage was not executed",
        "counters": {"dry_run": True},
        "artifacts": {},
    }


def _read_stage_report(store: PipelineRunStore, stage: str) -> dict[str, Any] | None:
    path = store.stage_dir / f"{stage}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_final_summary(store: PipelineRunStore, run_id: str, final: dict[str, Any], config: PipelineRunConfig) -> None:
    summary = store.read_summary()
    summary.update(
        {
            "run_id": run_id,
            "experiment_id": config.experiment_id,
            "status": final.get("state"),
            "finished_at": final.get("finished_at"),
            "progress_percent": final.get("progress_percent"),
            "mlflow": final.get("mlflow") or summary.get("mlflow") or {},
            "artifacts": final.get("artifacts") or {},
            "updated_at": utc_now(),
        }
    )
    (store.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a persisted MLSystem pipeline.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, default=None)
    args = parser.parse_args()
    return run_worker(args.run_id, args.run_root)


if __name__ == "__main__":
    raise SystemExit(main())
