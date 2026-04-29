from __future__ import annotations
import shutil
import socket
import time
from pathlib import Path
from typing import Any
from .codex_summary import build_codex_summary
from .debug.smoke_train import run_smoke_train, write_epoch_table as smoke_write_epoch_table, write_history as smoke_write_history
from .io_utils import append_text, write_json
from .job_queue import claim_next, finish, heartbeat, utc_now
from .job_schema import JobSpec
from .mlflow_adapter import MLFLOW_NOTE_TAG, MLflowJobRun, build_run_note, compact_run_label, set_run_tags, start_job_run, trace_stage
from .orchestration.job_context import JobContext
from .pipeline.pipeline_factory import PipelineFactory
from .pipeline_config import PipelineConfig, ensure_storage_layout
from .preprocess_inventory import build_preprocess_inventory
from .reporting.artifact_reporter import (
    log_epoch_table_if_present,
    update_final_run_note,
    write_eval_threshold_table,
    write_pipeline_timeline,
)
from .resource_manager import collect_status

IMPLEMENTED_TASKS = {"noop", "status_check", "inventory", "train", "predict"}

def _log(path: Path, message: str) -> None:
    append_text(path, f"{utc_now()} {message}\n")

def _inventory(config: PipelineConfig) -> dict[str, Any]:
    roots = []
    for root in config.known_data_roots:
        item: dict[str, Any] = {"path": str(root), "exists": root.exists(), "files": 0, "size_bytes_sampled": 0, "by_extension": {}}
        if root.exists():
            count = 0
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                count += 1
                if count > config.max_inventory_files:
                    item["truncated"] = True
                    break
                suffix = path.suffix.lower() or "<none>"
                item["by_extension"][suffix] = item["by_extension"].get(suffix, 0) + 1
                try:
                    item["size_bytes_sampled"] += path.stat().st_size
                except OSError:
                    pass
            item["files"] = min(count, config.max_inventory_files)
        roots.append(item)
    s3_inventory = build_preprocess_inventory(config, dry_run=True)
    return {"inventory_roots": roots, "s3_inventory": s3_inventory, "max_inventory_files": config.max_inventory_files}

def _is_smoke_train(job: JobSpec) -> bool:
    return bool(job.params.get("smoke") or job.params.get("smoke_train") or job.train.get("smoke"))

def _is_debug_pseudolabel(job: JobSpec) -> bool:
    pseudolabel_cfg = job.predict.get("pseudolabel") or job.params.get("pseudolabel") or {}
    return bool(job.params.get("debug_pseudolabel") or pseudolabel_cfg.get("debug"))

def _write_history(experiment_dir: Path, history: list[dict[str, float]]) -> list[Path]:
    return smoke_write_history(experiment_dir, history)

def _write_epoch_table(experiment_dir: Path, history: list[dict[str, float]]) -> Path:
    return smoke_write_epoch_table(experiment_dir, history)

def _smoke_train(job: JobSpec, experiment_dir: Path, mlflow_run: MLflowJobRun, job_log: Path) -> dict[str, Any]:
    return run_smoke_train(job, experiment_dir, mlflow_run, job_log, _log)

def _execute(config: PipelineConfig, job: JobSpec, experiment_dir: Path, mlflow_run: MLflowJobRun, job_log: Path) -> dict[str, Any]:
    pipelines = PipelineFactory()
    if job.task == "noop":
        return {"status": "done", "message": "noop completed"}
    if job.task == "status_check":
        status = collect_status(config, include_services=True)
        summary = build_codex_summary(config, status)
        return {"status": "done", "resource_status_path": str(config.system_root / "resource_status.json"), "codex_summary_path": str(config.system_root / "codex_summary.json"), "queue": summary.get("queue")}
    if job.task == "inventory":
        return {"status": "done", **_inventory(config)}
    if job.task == "train" and _is_smoke_train(job):
        return _smoke_train(job, experiment_dir, mlflow_run, job_log)
    if job.task == "train":
        return pipelines.training().run(config, job, experiment_dir, mlflow_run, job_log, _log)
    if job.task == "predict" and _is_debug_pseudolabel(job):
        return pipelines.prediction().run_debug_pseudolabel(config, job, experiment_dir, mlflow_run, job_log, _log)
    return {"status": "not_implemented", "message": f"Task {job.task} is validated but not implemented in MVP executor"}

def _job_params(job: JobSpec) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "task": job.task,
        "class_name": job.class_name,
        "priority": job.priority,
        "description": job.description,
        "params": job.params,
        "data": job.data,
        "preprocess": job.preprocess,
        "train": job.train,
        "predict": job.predict,
        "postprocess": job.postprocess,
        "resources": job.resources.model_dump(),
    }

def run_once(config: PipelineConfig) -> dict[str, Any]:
    ensure_storage_layout(config)
    claimed = claim_next(config)
    if not claimed:
        status = collect_status(config, include_services=True)
        summary = build_codex_summary(config, status)
        return {"claimed": False, "message": "No pending jobs", "summary": summary}
    running_dir: Path = claimed["run_dir"]
    job: JobSpec = claimed["job"]
    claim_id = claimed["claim_id"]
    job_log = running_dir / "job.log"
    experiment_dir = config.experiments_root / claim_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    context = JobContext(
        config=config,
        job=job,
        claim_id=claim_id,
        running_dir=running_dir,
        experiment_dir=experiment_dir,
        job_log=job_log,
        queue_metadata=claimed.get("queue_metadata") or {},
    )
    shutil.copy2(running_dir / "job.yml", experiment_dir / "job.yml")
    _log(job_log, f"claimed job_id={job.job_id} task={job.task}")
    mlflow_result: dict[str, Any] = {}
    try:
        heartbeat(running_dir)
        started = time.time()
        experiment_name = job.mlflow.experiment or config.mlflow_default_experiment
        model_cfg = (job.params.get("model") if isinstance(job.params.get("model"), dict) else {}) or {}
        model_name = model_cfg.get("name") or job.train.get("model_name")
        tile_size = job.preprocess.get("tile_size")
        stride = job.preprocess.get("stride")
        run_name = job.mlflow.tags.get("run_label") or compact_run_label(job.job_id, model_name, tile_size)
        existing_run_id = ((context.queue_metadata or {}).get("mlflow") or {}).get("run_id")
        if existing_run_id:
            set_run_tags(
                config,
                existing_run_id,
                {
                    "job_status": "running",
                    "queue_state": "running",
                    "started_at": utc_now(),
                    "claim_id": claim_id,
                },
            )
        initial_note = build_run_note(
            job_id=job.job_id,
            run_label=run_name,
            model_name=model_name,
            tile_size=tile_size,
            stride=stride,
            data_uri=job.data.get("images_uri"),
            layout_uri=job.data.get("layout_uri"),
            warnings=["running"],
        )
        with start_job_run(
            config,
            experiment_name,
            run_name,
            params=_job_params(job),
            tags={
                "job_id": job.job_id,
                "task": job.task,
                "class_name": job.class_name or "",
                "job_status": "running",
                "queue_state": "running",
                "claim_id": claim_id,
                "run_label": run_name,
                "experiment_group": job.class_name or "",
                "model_name": model_name or "",
                "tile_size": tile_size or "",
                "stride": stride or "",
                "train_time_limit_sec": job.train.get("time_limit_sec") or "",
                "pseudolabel_enabled": str(bool((job.predict.get("pseudolabel") or {}).get("enabled"))).lower(),
                "postprocess_profile": "auto_tune" if job.postprocess.get("auto_tune") else "default",
                MLFLOW_NOTE_TAG: initial_note,
                **job.mlflow.tags,
            },
            run_id=existing_run_id,
        ) as mlflow_run:
            with trace_stage(
                "execute_job",
                {"job_id": job.job_id, "task": job.task, "model_name": model_name, "tile_size": tile_size},
            ):
                result = _execute(config, job, experiment_dir, mlflow_run, job_log)
            log_epoch_table_if_present(result=result, experiment_dir=experiment_dir, mlflow_run=mlflow_run, log_fn=_log, job_log=job_log)
            duration = round(time.time() - started, 3)
            mlflow_run.log_metrics({"duration_sec": duration})
            finished_at = utc_now()
            timeline_path = write_pipeline_timeline(
                experiment_dir=experiment_dir,
                job=job,
                claim_id=claim_id,
                queued_at=(context.queue_metadata or {}).get("created_at"),
                started_at=claimed["claim"].get("claimed_at"),
                finished_at=finished_at,
                duration_sec=duration,
                result=result,
            )
            eval_table_path = write_eval_threshold_table(experiment_dir, result)
            mlflow_run.log_artifacts([timeline_path, eval_table_path])
            mlflow_result = mlflow_run.result()
            run_summary_status = result.get("status", "done")
            mlflow_run.set_tags({"job_status": run_summary_status, "queue_state": run_summary_status})
            update_final_run_note(
                mlflow_run=mlflow_run,
                job=job,
                run_name=run_name,
                model_name=model_name,
                tile_size=tile_size,
                stride=stride,
                result=result,
            )
            resource_summary = mlflow_run.resource_summary()
            run_summary = {
                "schema_version": 1,
                "claim_id": claim_id,
                "job_id": job.job_id,
                "task": job.task,
                "status": run_summary_status,
                "implemented": job.task in IMPLEMENTED_TASKS,
                "started_at": claimed["claim"].get("claimed_at"),
                "finished_at": finished_at,
                "duration_sec": duration,
                "result": result,
                "mlflow": mlflow_result,
                "resource_summary": resource_summary,
                "local_experiment_dir": str(experiment_dir),
            }
            with trace_stage("build_run_summary", {"job_id": job.job_id, "status": run_summary_status, "duration_sec": duration}):
                write_json(experiment_dir / "run_summary.json", run_summary)
                write_json(running_dir / "result.json", {**result, "mlflow": mlflow_result})
                status = collect_status(config, include_services=True)
                codex = build_codex_summary(
                    config,
                    status,
                    current_mlflow=mlflow_result,
                    current_job={
                        "claim_id": claim_id,
                        "job_id": job.job_id,
                        "task": job.task,
                        "status": run_summary["status"],
                        "server": socket.gethostname(),
                        "metrics": result.get("last_epoch_metrics", {}),
                        "resource_summary": resource_summary,
                        "errors": [],
                    },
                )
                write_json(experiment_dir / "codex_summary.json", codex)
            mlflow_run.log_artifacts([experiment_dir / "run_summary.json", experiment_dir / "codex_summary.json", experiment_dir / "job.yml"])
        _log(job_log, f"completed status={run_summary['status']} mlflow_ok={mlflow_result.get('ok')}")
        final_dir = finish(config, running_dir, success=True)
        return {"claimed": True, "success": True, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "result": result, "mlflow": mlflow_result}
    except Exception as exc:
        error = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "job_id": job.job_id, "task": job.task, "failed_at": utc_now(), "mlflow": mlflow_result}
        existing_run_id = ((claimed.get("queue_metadata") or {}).get("mlflow") or {}).get("run_id")
        if existing_run_id:
            set_run_tags(config, existing_run_id, {"job_status": "failed", "queue_state": "failed", "error": error["error"]})
        write_json(running_dir / "error.json", error)
        _log(job_log, f"failed error={error['error']}")
        final_dir = finish(config, running_dir, success=False)
        collect_status(config, include_services=True)
        build_codex_summary(config)
        return {"claimed": True, "success": False, "claim_id": claim_id, "job_id": job.job_id, "task": job.task, "final_dir": str(final_dir), "error": error}
