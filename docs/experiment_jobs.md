# Experiment jobs

Jobs are YAML files validated by `mlsystem/src/job_schema.py`.

Supported task names:

- `noop`
- `status_check`
- `inventory`
- `prepare`
- `train`
- `predict`
- `postprocess`
- `train_predict_pseudolabel`

The executor performs lightweight jobs and the current CPU-safe real `train` path. Heavy experiment series must still be submitted through the queue and resource limits, not by manual server commands.

Queue flow:

1. Put a YAML job in `jobs/pending/`.
2. `cicd-queue.yml` validates it and enqueues it through `/home/worker/mlsystem` CLI.
3. Enqueue creates an MLflow run with a compact run name such as `E04-r34-t512` or `SMK-r18-t512`, plus tags `job_status=queued`, `queue_state=pending`, `job_id`, `model_name`, `tile_size`, and `run_label`.
4. The server executor atomically claims the job by rename into `running/`.
5. The executor resumes the same MLflow run via `mlflow.start_run(run_id=...)`, updates tags to `running`, logs metrics/artifacts, and finishes with `done` or `failed`.
6. It writes `claim.json`, heartbeat data, `job.log`, and `result.json`.
7. Completed jobs move to `done/`; failures move to `failed/` with `error.json`.
8. Small summaries are written under `/home/worker/mlsystem/storage/experiments/<claim_id>/`.

MLflow runs include:

- run note via `mlflow.note.content`;
- epoch scalar metrics with `step=epoch`;
- `history.csv`, `history.json`, and `epoch_metrics_table.json`;
- `pipeline_timeline.json`;
- `eval_threshold_table.json`;
- `run_summary.json` and `codex_summary.json`;
- pseudolabel artifacts when enabled.

Queued/running jobs are visible in MLflow before the executor starts them. The web API exposes queue MLflow links at:

```text
http://127.0.0.1:8010/api/mlflow-queue
```

Use `jobs/examples/deforest_cpu_smoke.yml` for safe end-to-end tests.
Use `jobs/examples/deforest_train_predict.yml` only as a production schema template until heavy train is explicitly implemented.
