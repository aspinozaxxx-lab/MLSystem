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

The MVP executor only performs lightweight jobs: `noop`, `status_check`, `inventory`, and synthetic smoke `train` jobs.
Other task names validate but return `not_implemented`; this prevents GitHub Actions from launching heavy training.

Queue flow:

1. Put a YAML job in `jobs/pending/`.
2. `cicd-queue.yml` validates it and enqueues it through `/home/worker/mlsystem` CLI.
3. Enqueue creates an MLflow run with `run name = queued:<job_id>` and tags `job_status=queued`, `queue_state=pending`.
4. The server executor atomically claims the job by rename into `running/`.
5. The executor resumes the same MLflow run via `mlflow.start_run(run_id=...)`, updates tags to `running`, logs metrics/artifacts, and finishes with `done` or `failed`.
6. It writes `claim.json`, heartbeat data, `job.log`, and `result.json`.
7. Completed jobs move to `done/`; failures move to `failed/` with `error.json`.
8. Small summaries are written under `/home/worker/mlsystem/storage/experiments/<claim_id>/`.

Queued/running jobs are visible in MLflow before the executor starts them. The web API exposes queue MLflow links at:

```text
http://127.0.0.1:8010/api/mlflow-queue
```

Use `jobs/examples/deforest_cpu_smoke.yml` for safe end-to-end tests.
Use `jobs/examples/deforest_train_predict.yml` only as a production schema template until heavy train is explicitly implemented.
