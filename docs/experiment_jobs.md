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

The MVP executor only performs lightweight jobs: `noop`, `status_check`, and `inventory`.
Other task names validate but return `not_implemented`; this prevents GitHub Actions from launching heavy training.

Queue flow:

1. Put a YAML job in `jobs/pending/`.
2. `enqueue-jobs.yml` validates it and copies it to `/home/worker/mlsystem/storage/jobs/pending/`.
3. The server executor atomically claims the job by rename into `running/`.
4. It writes `claim.json`, `heartbeat.json`, `job.log`, and `result.json`.
5. Completed jobs move to `done/`; failures move to `failed/` with `error.json`.
6. Small summaries are written under `/home/worker/mlsystem/storage/experiments/<claim_id>/`.

Use `jobs/examples/deforest_cpu_smoke.yml` for safe end-to-end tests.
Use `jobs/examples/deforest_train_predict.yml` only as a production schema template until heavy train is explicitly implemented.
