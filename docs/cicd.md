# CI/CD

## Workflows

`ci.yml`

- runs on GitHub-hosted runner
- checks YAML syntax
- checks Ansible playbook syntax
- compiles Python sources
- validates job examples and pending jobs with Pydantic
- scans for obvious committed secrets
- never deploys and never starts training

`deploy.yml`

- runs on self-hosted runner
- triggers on `workflow_dispatch`
- also triggers on push to `main` when `mlsystem/`, `ansible/`, or deploy workflow files change
- runs Ansible deploy locally on `roskadastr-ml`
- checks:
  - `systemctl is-active mlsystem-web.service`
  - `systemctl is-active mlsystem-executor.service`
  - `curl http://127.0.0.1:8010/api/state`
  - MLflow endpoint
  - MinIO health endpoint
- writes a small deploy summary in the workflow workspace and step summary

`enqueue-jobs.yml`

- runs on self-hosted runner
- validates `jobs/pending/*.yml`
- enqueues jobs through the deployed server CLI:

```bash
cd /home/worker/mlsystem
/home/worker/ml-training/.venv/bin/python -m src.cli enqueue <job.yml>
```

It does not configure services and does not run train/predict directly.

`sync-results.yml`

- runs on self-hosted runner
- reads only small `run_summary.json` and `codex_summary.json`
- writes normalized JSON files to `results/summaries/`
- does not copy MLflow artifacts, TIFF, checkpoints, probability maps, GeoJSON/GPKG, or caches

## Reproducibility boundary

Covered by Ansible:

- `/home/worker/mlsystem`
- configs under `/home/worker/mlsystem/configs`
- lightweight storage directories
- `/home/worker/mlsystem/logs`
- `/etc/mlsystem/mlsystem.env`
- `mlsystem-web.service`
- `mlsystem-executor.service`
- service restart and health checks

Manual exception:

- GitHub runner registration, because it requires a short-lived GitHub registration token.

Not managed here:

- Geoalert
- DVC
- MinIO/MLflow platform containers
- heavy training data and artifacts
