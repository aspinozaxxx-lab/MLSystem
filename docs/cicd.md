# CI/CD

The repository uses four GitHub Actions workflows. Each workflow owns one change domain.

## cicd-ansible.yml

Purpose: infrastructure, MLflow platform deploy, systemd units, non-secret runtime env, storage/log directories.

Triggers:

- `workflow_dispatch`
- push to `main` for:
  - `ansible/**`
  - `.github/workflows/cicd-ansible.yml`

Runs on the self-hosted runner. It parses Ansible YAML, checks inventory, runs playbook syntax checks, applies `ansible/playbooks/deploy_mlflow.yml`, applies `ansible/playbooks/deploy_infra.yml`, and verifies:

- `mlsystem-web.service`
- `mlsystem-executor.service`
- `mlsystem-preprocess.service`
- `http://127.0.0.1:8010/api/state`
- `http://127.0.0.1:8010/api/preprocess`
- `http://127.0.0.1:5000`
- `http://127.0.0.1:5000/version` equals `3.11.1`
- `http://127.0.0.1:9000/minio/health/live`

## cicd-code.yml

Purpose: deploy application code under `mlsystem/`.

Triggers:

- `workflow_dispatch`
- push to `main` for:
  - `mlsystem/**`
  - `configs/**`
  - `.github/workflows/cicd-code.yml`

Runs on the self-hosted runner. It compiles Python, validates configs, scans for obvious secrets, syntax-checks `ansible/playbooks/deploy_code.yml`, deploys code through Ansible, restarts the MLSystem services, and verifies the web API. If `mlsystem-preprocess.service` already exists, code deploy also checks that it is active.

## cicd-queue.yml

Purpose: enqueue experiment jobs.

Triggers:

- `workflow_dispatch`
- push to `main` for:
  - `jobs/pending/**/*.yml`
  - `jobs/pending/**/*.yaml`
  - `.github/workflows/cicd-queue.yml`

Runs on the self-hosted runner. On push, it validates and enqueues only changed pending job YAML files. On manual `workflow_dispatch`, it processes all pending job YAML files. Enqueue goes through the deployed server CLI:

```bash
cd /home/worker/mlsystem
/home/worker/ml-training/.venv/bin/python -m src.cli enqueue <job.yml>
```

It does not deploy code, modify infrastructure, or start training directly.

## sync-results.yml

Purpose: sync lightweight result summaries back to Git.

Triggers:

- `workflow_dispatch`
- schedule

It reads only small `run_summary.json` and `codex_summary.json` files from the server, writes normalized JSON files to `results/summaries/`, and commits with `[skip ci]`.

No workflow triggers on `results/**`, so result sync commits do not start code, infrastructure, or queue workflows.

## Ansible playbooks

- `bootstrap.yml`: first-time server checks and directory bootstrap.
- `deploy_infra.yml`: infrastructure and services only.
- `deploy_code.yml`: application code only.
- `deploy_mlflow.yml`: MLflow Docker image/compose update with a lightweight PostgreSQL metadata backup.
- `deploy.yml`: manual wrapper for full deploy.
- `runner.yml`: audit-only runner status check.

The GitHub runner registration is the only manual exception because GitHub uses a short-lived registration token. Do not store that token in Git, docs, configs, logs, or Ansible vars.
