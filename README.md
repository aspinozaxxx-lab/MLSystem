# MLSystem

MLSystem is the repository for the current NSPD ML pipeline.

Current execution path:

```text
Airflow -> mlsystem-api -> persisted API job -> stage registry -> production modules
```

Runtime data, API jobs, Airflow stage status, model artifacts, probability maps, reports, caches and heavy geospatial files are not stored in git. They live on the server under paths such as:

```text
/data/mlsystem/api/jobs/
/data/mlsystem/airflow/status/
```

Deployment is managed through the repository, GitHub Actions and Ansible. Do not edit server compose files, env files, containers or secrets by hand.

GitHub Actions are split by responsibility:

- `mlservice` (`.github/workflows/mlservice.yml`) validates Python code, service contracts and deploys the MLSystem service code to the GPU server.
- `ansible` (`.github/workflows/ansible.yml`) validates and applies infrastructure changes: Ansible roles, compose templates, env generation, Airflow pools and platform services.

The retired combined workflow must not be restored. Code/service rollout and infrastructure rollout must stay separate.

Main references:

- [MLSystem API service](docs/mlsystem_api_service.md)
- [Airflow API execution](docs/airflow_api_execution.md)
- [Airflow stage inventory](docs/airflow_stage_inventory.md)
- [Airflow deploy validation](docs/airflow_deploy_validation.md)
