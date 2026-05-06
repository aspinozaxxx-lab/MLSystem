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

Main references:

- [MLSystem API service](docs/mlsystem_api_service.md)
- [Airflow API execution](docs/airflow_api_execution.md)
- [Airflow stage inventory](docs/airflow_stage_inventory.md)
- [Airflow deploy validation](docs/airflow_deploy_validation.md)
