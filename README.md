# MLSystem

MLSystem is the repository for the current NSPD ML pipeline.

Current execution path:

```text
Airflow -> mlsystem-api -> persisted API job -> stage registry -> production modules
```

Pseudolabel production path:

```text
Airflow -> mlsystem-api -> InferenceEngine API -> RabbitMQ stage workers -> Triton -> compatibility artifacts
```

Set `pseudolabel.source=inference_engine` to use the extracted InferenceEngine service. In that mode, `mlsystem` keeps training/MLflow semantics and downstream pseudolabel stages validate artifacts produced by InferenceEngine.
The production path is validated on the GPU server with RabbitMQ stage queues, Triton `segformer_b2`, a 2-scene mlsystem-api compatibility run, and a 20-scene InferenceEngine run; see [InferenceEngine validation report](docs/inference_engine_validation_report.md).

Runtime data, API jobs, Airflow stage status, model artifacts, probability maps, reports, caches and heavy geospatial files are not stored in git. They live on the server under paths such as:

```text
/data/mlsystem/api/jobs/
/data/mlsystem/airflow/status/
```

Deployment is managed through the repository, GitHub Actions and Ansible. Do not edit server compose files, env files, containers or secrets by hand.

GitHub Actions are split by responsibility:

- `mlservice` (`.github/workflows/mlservice.yml`) validates Python code, service contracts and deploys the MLSystem service code to the GPU server.
- `ansible` (`.github/workflows/ansible.yml`) validates and applies infrastructure changes: Ansible roles, compose templates, env generation, Airflow pools and platform services.
- `frontend-site` (`.github/workflows/frontend-site.yml`) validates frontend code and quickly updates/restarts only the web frontend service.
- `frontend-ansible` (`.github/workflows/frontend-ansible.yml`) applies frontend Ansible settings when `ansible/**` changes.
- `inference-engine-service` validates and deploys InferenceEngine service code.
- `inference-engine-infra` validates and applies RabbitMQ/InferenceEngine infrastructure.

The retired combined workflow must not be restored. Code/service rollout, infrastructure rollout, frontend code rollout and frontend settings rollout must stay separate.

Frontend lives in `frontend/` and is deployed with `ansible/playbooks/deploy_frontend.yml`.
It provides login and annotation checks. Annotation checks call `mlsystem-api`
stages `inventory_scenes` and `prepare_dataset`; they do not trigger Airflow DAGs.
Public frontend URL is `http://31.192.104.147/`; port `8090` is only the internal
frontend upstream behind the port 80 reverse proxy.

Frontend runtime data is not stored in git:

```text
/data/mlsystem/frontend/uploads/
```

Main references:

- [MLSystem API service](docs/mlsystem_api_service.md)
- [InferenceEngine](docs/inference_engine.md)
- [InferenceEngine inventory](docs/inference_engine_inventory.md)
- [Airflow API execution](docs/airflow_api_execution.md)
- [Airflow stage inventory](docs/airflow_stage_inventory.md)
- [Airflow deploy validation](docs/airflow_deploy_validation.md)
- [Frontend](docs/frontend.md)
