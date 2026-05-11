# MLSystem Frontend

Frontend is a separate FastAPI BFF service for the MLSystem web UI.

## URLs

Public entrypoint:

- `http://31.192.104.147/`
- `http://31.192.104.147/login`
- `http://31.192.104.147/health`

Internal frontend backend:

- `http://127.0.0.1:8090/health`

RabbitMQ Management UI:

- `FRONTEND_RABBITMQ_MANAGEMENT_URL` or `RABBITMQ_MANAGEMENT_PUBLIC_URL`
- production default: `http://31.192.104.147:15672/`

RabbitMQ credentials are not rendered in the page. The management UI requires RabbitMQ authentication; unauthenticated API calls to `/api/overview` must return `401`.

## Home Page

The home page includes a card titled `Очереди RabbitMQ`. It opens the native RabbitMQ Management UI and shows compact queue metrics fetched through the authenticated frontend session from:

```text
GET /api/inference-engine/queues
```

The frontend server calls InferenceEngine `/queues` internally using `INFERENCE_ENGINE_API_URL` and, if configured, `INFERENCE_ENGINE_API_TOKEN`.

## Annotation Check

The annotation check workflow uses `mlsystem-api` stage contracts and does not run ML logic in the frontend:

```text
POST /api/v1/runs/{run_id}/stages/inventory_scenes/start
GET  /api/v1/jobs/{job_id}
POST /api/v1/runs/{run_id}/stages/prepare_dataset/start
GET  /api/v1/jobs/{job_id}
```

Uploaded runtime data is outside git:

```text
/data/mlsystem/frontend/uploads/<run_id>/
```

## Environment

- `MLSYSTEM_FRONTEND_USER`
- `MLSYSTEM_FRONTEND_PASSWORD`
- `MLSYSTEM_FRONTEND_SESSION_SECRET`
- `MLSYSTEM_FRONTEND_SESSION_TTL_SECONDS`
- `MLSYSTEM_FRONTEND_COOKIE_SECURE`
- `MLSYSTEM_API_BASE_URL`
- `MLSYSTEM_API_TOKEN`
- `INFERENCE_ENGINE_API_URL`
- `INFERENCE_ENGINE_API_TOKEN`
- `FRONTEND_RABBITMQ_MANAGEMENT_URL`
- `RABBITMQ_MANAGEMENT_PUBLIC_URL`

Secrets stay in container/server env and are not sent to the browser.

## Deploy

Workflows:

```text
.github/workflows/frontend-site.yml
.github/workflows/frontend-ansible.yml
```

Playbook:

```text
ansible/playbooks/deploy_frontend.yml
```

Containers:

- `mlsystem-gpu-frontend`
- `mlsystem-gpu-frontend-proxy`

Validation:

```bash
curl -fsS http://127.0.0.1:8090/health
curl -fsS http://127.0.0.1/health
curl -fsSI http://31.192.104.147/login
```

Deploy and proxy setup are managed only through GitHub Actions and Ansible. Server-side manual edits to compose/env/container state are not part of the deployment process.
