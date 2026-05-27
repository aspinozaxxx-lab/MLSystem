# MLSystem Frontend

Frontend is a FastAPI BFF and authenticated admin gateway for MLSystem.

## Public URLs

Public entrypoint:

- `http://31.192.104.147/`
- `http://31.192.104.147/login`
- `http://31.192.104.147/health`

Admin UI routes are exposed through the same frontend session, except MinIO Console which uses MinIO's own login:

- `http://31.192.104.147/mlflow/`
- `http://31.192.104.147/rabbitmq/`
- `http://31.192.104.147/minio/browser/mlsystems/images/`
- `http://31.192.104.147/grafana/`
- `http://31.192.104.147/prometheus/`

The raw admin UI ports are not the primary public entrypoints. MLflow, RabbitMQ Management, MinIO Console, Grafana, and Prometheus bind to localhost/internal addresses where possible and are reached through the frontend reverse proxy.

## Gateway Auth

The nginx frontend proxy protects admin UI locations with:

```text
auth_request /auth/proxy-check
```

Frontend endpoint:

```text
GET /auth/proxy-check
```

Behavior:

- Valid frontend session: `204 No Content`
- Missing/expired session: `401 Unauthorized`
- Response headers for valid sessions:
  - `X-MLSystem-User`
  - `X-Remote-User`

The proxy forwards user headers to upstream services:

- `X-Forwarded-For`
- `X-Forwarded-Proto`
- `X-Forwarded-Host`
- `X-Forwarded-Prefix`
- `X-Real-IP`
- `X-Remote-User`
- `X-MLSystem-User`

MLflow has its own security middleware disabled, so it must stay behind frontend auth and should not be exposed directly.

RabbitMQ Management is served under `/rabbitmq/`. Nginx injects the RabbitMQ Basic Authorization header server-side from `/etc/mlsystem/gpu-platform.env`; credentials are not rendered into HTML, JavaScript, docs, or URLs.

Grafana is served under `/grafana/` with auth-proxy mode. Nginx passes `X-MLSystem-User` after validating the frontend session, so the user does not see a second Grafana login. Prometheus is available under `/prometheus/` and is protected by the same frontend session.

## MinIO

Native MinIO Console SSO is not enabled because the current frontend login is a simple session and not an OIDC/LDAP identity provider.
MinIO Console is exposed under the main domain without frontend session auth:

```text
GET /minio/
```

The home page opens the Console directly on:

```text
/minio/browser/mlsystems/images/
```

Users sign in to MinIO Console with a dedicated read-only MinIO user. The policy allows only `GetBucketLocation`, `ListBucket` for `images/`, and `GetObject` for `images/*` in the `mlsystems` bucket. It does not grant write/delete permissions and does not grant access to `mlflow-artifacts`, `models`, `layouts`, `reports`, `pseudolabels`, or other prefixes. MinIO root credentials are never rendered into HTML, JavaScript, docs, or URLs.

The `kanopus-reader` MinIO user has a separate images policy: read access to `mlsystems/images/kanopus/*` and read/write/delete access only to `mlsystems/images/incoming/*`.

## Home Page

The home page includes active cards:

- `MLflow`
- `MinIO Console`
- `Очереди RabbitMQ`
- `Grafana`
- `Prometheus`
- `Проверка разметок`
- `Документация`

The top monitoring section embeds:

```text
/grafana/d/mlsystem-overview/mlsystem-overview?orgId=1&kiosk
```

Compact service status comes from:

```text
GET /api/services/status
```

RabbitMQ queue metrics come from:

```text
GET /api/inference-engine/queues
```

The frontend backend calls internal services and InferenceEngine; browsers do not call internal Docker DNS names directly.

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
- `FRONTEND_MLFLOW_UI_URL=/mlflow/`
- `FRONTEND_MINIO_UI_URL=/minio/browser/mlsystems/images/`
- `FRONTEND_RABBITMQ_MANAGEMENT_URL=/rabbitmq/`
- `FRONTEND_GRAFANA_URL=/grafana/`
- `FRONTEND_PROMETHEUS_URL=/prometheus/`
- `FRONTEND_GRAFANA_MAIN_DASHBOARD_URL=/grafana/d/mlsystem-overview/mlsystem-overview?orgId=1&kiosk`
- `RABBITMQ_MANAGEMENT_PROXY_AUTH`

Secrets stay in server/container env and are not sent to the browser.

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
curl -fsSI http://127.0.0.1/login
```

Deploy and proxy setup are managed only through GitHub Actions and Ansible. Server-side manual edits to compose/env/container state are not part of the deployment process.
