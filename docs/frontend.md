# MLSystem frontend

Frontend - отдельный FastAPI BFF-сервис для веб-интерфейса MLSystem.

## URL

Публичный вход:

- `http://31.192.104.147/`
- `http://31.192.104.147/login`
- `http://31.192.104.147/health`

Внутренний backend port frontend:

- `http://127.0.0.1:8090/health`

Пользователь не должен открывать `:8090` снаружи. Внешний HTTP идет через reverse proxy на порту 80.

## Назначение

Сервис дает единый вход по логину/паролю и страницу проверки разметок. Проверка разметок не запускает Airflow DAG и не содержит собственной ML-логики. Frontend вызывает внутренний `mlsystem-api` тем же stage-контрактом, который использует Airflow:

1. `POST /api/v1/runs/{run_id}/stages/inventory_scenes/start`
2. `GET /api/v1/jobs/{job_id}`
3. `POST /api/v1/runs/{run_id}/stages/prepare_dataset/start`
4. `GET /api/v1/jobs/{job_id}`

## Аутентификация

Настройки задаются через env:

- `MLSYSTEM_FRONTEND_USER`, default для dev/test: `mluser`
- `MLSYSTEM_FRONTEND_PASSWORD`, default для dev/test: `qazwsxedc`
- `MLSYSTEM_FRONTEND_SESSION_SECRET`
- `MLSYSTEM_FRONTEND_SESSION_TTL_SECONDS`
- `MLSYSTEM_FRONTEND_COOKIE_SECURE=false` для текущего HTTP-доступа

Пароль и session secret должны переопределяться через Ansible/GitHub secrets для production. API token хранится только в контейнере frontend и не отдается браузеру.

## Проверка разметок

Пользователь загружает только GeoJSON/JSON файл разметки и TXT файл со списком сцен или папок.

Frontend сохраняет файлы вне git:

```text
/data/mlsystem/frontend/uploads/<run_id>/
```

Затем загружает их в MinIO prefix:

```text
s3://mlsystems/frontend-checks/<run_id>/
```

После этого stages получают обычный production payload:

- `images_uri`
- `layout_uri=s3://mlsystems/frontend-checks/<run_id>/`
- `scenes_file`
- `annotation_file`
- `preprocess.split_strategy`
- optional `preprocess.annotation_crs`
- `preprocess.allow_inferred_annotation_crs`

## Отчеты

Frontend читает готовые artifacts stage run из:

```text
/data/mlsystem/airflow/status/<run_id>/
```

Используются:

- `inventory_scenes.json`
- `scene_matching_report.json`
- `matched_scenes.txt`
- `missing_scenes.txt`
- `dataset_manifest.json`
- `scene_object_counts.txt`
- `split_summary.json`
- `dataset_validation_report.json`
- stage JSON reports

Frontend не пересчитывает GeoJSON, не ищет сцены в S3 и не делает train/val split самостоятельно.

## Deploy

GitHub Actions workflows:

```text
.github/workflows/frontend-site.yml
.github/workflows/frontend-ansible.yml
```

`frontend-site` is the fast code rollout: it runs frontend tests, syncs `frontend/` and `docs/frontend.md`, then rebuilds/restarts only `mlsystem-frontend` and `mlsystem-frontend-proxy`.

`frontend-ansible` is the settings rollout: it runs the frontend Ansible playbook when `ansible/**` changes.

Ansible playbook:

```text
ansible/playbooks/deploy_frontend.yml
```

Контейнеры:

- `mlsystem-gpu-frontend`
- `mlsystem-gpu-frontend-proxy`

Reverse proxy:

- host port `0.0.0.0:80`
- upstream `mlsystem-frontend:8090` inside docker network

Health checks:

```bash
curl -fsS http://127.0.0.1:8090/health
curl -fsS http://127.0.0.1/health
curl -fsS http://31.192.104.147/health
curl -fsSI http://31.192.104.147/login
```

Deploy and proxy setup are managed only through GitHub Actions and Ansible. Server-side manual edits to compose/env/container state are not part of the deployment process.
