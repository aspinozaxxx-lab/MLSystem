# MLSystem frontend

Frontend - отдельный FastAPI BFF-сервис для веб-интерфейса MLSystem.

## Назначение

Сервис дает единый вход по логину/паролю и страницу проверки разметок.
Проверка разметок не запускает Airflow DAG и не содержит собственной
ML-логики. Frontend вызывает внутренний `mlsystem-api` тем же контрактом,
который использует Airflow:

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

Пароль и session secret должны переопределяться через Ansible/GitHub
secrets для production. API token хранится только в контейнере frontend и
не отдается браузеру.

## Проверка разметок

Пользователь загружает GeoJSON/JSON файл разметки и TXT файл со списком сцен.

Frontend сохраняет файлы вне git:

`/data/mlsystem/frontend/uploads/<run_id>/`

Затем загружает их в MinIO prefix:

`s3://mlsystems/frontend-checks/<run_id>/`

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

`/data/mlsystem/airflow/status/<run_id>/`

Используются `inventory_scenes.json`, `scene_matching_report.json`,
`matched_scenes.txt`, `missing_scenes.txt`, `dataset_manifest.json`,
`scene_object_counts.txt`, `split_summary.json`,
`dataset_validation_report.json` и stage JSON reports.

Frontend не пересчитывает GeoJSON, не ищет сцены в S3 и не делает train/val
split самостоятельно.

## Deploy

GitHub Actions workflow:

`.github/workflows/frontend.yml`

Ansible playbook:

`ansible/playbooks/deploy_frontend.yml`

Контейнер:

`mlsystem-gpu-frontend`

Health checks:

```bash
curl -fsS http://127.0.0.1:8090/health
curl -fsS -I http://127.0.0.1:8090/login
```

