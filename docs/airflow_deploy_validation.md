# Проверка деплоя Airflow, API и frontend

Деплой выполняется только через GitHub Actions и Ansible. На сервере руками разрешена только диагностика: `docker ps`, `docker logs`, `curl`, Airflow CLI и чтение status/job файлов. Compose, env, контейнеры и секреты вручную не правятся.

## GitHub Actions

В репозитории используются три актуальных workflow:

| Workflow | Файл | Назначение |
| --- | --- | --- |
| `mlservice` | `.github/workflows/mlservice.yml` | Тесты, compile checks, service source bundle, деплой кода `mlsystem-api`, Airflow DAG и runtime modules. |
| `ansible` | `.github/workflows/ansible.yml` | Ansible syntax/check/apply для compose, env, platform, pools и инфраструктуры. |
| `frontend` | `.github/workflows/frontend.yml` | Тесты, сборка и деплой веб-интерфейса `mlsystem-gpu-frontend`. |

Кодовый rollout, frontend rollout и infrastructure rollout не должны объединяться в один workflow.

## Containers

```bash
ssh gpu-mlserver "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

Ожидаемые контейнеры:

- `mlsystem-gpu-airflow-webserver`
- `mlsystem-gpu-airflow-scheduler`
- `mlsystem-gpu-airflow-triggerer`
- `mlsystem-gpu-api`
- `mlsystem-gpu-frontend`
- `mlsystem-gpu-mlflow`
- `mlsystem-gpu-minio`
- `mlsystem-gpu-triton`
- postgres containers

Лишние контейнеры старой очереди не должны появляться.

## Health

```bash
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8081/api/v1/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:5000/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:9000/minio/health/live"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8000/v2/health/ready"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8088/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8088/ready"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8090/health"
ssh gpu-mlserver "curl -fsS -I http://127.0.0.1:8090/login"
```

`/health` у `mlsystem-api` должен показывать git commit текущего service deploy. Значение `unknown` означает, что `mlservice` не прокинул `MLSYSTEM_COMMIT` или контейнер не перезапущен.

Если API или frontend недоступны на host, проверка выполняется из контейнерной сети:

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler curl -fsS http://mlsystem-api:8088/ready"
ssh gpu-mlserver "docker exec mlsystem-gpu-frontend python - <<'PY'
import os, urllib.request
base = os.environ.get('MLSYSTEM_API_BASE_URL', 'http://mlsystem-api:8088')
print(urllib.request.urlopen(base + '/ready', timeout=5).status)
PY"
```

## Airflow

```bash
ssh gpu-mlserver "docker compose --env-file /etc/mlsystem/gpu-platform.env -f /data/mlsystem/platform/docker-compose.yml exec -T airflow-scheduler airflow dags list-import-errors"
ssh gpu-mlserver "docker compose --env-file /etc/mlsystem/gpu-platform.env -f /data/mlsystem/platform/docker-compose.yml exec -T airflow-scheduler airflow dags list | grep mlsystem"
ssh gpu-mlserver "docker compose --env-file /etc/mlsystem/gpu-platform.env -f /data/mlsystem/platform/docker-compose.yml exec -T airflow-scheduler airflow tasks list mlsystem_experiment_pipeline"
ssh gpu-mlserver "docker compose --env-file /etc/mlsystem/gpu-platform.env -f /data/mlsystem/platform/docker-compose.yml exec -T airflow-scheduler airflow pools list"
```

Ожидается:

- import errors отсутствуют;
- `mlsystem_experiment_pipeline` и `mlsystem_smoke_pipeline` видны;
- task list совпадает с `MAIN_DAG_STAGES`;
- pools содержат `gpu_training`, `gpu_inference`, `cpu_heavy`, `cpu_light`, `io_light`.

## API stages

```bash
ssh gpu-mlserver "docker compose --env-file /etc/mlsystem/gpu-platform.env -f /data/mlsystem/platform/docker-compose.yml exec -T airflow-scheduler curl -fsS http://mlsystem-api:8088/api/v1/stages"
```

В ответе:

- `aliases` должен быть `{}`;
- stage names должны совпадать с текущим `MAIN_DAG_STAGES`;
- runtime artifacts должны ссылаться на `/data/mlsystem/...`, а не на repo.

## Frontend

Frontend проверяется отдельно от Airflow:

```bash
ssh gpu-mlserver "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep mlsystem-gpu-frontend"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8090/health"
ssh gpu-mlserver "curl -fsS -I http://127.0.0.1:8090/login"
ssh gpu-mlserver "docker exec mlsystem-gpu-frontend python - <<'PY'
import os, urllib.request
base = os.environ.get('MLSYSTEM_API_BASE_URL', 'http://mlsystem-api:8088')
print(urllib.request.urlopen(base + '/ready', timeout=5).status)
PY"
```

Страница проверки разметок запускает только `mlsystem-api` stages `inventory_scenes` и `prepare_dataset`. Airflow DAG для этой проверки не запускается.

Uploads находятся вне git:

- `/data/mlsystem/frontend/uploads/<run_id>/`

Stage artifacts остаются в общем runtime status root:

- `/data/mlsystem/airflow/status/<run_id>/`

## Smoke and real run

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-webserver airflow dags trigger mlsystem_smoke_pipeline -r smoke_api_runtime_YYYYMMDD_HHMMSS"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags list-runs -d mlsystem_smoke_pipeline | head -20"
```

После smoke запускается маленький production-like run, не `all_images`.

При failure:

1. Найти первый failed task.
2. Прочитать Airflow task log и `job_id`.
3. Прочитать `/data/mlsystem/api/jobs/<job_id>/job.json`.
4. Прочитать `error.json` и `stderr_tail.txt`.
5. Исправлять только через repo, GitHub Actions и Ansible.
