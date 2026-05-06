# Проверка деплоя Airflow/API

Проверки выполняются после deploy через CI/CD. На сервере руками разрешена только диагностика.

## Containers

```bash
ssh gpu-mlserver "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

Ожидаемые контейнеры:

- `mlsystem-gpu-airflow-webserver`
- `mlsystem-gpu-airflow-scheduler`
- `mlsystem-gpu-airflow-triggerer`
- `mlsystem-gpu-api`
- `mlsystem-gpu-mlflow`
- `mlsystem-gpu-minio`
- `mlsystem-gpu-triton`
- postgres containers

## Health

```bash
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8081/api/v1/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:5000/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:9000/minio/health/live"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8000/v2/health/ready"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8088/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8088/ready"
```

Если API недоступен на host:

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler curl -fsS http://mlsystem-api:8088/health"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler curl -fsS http://mlsystem-api:8088/ready"
```

## Airflow

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags list-import-errors"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags list | grep mlsystem"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow tasks list mlsystem_experiment_pipeline"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow pools list"
```

Ожидается:

- import errors отсутствуют;
- `mlsystem_experiment_pipeline` и `mlsystem_smoke_pipeline` видны;
- task list совпадает с `MAIN_DAG_STAGES`;
- pools содержат `gpu_training`, `gpu_inference`, `cpu_heavy`, `cpu_light`, `io_light`.

## API stages

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler curl -fsS http://mlsystem-api:8088/api/v1/stages"
```

В ответе не должно быть старых stage aliases.

## Smoke and real run

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-webserver airflow dags trigger mlsystem_smoke_pipeline -r smoke_api_runtime_YYYYMMDD_HHMMSS"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags state mlsystem_smoke_pipeline smoke_api_runtime_YYYYMMDD_HHMMSS"
```

После smoke запускается маленький real deforest run, не `all_images`.

При failure:

1. Найти первый failed task.
2. Прочитать Airflow task log и `job_id`.
3. Прочитать `/data/mlsystem/api/jobs/<job_id>/job.json`.
4. Прочитать `error.json` и `stderr_tail.txt`.
5. Исправлять только через repo + CI/CD.
