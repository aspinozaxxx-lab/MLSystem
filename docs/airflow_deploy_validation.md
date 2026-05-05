# Airflow deploy validation

Проверки выполняются после deploy через CI/CD. На сервере руками разрешена только диагностика: `docker ps`, `docker logs`, `curl`, `airflow CLI`, `find`, `cat status files`.

## Контейнеры

```bash
ssh gpu-mlserver "docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'"
```

Ожидаются:

- `mlsystem-gpu-airflow-webserver`
- `mlsystem-gpu-airflow-scheduler`
- `mlsystem-gpu-airflow-triggerer`
- `mlsystem-gpu-mlflow`
- `mlsystem-gpu-minio`
- `mlsystem-gpu-triton`
- `mlsystem-gpu-rabbitmq`
- `mlsystem-gpu-api`
- postgres containers

## Health checks

```bash
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8081/api/v1/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:5000/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:9000/minio/health/live"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8000/v2/health/ready"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8088/health"
ssh gpu-mlserver "curl -fsS http://127.0.0.1:8088/ready"
```

Если API не опубликован на host:

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler curl -fsS http://mlsystem-api:8088/health"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler curl -fsS http://mlsystem-api:8088/ready"
```

## Airflow import

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags list-import-errors"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags list | grep mlsystem"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow tasks list mlsystem_experiment_pipeline"
```

Ожидаемо:

- import errors отсутствуют;
- `mlsystem_experiment_pipeline` виден;
- `mlsystem_smoke_pipeline` виден;
- task list содержит актуальный `MAIN_DAG_STAGES`.

## API stages

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler curl -fsS http://mlsystem-api:8088/api/v1/stages"
```

Проверить наличие:

- `inventory_scenes`
- `prepare_dataset`
- `create_mlflow_run`
- `train_model`
- `prepare_inference_scenes`
- `run_pseudolabel_inference`
- `validate_probability_maps`
- `finalize_mlflow_run`

## Smoke DAG

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-webserver airflow dags trigger mlsystem_smoke_pipeline -r smoke_api_runtime_YYYYMMDD_HHMMSS"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags state mlsystem_smoke_pipeline smoke_api_runtime_YYYYMMDD_HHMMSS"
```

При failure:

1. найти первый failed task;
2. открыть Airflow task log и `job_id`;
3. прочитать `/data/mlsystem/api/jobs/<job_id>/job.json`;
4. прочитать `error.json` и `stderr_tail.txt`;
5. исправлять только через repo + CI/CD.

## Маленький real deforest run

Запускать только после успешного smoke. Не запускать all-images.

Цель:

- проверить `inventory_scenes`;
- проверить `prepare_dataset` и prepared manifest adapter;
- проверить `prepare_inference_scenes`;
- проверить `run_pseudolabel_inference`;
- проверить `validate_probability_maps`;
- проверить MLflow finalize.
