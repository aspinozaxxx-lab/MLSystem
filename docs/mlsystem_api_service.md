# MLSystem API service

`mlsystem-api` - внутренний FastAPI-сервис, через который Airflow запускает MLSystem stages.

```text
Airflow task -> mlsystem-api -> persistent API job -> subprocess worker -> stage registry -> production code
```

Сервис не должен содержать бизнес-логику stage. Он создает job, запускает worker subprocess, хранит status/report/error и возвращает результат через стабильный API.

## Endpoints

| Endpoint | Назначение |
|---|---|
| `GET /health` | Легкая проверка, что сервис жив. |
| `GET /ready` | Проверка status root, job root, env и импорта stage registry. |
| `GET /api/v1/stages` | Список `MAIN_DAG_STAGES`, registry stages, aliases, legacy fallback и pools. |
| `POST /api/v1/runs/{run_id}/stages/{stage}/start` | Создать persistent job и запустить stage worker subprocess. |
| `GET /api/v1/jobs/{job_id}` | Получить state/report/error/artifacts job. |
| `GET /api/v1/runs/{run_id}/summary` | Прочитать summary текущего run из status artifacts. |
| `POST /api/v1/debug/run-stage-sync` | Только короткая локальная отладка, не для train/inference. |
| `POST /api/v1/debug/inference-rabbit-smoke` | Experimental RabbitMQ smoke. |

## Job states

```text
queued
running
succeeded
failed
cancelled
timed_out
```

## Job store

По умолчанию:

```text
/data/mlsystem/api/jobs/<job_id>/
```

Файлы:

- `job.json` - текущее состояние job;
- `request.json` - masked request;
- `report.json` - stage payload;
- `error.json` - ошибка и traceback tail;
- `stdout_tail.txt`, `stderr_tail.txt` - хвосты worker output.

Env:

```text
MLSYSTEM_API_JOB_ROOT=/data/mlsystem/api/jobs
```

## Stage reports

Полные stage reports лежат не в XCom, а в status root:

```text
/data/mlsystem/airflow/status/<run_id>/stages/<stage>.json
/data/mlsystem/airflow/status/<run_id>/stages/<stage>.report.md
```

Markdown report содержит:

- stage/status/run_id/job_id/duration;
- checks;
- counters;
- metrics;
- warnings/errors;
- artifacts;
- resource summary;
- container и host paths;
- diagnostic commands при failure.

## XCom-safe response

API возвращает полный report в job status, но Airflow wrapper делает compact summary и пушит только scalar XCom keys:

```text
stage=prepare_dataset
status=success
job_id=<job_id>
summary=prepare_dataset completed
report_path=/opt/airflow/mlsystem_runs/<run>/stages/prepare_dataset.report.md
stage_json_path=/opt/airflow/mlsystem_runs/<run>/stages/prepare_dataset.json
counter_total_scenes=24
metric_pixel_f1=0.73
url_mlflow_run=http://...
```

XCom не хранит `resources`, `stage_report`, большие списки scenes или полный dump artifacts.

## MLflow URL

`create_mlflow_run` пишет в report и XCom:

```text
mlflow_run_id
url_mlflow_run
url_mlflow_experiment
```

Если задан `MLSYSTEM_MLFLOW_PUBLIC_URL`, ссылки строятся от него. Если public URL не задан, используется tracking URI и report пишет warning.

## Secrets

Request/job store маскирует ключи, содержащие:

```text
password, secret, token, access_key, secret_key,
AWS_SECRET_ACCESS_KEY, MLFLOW_TRACKING_PASSWORD,
MINIO_SECRET_KEY, S3_SECRET_KEY
```

Значения `SECRET/TOKEN/PASSWORD` из env не печатаются в логах и не коммитятся.

## Dependency risk

Текущая схема runtime-зависимостей еще требует отдельного follow-up:

1. Завести pinned requirements для Airflow/API image.
2. Собирать отдельный image вместо runtime `_PIP_ADDITIONAL_REQUIREMENTS`.
3. Зафиксировать версии `fastapi`, `uvicorn`, `mlflow`, `boto3`, `rasterio`, `shapely`, `pika`, `tritonclient`, `torch`.
