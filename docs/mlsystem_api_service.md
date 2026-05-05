# MLSystem API service

`mlsystem-api` - внутренний FastAPI-сервис для запуска MLSystem stages из Airflow через устойчивый HTTP API.

Схема:

```text
Airflow task -> mlsystem-api -> persistent API job -> subprocess worker -> stage registry -> production code
```

Airflow больше не должен исполнять тяжелый MLSystem-код внутри своего процесса. Он создает API job, опрашивает статус и валит task, если job завершился ошибкой.

## Endpoints

| Endpoint | Назначение |
|---|---|
| `GET /health` | Легкая проверка, что сервис жив. Без секрета. |
| `GET /ready` | Проверяет status root, job root, импорт stage registry и обязательные env-ключи. Если базовая готовность нарушена, возвращает HTTP 503. RabbitMQ не делает ready=false, потому что workflow experimental. |
| `GET /api/v1/stages` | Возвращает `MAIN_DAG_STAGES`, registry stages, aliases, legacy fallback stages и pools. |
| `POST /api/v1/runs/{run_id}/stages/{stage}/start` | Создает persistent job и запускает stage worker subprocess. |
| `GET /api/v1/jobs/{job_id}` | Возвращает состояние job, report/error/artifacts. |
| `GET /api/v1/runs/{run_id}/summary` | Читает summary из Airflow status artifacts. |
| `POST /api/v1/debug/run-stage-sync` | Синхронный debug endpoint для коротких локальных проверок. Не использовать для train/inference. |
| `POST /api/v1/debug/inference-rabbit-smoke` | Smoke-проверка RabbitMQ inference backend, experimental. |

## Job states

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`
- `timed_out`

## Где лежат job artifacts

По умолчанию:

```text
/data/mlsystem/api/jobs/<job_id>/
```

Файлы:

- `job.json` - состояние job;
- `request.json` - запрос с замаскированными секретами;
- `report.json` - результат stage;
- `error.json` - ошибка, если stage упал;
- `stdout_tail.txt`, `stderr_tail.txt` - хвосты worker output.

Путь задается через:

```text
MLSYSTEM_API_JOB_ROOT=/data/mlsystem/api/jobs
```

## Секреты

API/job store маскирует ключи, содержащие:

```text
password, secret, token, access_key, secret_key,
AWS_SECRET_ACCESS_KEY, MLFLOW_TRACKING_PASSWORD,
MINIO_SECRET_KEY, S3_SECRET_KEY
```

Также маскируются env-переменные с `SECRET`, `TOKEN`, `PASSWORD`.

## Авторизация

Мутирующие и debug endpoints используют bearer token, если задан:

```text
MLSYSTEM_API_TOKEN
```

`/health`, `/ready`, `/api/v1/stages` оставлены без token для внутренней диагностики контейнеров.

Секрет генерируется/сохраняется через ansible template `/etc/mlsystem/gpu-platform.env`; значение не коммитится.
