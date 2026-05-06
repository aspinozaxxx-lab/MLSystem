# MLSystem API service

`mlsystem-api` является единственной production точкой исполнения MLSystem stages для Airflow.

## Endpoints

| Endpoint | Назначение |
| --- | --- |
| `GET /health` | Быстрая проверка процесса API. |
| `GET /ready` | Проверка job root, status root, env и stage registry. |
| `GET /api/v1/stages` | Текущие `MAIN_DAG_STAGES`, registry stages, dispatcher stages и pools. |
| `POST /api/v1/runs/{run_id}/stages/{stage}/start` | Создать persisted API job для stage. |
| `GET /api/v1/jobs/{job_id}` | Получить состояние job, report, artifacts и error. |
| `GET /api/v1/runs/{run_id}/summary` | Прочитать summary текущего run из status artifacts. |
| `POST /api/v1/debug/run-stage-sync` | Короткий debug endpoint для tests/handtests, не для долгих production stages. |

Текущий production inference идет через direct Triton path.

## Job storage

По умолчанию job state лежит вне git:

```text
/data/mlsystem/api/jobs/<job_id>/
```

Файлы job:

- `job.json`
- `request.json`
- `report.json`
- `error.json`, если stage failed
- `stdout_tail.txt`
- `stderr_tail.txt`

Stage artifacts лежат отдельно:

```text
/data/mlsystem/airflow/status/<run_id>/
```

## Security

API token берется из env. Secrets maskируются перед записью request/report/log tail. В git не должны попадать env files, job files, stage reports, runtime outputs, probability maps или accepted GeoJSON.
