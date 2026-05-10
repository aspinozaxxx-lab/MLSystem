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

## InferenceEngine integration

For pseudolabel runs with `pseudolabel.source=inference_engine`, `mlsystem-api` remains the Airflow stage execution endpoint, but it does not run pseudolabel domain logic. The `run_pseudolabel_inference` stage submits an InferenceEngine job to `INFERENCE_ENGINE_API_URL`, polls `/api/v1/jobs/{job_id}`, and then validates artifacts written into the Airflow run directory.

Required env:

- `INFERENCE_ENGINE_API_URL`
- `INFERENCE_ENGINE_API_TOKEN`
- `INFERENCE_ENGINE_RABBITMQ_URL`
- `INFERENCE_ENGINE_JOB_ROOT`
- `INFERENCE_ENGINE_SPOOL_ROOT`
- `INFERENCE_ENGINE_TRITON_URL`

The downstream pseudolabel stages are compatibility validators in this mode.
