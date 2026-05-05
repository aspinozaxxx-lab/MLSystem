# Airflow API execution

Airflow остается оркестратором: DAG, task_id, pools и линейные зависимости сохраняются. Исполнение MLSystem stage переносится в `mlsystem-api`.

## Execution modes

```text
MLSYSTEM_AIRFLOW_EXECUTION_MODE=api
```

В production compose это значение задается через `/etc/mlsystem/gpu-platform.env`.

Для локальной отладки и unit tests доступен fallback:

```text
MLSYSTEM_AIRFLOW_EXECUTION_MODE=local
```

В local mode `run_airflow_stage` напрямую вызывает `run_stage`.

## Как Airflow вызывает API

1. Airflow task берет `dag_run.conf`.
2. Берет `dag_run.run_id`.
3. POST в `MLSYSTEM_API_URL/api/v1/runs/{run_id}/stages/{stage}/start`.
4. Получает `job_id`.
5. Poll `GET /api/v1/jobs/{job_id}` с интервалом `MLSYSTEM_AIRFLOW_API_POLL_SEC`.
6. При `succeeded` возвращает report.
7. При `failed`, `cancelled`, `timed_out` поднимает exception, чтобы Airflow task стал failed.

## Env

```text
MLSYSTEM_API_URL=http://mlsystem-api:8088
MLSYSTEM_API_TOKEN=<secret from env>
MLSYSTEM_AIRFLOW_API_POLL_SEC=10
MLSYSTEM_AIRFLOW_API_NO_PROGRESS_TIMEOUT_SEC=3600
```

## Что видно в Airflow logs

Минимально:

- `stage`;
- `job_id`;
- state transitions;
- summary при success;
- error message и traceback tail при failure.

Секреты не печатаются: payload проходит через masking.

## Диагностика failed stage

1. Найти `job_id` в Airflow task log.
2. На сервере:

```bash
cat /data/mlsystem/api/jobs/<job_id>/job.json
cat /data/mlsystem/api/jobs/<job_id>/error.json
tail -n 100 /data/mlsystem/api/jobs/<job_id>/stderr_tail.txt
```

3. Проверить stage artifacts:

```bash
find /data/mlsystem/airflow/status -maxdepth 3 -type f -name '<stage>.json'
```

На сервере не править файлы руками; исправления только через repo + CI/CD.
