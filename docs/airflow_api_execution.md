# Выполнение Airflow stages через API

Airflow остается оркестратором: DAG, task_id, pools и зависимости сохраняются. Код MLSystem исполняется в `mlsystem-api`.

## Execution modes

Production:

```text
MLSYSTEM_AIRFLOW_EXECUTION_MODE=api
MLSYSTEM_API_URL=http://mlsystem-api:8088
MLSYSTEM_API_TOKEN=<secret from env>
```

Локальный fallback для unit tests и аварийной отладки:

```text
MLSYSTEM_AIRFLOW_EXECUTION_MODE=local
```

## Что делает Airflow task

1. Читает `dag_run.conf` и `dag_run.run_id`.
2. Создает API job через `POST /api/v1/runs/{run_id}/stages/{stage}/start`.
3. Пишет `job_id` в лог.
4. Poll-ит `GET /api/v1/jobs/{job_id}`.
5. Печатает человекочитаемый stage report.
6. Пушит только scalar XCom keys.
7. При `failed/cancelled/timed_out` поднимает exception.

## XCom key/value

`PythonOperator` создан с `do_xcom_push=False`, поэтому `return_value` не должен попадать в XCom. Wrapper явно пушит маленькие keys:

```text
stage
status
run_id
job_id
summary
report_path
stage_json_path
warnings_count
errors_count
duration_sec
pool
execution_mode
counter_<name>
metric_<name>
url_mlflow_run
url_mlflow_experiment
```

Все значения проходят через `xcom_safe_value`:

- `numpy.int64` -> `int`;
- `numpy.float32` -> `float`;
- `NaN/Inf` -> `None`;
- `Path` -> строка;
- `datetime/date` -> ISO строка;
- dict/list/tuple/set -> короткая JSON-строка, не объект;
- строки обрезаются до безопасной длины.

Источник истины для больших данных: stage artifacts, а не XCom.

## Что видно в логах

Каждый stage печатает:

- stage/status/run_id/job_id/duration;
- checks;
- counters;
- metrics;
- warnings/errors;
- artifacts;
- resource summary;
- container path и host path;
- диагностические команды при failure.

В XCom для GPU stages дополнительно появляются только короткие scalar keys:

```text
requested_pool
effective_pool
cuda_available
gpu_name
device
```

GPU memory и подробный resource snapshot остаются в full stage report и resource artifacts, а не в XCom. CPU stages не должны пушить `cuda_available` / `gpu_name`.

## Диагностика на сервере

Только чтение/диагностика, без ручного редактирования файлов:

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags list-import-errors"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow tasks list mlsystem_experiment_pipeline"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow pools list"
ssh gpu-mlserver "docker logs --tail 300 mlsystem-gpu-api"
```

По `job_id` из Airflow log:

```bash
ssh gpu-mlserver "cat /data/mlsystem/api/jobs/<job_id>/job.json"
ssh gpu-mlserver "cat /data/mlsystem/api/jobs/<job_id>/error.json"
ssh gpu-mlserver "tail -n 100 /data/mlsystem/api/jobs/<job_id>/stderr_tail.txt"
```

По stage artifact:

```bash
ssh gpu-mlserver "cat /data/mlsystem/airflow/status/<run_id>/stages/<stage>.report.md"
ssh gpu-mlserver "cat /data/mlsystem/airflow/status/<run_id>/stages/<stage>.json"
```
