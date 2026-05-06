# Выполнение Airflow stages через API

Airflow не исполняет MLSystem domain code напрямую. Каждый `PythonOperator` вызывает `mlsystem-api`, получает `job_id`, опрашивает состояние job и пишет короткие scalar XCom keys.

## Production env

```text
MLSYSTEM_AIRFLOW_EXECUTION_MODE=api
MLSYSTEM_API_URL=http://mlsystem-api:8088
MLSYSTEM_API_TOKEN=<secret from env>
MLSYSTEM_AIRFLOW_API_POLL_SEC=10
```

Локального execution fallback в Airflow wrapper нет. Unit tests должны мокать API client или вызывать stage entrypoints напрямую.

## Runtime artifacts

Полные отчеты не хранятся в Airflow metadata DB и не коммитятся в git:

```text
/data/mlsystem/api/jobs/<job_id>/
/data/mlsystem/airflow/status/<run_id>/
```

В Airflow XCom остаются только небольшие scalar values:

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

`return_value` для stage tasks отключен через `do_xcom_push=False`.

## Диагностика

```bash
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow dags list-import-errors"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow tasks list mlsystem_experiment_pipeline"
ssh gpu-mlserver "docker exec mlsystem-gpu-airflow-scheduler airflow pools list"
ssh gpu-mlserver "docker logs --tail 300 mlsystem-gpu-api"
```

По `job_id`:

```bash
ssh gpu-mlserver "cat /data/mlsystem/api/jobs/<job_id>/job.json"
ssh gpu-mlserver "cat /data/mlsystem/api/jobs/<job_id>/error.json"
ssh gpu-mlserver "tail -n 100 /data/mlsystem/api/jobs/<job_id>/stderr_tail.txt"
```

По stage:

```bash
ssh gpu-mlserver "cat /data/mlsystem/airflow/status/<run_id>/stages/<stage>.report.md"
ssh gpu-mlserver "cat /data/mlsystem/airflow/status/<run_id>/stages/<stage>.json"
```
