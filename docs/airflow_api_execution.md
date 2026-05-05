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

## XCom key/value

`PythonOperator` для stage tasks настроен с `do_xcom_push=False`, поэтому полный stage payload не попадает в `return_value`.
После завершения stage wrapper явно пушит маленькие XCom keys:

- `stage`
- `status`
- `run_id`
- `job_id`
- `summary`
- `report_path`
- `stage_json_path`
- `warnings_count`
- `errors_count`
- `duration_sec`
- `counter_<name>`, например `counter_total_scenes`, `counter_train_objects`, `counter_split_strategy`

Источник истины для больших данных остается в stage artifacts, а XCom нужен только для быстрых UI/debug checks.

## Human-readable task log

После завершения API job Airflow печатает общий отчет из formatter. Для типовых stages в логе должны быть видны ключевые счетчики:

- `inventory_scenes`: config/layout/images/scenes checks, `available_images`, `scene_rows`, `matched_scenes`, `missing_scenes`, `ambiguous_scenes`, пути к `inventory_scenes.json`, `scene_inventory_report.txt`, full report.
- `prepare_dataset`: `Input lineage`, `split_strategy`, `upstream_inventory_matched_scenes`, `selected_dataset_scenes`, `excluded_dataset_scenes`, `total_scenes`, `total_objects`, `scenes_without_objects`, `train_scenes/train_objects`, `val_scenes/val_objects`, пути к `prepare_dataset_input_audit.json`, `scene_object_counts.txt`, `train_val_split.txt`, `dataset_manifest.json`.
- `prepare_inference_scenes`: `run_on`, `inference_scenes`, `missing`, `bad_scene_policy`, source manifest, `inference_manifest.json`, `inference_scenes.txt`.
- `run_pseudolabel_inference`: `backend`, `scene_count`, `scenes_processed`, `scenes_failed`, `scenes_skipped`, `total_predicted_windows`, `probability_maps_index.json`, `inference_timing_report.json`.

При failure formatter добавляет диагностические команды с `job_id`, `<stage>.json`, `<stage>.report.md` и `docker logs --tail 300 mlsystem-gpu-api`.
