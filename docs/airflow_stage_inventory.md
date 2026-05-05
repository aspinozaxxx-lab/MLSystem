# Airflow pipeline after stage refactor

Дата обновления: 2026-05-05.

Документ описывает фактическое состояние после локального рефакторинга stage routing. Основной DAG по-прежнему строит линейную цепочку `PythonOperator`, но список stages стал крупнее по смысловым блокам, а новые ключевые stages имеют entrypoint в `mlsystem/src/pipeline/stages`.

## DAGs found

| DAG | Файл | Назначение | Примечания |
|---|---|---|---|
| `mlsystem_experiment_pipeline` | `airflow/dags/mlsystem_experiment_pipeline.py` | Основной экспериментальный pipeline. | Создает один `PythonOperator` на каждый stage из `MAIN_DAG_STAGES`. Pool берется из `STAGE_POOLS`. |
| `mlsystem_smoke_pipeline` | `airflow/dags/mlsystem_smoke_pipeline.py` | Smoke-запуск основного DAG. | Через `TriggerDagRunOperator` запускает основной DAG со smoke-конфигом. |
| `mlsystem_maintenance_cleanup` | `airflow/dags/mlsystem_maintenance_cleanup.py` | Регламентная очистка. | Найден поиском DAG, не входит в experiment/smoke pipeline. |

## High-level flow

1. `inventory_scenes`
2. `prepare_dataset`
3. training/evaluation compatibility stages
4. `prepare_inference_scenes`
5. pseudolabel inference/probability validation/vectorize/postprocess/export
6. reporting/finalize

## Main DAG: `mlsystem_experiment_pipeline`

`airflow/dags/mlsystem_experiment_pipeline.py` импортирует `MAIN_DAG_STAGES`, `STAGE_POOLS` и `run_airflow_stage` из `mlsystem/src/pipeline/airflow_tasks.py`. Внутри DAG каждая задача вызывает `_run_stage`, который передает `dag_run.conf`, `dag_run.run_id` и `/opt/airflow/mlsystem_runs` в `run_airflow_stage`.

В production `MLSYSTEM_AIRFLOW_EXECUTION_MODE=api`, поэтому `run_airflow_stage` вызывает `mlsystem-api` и опрашивает persistent API job. Для локальных unit tests и emergency остается `MLSYSTEM_AIRFLOW_EXECUTION_MODE=local`, где stage исполняется напрямую.

| # | task_id | Назначение | Airflow wrapper | Entry point / callable | Основная работа | Вход | Выход | Pool | Dependencies |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | `inventory_scenes` | Инвентаризация входных сцен и layout. | `PythonOperator` | `stages.inventory_scenes.run` через registry. | Проверяет config, S3 layout, `images_uri`, `layout_uri`, `scenes_file`, `annotation_file`; строит scene matching. | `dag_run.conf`, S3/MinIO, `images_uri`, `layout_uri`. | `inventory_scenes.json`, `scene_inventory_report.txt`, `matched_scenes.txt`, `missing_scenes.txt`, `scene_matching_report.json`. | `io_light` | upstream: нет; downstream: `prepare_dataset` |
| 2 | `prepare_dataset` | Формирование dataset manifest и train/val split. | `PythonOperator` | `stages.prepare_dataset.run` через registry. | Читает inventory, считает объекты через `data.dataset_split`, делает object-balanced split, пишет compatibility manifest. | `inventory_scenes.json`, annotation GeoJSON, matched scenes, `preprocess.*`. | `dataset_manifest.json`, `train_scenes.txt`, `val_scenes.txt`, `scene_object_counts.txt`, `train_val_split.txt`, `split_summary.json`, `dataset_validation_report.json`. | `cpu_heavy` | `inventory_scenes` -> `create_mlflow_run` |
| 3 | `create_mlflow_run` | Создает MLflow run. | `PythonOperator` | compatibility fallback в `airflow_tasks._run_legacy_stage`. | `_create_mlflow_run`, прямой MLflow API. | config, MLflow settings. | summary `mlflow`, stage JSON. | `io_light` | `prepare_dataset` -> `train_model` |
| 4 | `train_model` | Обучение модели. | `PythonOperator` | compatibility fallback. | `_run_training_pipeline` -> `TrainingPipeline().run` -> `real_train.run_real_train`. | config, dataset artifacts, MLflow run id. | `training_result.json`, checkpoint, train history, MLflow metrics/artifacts. | `gpu_training` | `create_mlflow_run` -> `evaluate_pixel_metrics` |
| 5 | `evaluate_pixel_metrics` | Читает pixel metrics обучения. | `PythonOperator` | compatibility fallback. | `_read_training_result`. | `training_result.json`. | stage JSON с pixel metrics. | `cpu_light` | `train_model` -> `predict_validation_scenes` |
| 6 | `predict_validation_scenes` | Compatibility gate по checkpoint. | `PythonOperator` | compatibility fallback. | Проверяет наличие checkpoint; фактическое validation prediction пока остается в старом контуре. | training result, explicit checkpoint config. | stage JSON. | `gpu_training` | `evaluate_pixel_metrics` -> `vectorize_validation_predictions` |
| 7 | `vectorize_validation_predictions` | Compatibility/deferred stage. | `PythonOperator` | compatibility fallback. | Читает `pseudolabel_summary.json`, если он уже есть. | local status dir. | stage JSON. | `cpu_heavy` | `predict_validation_scenes` -> `compute_object_f1` |
| 8 | `compute_object_f1` | Compatibility/deferred object metrics stage. | `PythonOperator` | compatibility fallback. | Читает `postprocess_metrics` из training result. | `training_result.json`. | stage JSON. | `cpu_heavy` | `vectorize_validation_predictions` -> `prepare_inference_scenes` |
| 9 | `prepare_inference_scenes` | Отдельная подготовка списка сцен для inference. | `PythonOperator` | `stages.prepare_inference_scenes.run` через registry. | Поддерживает `run_on=dataset_scenes`, `all_images`, `explicit_scene_list`, а также совместимые train/val режимы. | `inventory_scenes.json`, `dataset_manifest.json`, `pseudolabel.run_on`. | `inference_manifest.json`, `inference_scenes.txt`, `inference_inventory_report.json`. | `io_light` | `compute_object_f1` -> `run_pseudolabel_inference` |
| 10 | `run_pseudolabel_inference` | GPU pseudolabel inference. | `PythonOperator` | `stages.pseudolabel_inference.run` через registry. | Читает `inference_manifest.json`, передает `scene_entries` в старый `_run_pseudolabel_pipeline(..., stage_mode="inference")`; использует direct Triton path. | checkpoint, config, `inference_manifest.json`, inference config. | `inference_results.json`, `probability_maps_index.json`, `pseudolabel_scene_results_manifest.json`, `coverage_report.json`. | `gpu_training` | `prepare_inference_scenes` -> `validate_probability_maps` |
| 11 | `validate_probability_maps` | Проверка и сбор индекса probability maps. | `PythonOperator` | `stages.probability_maps.run` через registry. | Не переносит actual stitching: карты собираются внутри inference runner; stage валидирует coverage/index. Старое имя `stitch_probability_maps` оставлено alias. | `coverage_report.json`, `pseudolabel_scene_results_manifest.json`. | `probability_maps_index.json`, stage JSON. | `cpu_heavy` | `run_pseudolabel_inference` -> `vectorize_pseudolabel` |
| 12 | `vectorize_pseudolabel` | CPU vectorization compatibility stage. | `PythonOperator` | `stages.vectorize_pseudolabel.run` через registry. | Если accepted GeoJSON отсутствует, вызывает `_run_pseudolabel_pipeline(..., stage_mode="postprocess")`. | probability map artifacts, postprocess config. | accepted GeoJSON, `vectorization_summary.json`, postprocess artifacts. | `cpu_heavy` | `validate_probability_maps` -> `postprocess_pseudolabel` |
| 13 | `postprocess_pseudolabel` | Проверка postprocess metrics. | `PythonOperator` | `stages.postprocess_pseudolabel.run` через registry. | Читает `training_result.postprocess_metrics`, пишет summary. | `training_result.json`. | `postprocess_summary.json`, stage JSON. | `cpu_heavy` | `vectorize_pseudolabel` -> `export_pseudolabel_artifacts` |
| 14 | `export_pseudolabel_artifacts` | Проверка обязательных pseudolabel artifacts. | `PythonOperator` | `stages.export_pseudolabel.run` через registry. | Проверяет accepted GeoJSON, coverage, summary и список сцен. | local run dir. | stage JSON с artifact map. | `io_light` | `postprocess_pseudolabel` -> `generate_prediction_examples` |
| 15 | `generate_prediction_examples` | Проверка HTML-примеров. | `PythonOperator` | compatibility fallback. | Проверяет `prediction_examples.html`. | local run dir. | stage JSON. | `cpu_heavy` | `export_pseudolabel_artifacts` -> `log_mlflow_artifacts` |
| 16 | `log_mlflow_artifacts` | Логирование summaries в MLflow. | `PythonOperator` | compatibility fallback. | `_write_run_summaries`, `mlflow.log_artifact`. | summary, training result, MLflow run id. | `run_summary.json`, `codex_summary.json`, MLflow artifacts. | `io_light` | `generate_prediction_examples` -> `write_codex_api_summary` |
| 17 | `write_codex_api_summary` | Локальные summary-файлы. | `PythonOperator` | compatibility fallback. | `_write_run_summaries`. | summary/training result. | `run_summary.json`, `codex_summary.json`. | `io_light` | `log_mlflow_artifacts` -> `finalize_mlflow_run` |
| 18 | `finalize_mlflow_run` | Cleanup и финализация MLflow run. | `PythonOperator` | compatibility fallback. | `_finalize_mlflow_run`, `_cleanup_runtime_intermediates`, MLflow tags/status. | summary, MLflow run id, cleanup config. | финальный summary/status. | `io_light` | `write_codex_api_summary` -> downstream: нет |

## Stage contracts

### `inventory_scenes`

- Entry point: `mlsystem/src/pipeline/stages/inventory_scenes.py::run`.
- Fail conditions: invalid S3 layout, недоступный `images_uri`, отсутствующие `scenes_file` или `annotation_file`, missing scenes, ambiguous matches, ноль matched scenes, ошибка записи inventory.
- Warnings: дополнительные TIFF/TIF в `images_uri`, которые не входят в `scenes_file`.
- Airflow log: `StageReport.to_airflow_log()` с checks, counters, warnings/errors и artifact paths.

### `prepare_dataset`

- Entry point: `mlsystem/src/pipeline/stages/prepare_dataset.py::run`.
- Split strategy без явного `preprocess.split_strategy`: `legacy_75_25` для configs без `schema_version` или с `schema_version<2`.
- Для новых configs `schema_version>=2` default: `object_balanced`.
- Явные режимы: `preprocess.split_strategy="object_balanced"` или `preprocess.split_strategy="legacy_75_25"`.
- Перед `train_model` Airflow передает `dataset_manifest.json` как `preprocess.prepared_dataset_manifest`; `real_train` использует этот split и не пересчитывает train/val заново.
- CRS policy: `preprocess.annotation_crs` имеет приоритет; затем CRS из GeoJSON; если CRS отсутствует и `allow_inferred_annotation_crs=true`, геометрический fallback делает эвристику и пишет warning. Если inference CRS отключен, geometry fallback не используется.
- Fail conditions: нет inventory/matched scenes, annotation не читается, object counting падает, train/val пустой, train/val пересекаются, lost scenes.

### `prepare_inference_scenes`

- Entry point: `mlsystem/src/pipeline/stages/prepare_inference_scenes.py::run`.
- Поддержанные режимы: `dataset_scenes`, `all_images`, `explicit_scene_list`, совместимые `validation_scenes` и `train_scenes`.
- Fail conditions: неизвестный `run_on`, missing scenes для dataset/explicit/train/val режимов, ноль выбранных сцен.
- Bad scene policy фиксируется в manifest как `pseudolabel.bad_scene_policy`, но глубокая проверка zero-byte/bad TIFF пока остается downstream-задачей inference.

## External services usage

- S3/MinIO: `inventory_scenes` использует `storage.s3.list_s3_objects`, `find_layout_files`, `read_s3_text`; `prepare_dataset` читает annotation через `read_s3_text` и строит raster paths через `raster_path_for_s3_key`.
- MLflow: пока остается в compatibility fallback внутри `airflow_tasks.py` и training/pseudolabel wrappers.
- Triton HTTP: production inference path находится в `pseudolabel_pipeline.py` и `inference/triton_client.py`; новый stage `run_pseudolabel_inference` вызывает этот direct path.
- RabbitMQ: найден только как smoke/utility в `inference/rabbitmq_triton_queue.py`; основной DAG его не вызывает.
- Airflow metadata: основной DAG использует `PythonOperator` и pools; task-to-task data flow в коде в основном идет через local status dir, `summary.json` и stage JSON.

## Важные ограничения текущего рефакторинга

- Обучение, MLflow и `real_train.py` не переписаны.
- `run_stage` теперь сначала ищет stage entrypoint в registry; старые unchanged stages идут через compatibility fallback.
- `validate_probability_maps` пока не является настоящим отдельным stitch use case. Actual probability map accumulation остается внутри `SceneInferenceRunner`/`ProbabilityMapAccumulator`; старое имя `stitch_probability_maps` оставлено только alias для совместимости.
- `vectorize_pseudolabel` все еще вызывает старый compatibility `stage_mode="postprocess"`, если accepted GeoJSON еще не создан. Полное разделение vectorize/postprocess/export требует следующего патча.
