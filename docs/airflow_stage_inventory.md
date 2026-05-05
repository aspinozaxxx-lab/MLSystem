# Инвентаризация Airflow pipeline

Дата обновления: 2026-05-05.

Основной DAG `mlsystem_experiment_pipeline` остается линейной цепочкой `PythonOperator`, но исполнение stage в production идет через API:

```text
Airflow PythonOperator -> mlsystem-api -> persistent job -> subprocess worker -> stage registry -> production modules
```

Полный источник истины по stage лежит в status artifacts:

```text
/data/mlsystem/airflow/status/<run_id>/stages/<stage>.json
/data/mlsystem/airflow/status/<run_id>/stages/<stage>.report.md
```

## Найденные DAG

| DAG | Файл | Назначение |
|---|---|---|
| `mlsystem_experiment_pipeline` | `airflow/dags/mlsystem_experiment_pipeline.py` | Основной экспериментальный pipeline. |
| `mlsystem_smoke_pipeline` | `airflow/dags/mlsystem_smoke_pipeline.py` | Smoke-запуск основного DAG. |
| `mlsystem_maintenance_cleanup` | `airflow/dags/mlsystem_maintenance_cleanup.py` | Регламентная очистка, не входит в экспериментальный pipeline. |

## MAIN_DAG_STAGES

| # | stage | Entry point | Pool | Статус реализации |
|---:|---|---|---|---|
| 1 | `inventory_scenes` | `stages.inventory_scenes.run` | `io_light` | Реальный stage registry entrypoint. |
| 2 | `prepare_dataset` | `stages.prepare_dataset.run` | `cpu_heavy` | Реальный stage registry entrypoint, object-balanced split для `schema_version>=2`. |
| 3 | `create_mlflow_run` | legacy fallback `_create_mlflow_run` | `io_light` | Совместимость; пишет MLflow run URL и counters. |
| 4 | `train_model` | legacy fallback `_run_training_pipeline` | `gpu_training` | Обучение и MLflow оставлены как есть. |
| 5 | `evaluate_pixel_metrics` | legacy fallback | `cpu_light` | Извлекает доступные pixel metrics из `training_result.json`. |
| 6 | `predict_validation_scenes` | legacy fallback | `gpu_inference` | Compatibility gate; отдельный prediction пока не выполняется, report явно пишет skipped/processed counts. |
| 7 | `vectorize_validation_predictions` | legacy fallback | `cpu_heavy` | Compatibility/deferred; пишет summary artifacts и counters. |
| 8 | `compute_f1` | legacy fallback | `cpu_heavy` | Новое имя stage. Старое `compute_object_f1` оставлено совместимым fallback alias. |
| 9 | `prepare_inference_scenes` | `stages.prepare_inference_scenes.run` | `io_light` | Реальный stage registry entrypoint. |
| 10 | `run_pseudolabel_inference` | `stages.pseudolabel_inference.run` | `gpu_inference` | GPU direct Triton path; CPU vectorization не выполняет. |
| 11 | `validate_probability_maps` | `stages.probability_maps.run` | `cpu_heavy` | Валидирует/собирает индекс probability maps. Старое имя `stitch_probability_maps` осталось alias. |
| 12 | `vectorize_pseudolabel` | `stages.vectorize_pseudolabel.run` | `cpu_heavy` | Legacy mode по умолчанию; `block_parallel` доступен через feature flag. |
| 13 | `postprocess_pseudolabel` | `stages.postprocess_pseudolabel.run` | `cpu_heavy` | Compatibility stage поверх postprocess metrics. |
| 14 | `export_pseudolabel_artifacts` | `stages.export_pseudolabel.run` | `io_light` | Проверяет обязательные pseudolabel artifacts. |
| 15 | `generate_prediction_examples` | legacy fallback | `cpu_heavy` | Проверяет HTML examples и пишет counters. |
| 16 | `log_mlflow_artifacts` | legacy fallback | `io_light` | Логирует summary artifacts в MLflow. |
| 17 | `write_codex_api_summary` | legacy fallback | `io_light` | Пишет `run_summary.json` и `codex_summary.json`. |
| 18 | `finalize_mlflow_run` | legacy fallback | `io_light` | Финализирует MLflow run и cleanup. |

## XCom counters и metrics

XCom является только коротким key/value интерфейсом для Airflow UI. Полные reports и большие списки не пушатся.

Базовые keys:

```text
stage, status, run_id, job_id, summary, report_path, stage_json_path,
warnings_count, errors_count, duration_sec, pool, execution_mode
```

Счетчики и метрики:

```text
counter_<name>
metric_<name>
url_mlflow_run
url_mlflow_experiment
```

Все значения проходят через scalar-safe normalizer: `numpy` числа приводятся к `int/float`, `NaN/Inf` становятся `None`, `Path/datetime` становятся строками, dict/list сериализуются короткой строкой и не сохраняются как объект.

## Важные artifacts по stage

| Stage | Основные artifacts |
|---|---|
| `inventory_scenes` | `inventory_scenes.json`, `scene_inventory_report.txt`, `matched_scenes.txt`, `missing_scenes.txt`, `scene_matching_report.json` |
| `prepare_dataset` | `dataset_manifest.json`, `train_scenes.txt`, `val_scenes.txt`, `scene_object_counts.txt`, `train_val_split.txt`, `split_summary.json`, `prepare_dataset_input_audit.json` |
| `evaluate_pixel_metrics` | `pixel_metrics.json`, `pixel_metrics.txt` |
| `predict_validation_scenes` | `validation_prediction_summary.json`, `validation_prediction_summary.txt` |
| `vectorize_validation_predictions` | `validation_vectorization_summary.json`, `validation_vectorization_summary.txt`, `validation_objects_by_scene.txt` |
| `compute_f1` | `f1_summary.json`, `f1_summary.txt`, `object_matching_report.json`, `object_matching_report.txt` |
| `prepare_inference_scenes` | `inference_manifest.json`, `inference_scenes.txt`, `inference_inventory_report.json` |
| `run_pseudolabel_inference` | `inference_results.json`, `probability_maps_index.json`, `pseudolabel_scene_results_manifest.json`, `inference_timing_report.json` |
| `vectorize_pseudolabel` | legacy: `vectorization_summary.json`; block_parallel: `vectorization_block_parallel/*`, accepted GeoJSON |

## External services

- S3/MinIO используется в `inventory_scenes`, `prepare_dataset`, training/inference storage adapters.
- MLflow используется в legacy fallback stages и training/pseudolabel wrappers.
- Triton HTTP используется в `run_pseudolabel_inference` через текущий direct path.
- RabbitMQ сейчас только experimental/smoke path; основной DAG его не использует.
- Airflow metadata хранит только task state и scalar XCom; большие данные идут через status artifacts.

## Текущий технический долг

- `predict_validation_scenes`, `vectorize_validation_predictions`, `compute_f1` пока являются compatibility stages. Они не переписывают модельный inference/training code, а извлекают доступные данные из существующих artifacts.
- `validate_probability_maps` не выносит actual stitching из `SceneInferenceRunner`; он валидирует индекс готовых probability maps.
- `vectorize_pseudolabel` имеет новый `block_parallel` path, но default остается `legacy`, чтобы не ломать существующие runs без отдельного включения.
