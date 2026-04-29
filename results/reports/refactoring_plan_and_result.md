# MLSystem refactoring plan and local result

Дата: 2026-04-28

## Ограничения

- Сервер не трогался.
- GitHub Actions не запускались.
- Push не выполнялся.
- Тяжелое обучение не запускалось.
- Geoalert и DVC не трогались.
- Job YAML, S3 layout, MLflow experiment names и ключевые artifact names сохранялись по возможности.

## Что создано

Создана целевая пакетная структура под `mlsystem/src`:

- `app/`
- `config/`
- `orchestration/`
- `storage/`
- `data/`
- `tiling/`
- `preprocessing/`
- `models/`
- `training/`
- `inference/`
- `postprocessing/`
- `metrics/`
- `reporting/`
- `debug/`
- `pipeline/`

## Что вынесено из `real_train.py`

Файл `real_train.py` оставлен compatibility facade. Старые функции продолжают существовать.

Вынесено и подключено через wrappers:

- S3 helpers:
  - `storage/s3.py`
  - wrappers: `_s3_parts`, `_mc_credentials`, `_s3_client`, `_aws_session`, `_list_s3_objects`, `_read_s3_text`, `_read_s3_json`, `_find_layout_files`
- Scene matching:
  - `data/scene_matching.py`
  - wrappers: `SceneMatch`, `_norm_scene_name`, `_scene_signature`, `_scene_score`, `build_scene_matching_report`, `_match_scenes`
- Normalization:
  - `preprocessing/normalization.py`
  - wrapper: `_normalize_image`
- Loss:
  - `training/losses.py`
  - wrapper: `_loss_fn`
- BatchNorm freeze helper:
  - `models/factory.py`
  - wrapper: `_set_batchnorm_eval`
- Tiling/stitching primitives:
  - `tiling/windows.py`
  - `tiling/stitching.py`
  - `_write_pseudolabel_outputs()` now delegates nested `origins`, `window_grid`, `tile_insert_slices`, `tile_weight_window` to `tiling.windows`
- Prediction examples reporting:
  - `reporting/prediction_examples.py`
  - wrapper: `_write_prediction_examples_report`

Также созданы целевые модули для следующего этапа:

- `data/layout_loader.py`
- `data/raster_metadata.py`
- `data/dataset_split.py`
- `data/dataset_builder.py`
- `postprocessing/thresholding.py`
- `postprocessing/vectorization.py`
- `postprocessing/filtering.py`
- `postprocessing/simplification.py`
- `postprocessing/pseudolabel_export.py`
- `models/unet.py`
- `models/checkpoints.py`
- `models/segformer.py`
- `models/deeplab.py`
- `training/train_loop.py`
- `training/early_stopping.py`
- `metrics/pixel_metrics.py`

## Что вынесено из `job_executor.py`

- Smoke train вынесен в `debug/smoke_train.py`.
- Epoch/history helper wrappers в `job_executor.py` делегируют в `debug/smoke_train.py`.
- Добавлен `orchestration/job_context.py`.
- Добавлен `pipeline/pipeline_factory.py`.
- Train/predict dispatch в `_execute()` теперь идет через `PipelineFactory`, а не напрямую через imports из `real_train.py`.

`job_executor.py` еще содержит MLflow lifecycle и run summary assembly. Это оставлено для совместимости в этом шаге; следующий этап должен вынести это в `reporting/run_summary.py` и generic reporter.

## DTO / contracts

Добавлен `pipeline/contracts.py`:

- `SceneRef`
- `SceneMatchReport`
- `TileWindow`
- `TileBatch`
- `PredictionTile`
- `ProbabilityMap`
- `VectorizationResult`
- `PostprocessResult`
- `TrainingResult`
- `PseudolabelResult`
- `ObjectMetricsResult`
- `JobRunSummary`

## Compatibility wrappers, которые остались

В `real_train.py`:

- `SceneMatch`
- `_s3_parts`
- `_mc_credentials`
- `_s3_client`
- `_aws_session`
- `_list_s3_objects`
- `_read_s3_text`
- `_read_s3_json`
- `_find_layout_files`
- `_norm_scene_name`
- `_scene_signature`
- `_scene_score`
- `build_scene_matching_report`
- `_match_scenes`
- `_normalize_image`
- `_loss_fn`
- `_set_batchnorm_eval`
- `_write_prediction_examples_report`
- `run_real_train`
- `run_debug_pseudolabel`

В старых root modules также оставлены старые import paths:

- `mlsystem/src/job_queue.py`
- `mlsystem/src/job_executor.py`
- `mlsystem/src/pipeline_config.py`
- `mlsystem/src/job_schema.py`
- `mlsystem/src/mlflow_adapter.py`
- `mlsystem/src/object_metrics.py`

Новые import paths уже существуют через target packages.

## Добавленные тесты

- `tests/test_scene_matching.py`
- `tests/test_window_generation.py`
- `tests/test_stitching.py`
- `tests/test_postprocess.py`
- `tests/test_job_schema.py`

Существующий `tests/test_object_metrics.py` сохранен.

## Проверки

Пройдены локально:

```powershell
python -m compileall mlsystem\src tests
python -m unittest discover -s tests
```

Результат unittest:

```text
Ran 26 tests in 0.016s
OK
```

## Что не менялось

- Формат job YAML.
- S3 layout.
- MLflow experiment names.
- Server runtime/deploy.
- GitHub Actions/workflows.
- Geoalert.
- DVC.
- Алгоритмическое поведение train/inference/postprocess намеренно не переписывалось.

## Оставшиеся риски

- `real_train.py` все еще содержит большой блок `_write_pseudolabel_outputs()`: inference/vectorization/export пока не полностью разбиты на независимые классы.
- `job_executor.py` все еще содержит MLflow lifecycle и run summary assembly.
- `models/factory.py` создан как целевой модуль, но legacy `_build_model()` пока оставлен в `real_train.py`, чтобы не менять поведение `segmentation_models_pytorch`.
- `postprocessing/*` модули покрыты unit tests, но production path пока использует их не полностью.
- `orchestration/job_executor.py` пока является compatibility import path для root executor.

## Следующий коммит

1. Полностью вынести `_write_pseudolabel_outputs()`:
   - `inference/scene_inference.py`
   - `tiling/stitching.py`
   - `postprocessing/vectorization.py`
   - `postprocessing/pseudolabel_export.py`
2. Вынести MLflow/run summary из `job_executor.py` в `reporting`.
3. Перенести root `object_metrics.py` на wrapper вокруг `metrics/object_metrics.py`, когда будет чистый индекс.
4. Перенести `cli.py` на новые public imports и убрать private imports из `real_train.py`.
5. После smoke/integration проверки удалить unreachable legacy code внутри wrapper functions.

---

# Refactoring step 2 result

Дата: 2026-04-28

## Измерение до шага

- `real_train.py`: 1774 строки.
- `job_executor.py`: 330 строк.
- `_write_pseudolabel_outputs()`: строки 601-1241, большой монолитный блок inference/stitching/vectorization/export/debug.
- Private imports из `real_train.py`:
  - `mlsystem/src/cli.py`: `_find_layout_files`, `_list_s3_objects`, `_read_s3_text`, `build_scene_matching_report`
  - `mlsystem/src/pipeline/training_pipeline.py`: `run_real_train`
- Почти пустые transitional modules до шага:
  - `inference/scene_inference.py`
  - `inference/probability_map.py`
  - `inference/predictor.py`
  - `data/dataset_builder.py`
  - `training/trainer.py`
  - несколько compatibility import wrappers в `app/`, `config/`, `orchestration/`, `reporting/`

## Измерение после шага

- `real_train.py`: 1156 строк.
- `job_executor.py`: 268 строк.
- `_write_pseudolabel_outputs()`: 28 строк.
- Уменьшение:
  - `real_train.py`: минус 618 строк.
  - `job_executor.py`: минус 62 строки.

## Что реально вынесено из `_write_pseudolabel_outputs()`

Основной pseudolabel flow вынесен в `pipeline/pseudolabel_pipeline.py`:

- разбор `predict.pseudolabel` / `postprocess` config;
- выбор `crop_mode` и `stitch_mode`;
- запуск inference по сценам;
- сбор raw vector features;
- postprocess;
- export artifacts;
- coverage/inference timing reports;
- object metrics artifacts;
- pseudolabel summary.

Inference одной сцены вынесен в `inference/scene_inference.py`:

- `SceneInferenceConfig`
- `SceneInferenceResult`
- `SceneInferenceRunner`
- `run_synthetic_scene_inference`

Probability map accumulation вынесен в `inference/probability_map.py`:

- `ProbabilityMapConfig`
- `ProbabilityMapAccumulator`
- `coverage_stats`

Vectorization/postprocess/export вынесены в:

- `postprocessing/vectorization.py`
- `postprocessing/filtering.py`
- `postprocessing/simplification.py`
- `postprocessing/pseudolabel_export.py`

Debug helpers вынесены в:

- `debug/pseudolabel_debug.py`
- `debug/stitching_debug.py`

Object metric artifact export вынесен в:

- `metrics/object_metric_artifacts.py`

MLflow/job artifact assembly частично вынесен из `job_executor.py` в:

- `reporting/artifact_reporter.py`

## Наполненные модули

Реальную логику теперь содержат:

- `inference/scene_inference.py`
- `inference/probability_map.py`
- `pipeline/pseudolabel_pipeline.py`
- `postprocessing/vectorization.py`
- `postprocessing/filtering.py`
- `postprocessing/pseudolabel_export.py`
- `debug/pseudolabel_debug.py`
- `debug/stitching_debug.py`
- `metrics/object_metric_artifacts.py`
- `reporting/artifact_reporter.py`
- `data/dataset_builder.py`
- `training/trainer.py`

## Пустые модули

Пакетные `__init__.py` оставлены пустыми намеренно.

Отдельные compatibility wrappers не удалялись, чтобы не ломать import paths:

- `app/cli.py`
- `app/web_app.py`
- `config/job_schema.py`
- `config/pipeline_config.py`
- `orchestration/job_executor.py`
- `orchestration/job_queue.py`
- `orchestration/resource_manager.py`
- `reporting/codex_summary.py`

Удаленных модулей в этом шаге нет. Пустые содержательные модули `scene_inference.py`, `probability_map.py`, `dataset_builder.py`, `trainer.py` были наполнены.

## Compatibility wrappers, которые остались

В `real_train.py` остались:

- S3 wrappers;
- scene matching wrappers;
- `_normalize_image`;
- `_loss_fn`;
- `_set_batchnorm_eval`;
- `_write_prediction_examples_report`;
- `_write_pseudolabel_outputs`;
- `run_debug_pseudolabel`;
- `run_real_train`.

`run_real_train()` и `run_debug_pseudolabel()` пока остаются в `real_train.py`, но pseudolabel часть внутри них уже идет через новый `run_pseudolabel_pipeline()`.

## Добавленные тесты во втором шаге

- `tests/test_probability_map.py`
- `tests/test_scene_inference.py`
- `tests/test_vectorization.py`
- `tests/test_pseudolabel_export.py`

Всего сейчас `unittest discover` запускает 31 тест.

## Проверки второго шага

Пройдены:

```powershell
python -m compileall mlsystem\src tests
python -m unittest discover -s tests
python -m mlsystem.src.debug.pseudolabel_debug --synthetic
```

Результат unittest:

```text
Ran 31 tests
OK
```

Synthetic smoke:

```json
{
  "ok": true,
  "coverage_fraction": 1.0,
  "min_weight_sum": 1.0,
  "accepted_objects": 1,
  "accepted_geojson": "results\\reports\\synthetic_pseudolabel_smoke\\synthetic_smoke.accepted.geojson"
}
```

## Что не менялось

- Job YAML format.
- CLI contracts.
- S3 layout.
- MLflow experiment names.
- Server/deploy.
- GitHub Actions.
- DVC.
- Geoalert.
- Heavy training/jobs.

## Оставшиеся риски

- `run_real_train()` и `run_debug_pseudolabel()` все еще живут в `real_train.py`.
- Dataset sampling и training loop еще не вынесены полностью.
- `job_executor.py` еще открывает MLflow run сам; следующий шаг должен вынести `RunReporter`/`MLflowRunManager`.
- Новый `SceneInferenceRunner` сейчас выполняет сцены последовательно; старые parallel config values сохраняются в отчетах, но parallel execution нужно вернуть отдельным безопасным PR с тестом.
- Production path теперь использует новый pseudolabel pipeline, но тяжелый real raster smoke не запускался по ограничению.

## Следующий шаг

1. Вынести dataset sampling `_read_samples()` и `_sample_windows()` в `data/dataset_builder.py`.
2. Вынести train loop из `run_real_train()` в `training/trainer.py`.
3. Вынести `run_debug_pseudolabel()` в `debug/pseudolabel_debug.py` без circular import.
4. Вынести MLflow lifecycle из `job_executor.py` в `reporting`.
5. Переключить CLI private imports с `real_train.py` на `storage.s3` и `data.scene_matching`.
