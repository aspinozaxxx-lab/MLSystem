# Модуль train_pipeline

## Назначение

`mlsystem/src/train_pipeline` - orchestration модуль для training pipeline run.

Он отвечает за создание run, хранение trace, запуск worker process, lifecycle стадий обучения, status/log/progress/cancel/final summary и MLflow metadata logging через `mlflow_adapter`.

## Public API

```python
from mlsystem.src.train_pipeline.api import (
    PipelineRunConfig,
    TrainPipelineRunStore,
    TrainPipelineRunner,
    cancel_train_pipeline_run,
    get_train_pipeline_run,
    start_train_pipeline_run,
    tail_train_pipeline_log,
)
```

### `PipelineRunConfig`

- Public DTO конфигурации training pipeline.
- Содержит `experiment_id`, `run_id`, `pipeline.stages`, dataset/model/train/evaluate/mlflow секции.

### `TrainPipelineRunner`

- Public runner training pipeline.
- Создает worker process и обновляет run lifecycle через `TrainPipelineRunStore`.

### `TrainPipelineRunStore`

- Public store для статуса training pipeline run, stage reports, summary и logs.

### `start_train_pipeline_run(config: PipelineRunConfig | dict, source: str = "api", run_root: str | Path | None = None) -> PipelineRun`

- `config` - trace/config training pipeline.
- `source` - источник запуска.
- `run_root` - необязательный root для run store.
- Возвращает созданный/запущенный `PipelineRun`.

### `get_train_pipeline_run(run_id: str, run_root: str | Path | None = None) -> PipelineRun`

- Возвращает актуальный статус run.

### `cancel_train_pipeline_run(run_id: str, run_root: str | Path | None = None) -> PipelineRun`

- Запрашивает отмену run.

### `tail_train_pipeline_log(run_id: str, max_chars: int = 20000, run_root: str | Path | None = None) -> str`

- Возвращает хвост training pipeline log.

## CLI

```bash
python -m mlsystem.src.train_pipeline.cli
```

## Lifecycle

1. API/CLI валидирует trace.
2. `TrainPipelineRunStore.create_run` создает `/data/mlsystem/runs/<run_id>/`.
3. `TrainPipelineRunner.start_run` запускает worker process.
4. Worker последовательно выполняет training stages.
5. Stage `train_model` вызывает только `mlsystem.src.train.api.train_model`.
6. Pipeline записывает JSON/report/log/status.
7. Pipeline логирует MLflow metadata/artifacts через `mlflow_adapter`.
8. Финальный статус и summary фиксируются после завершения.

## Границы

- `train_pipeline` не обучает модель сам, а вызывает `train.api`.
- `train_pipeline` не пишет в MLflow напрямую, а использует `mlflow_adapter`.
- `train_pipeline` не создает псевдоразметку. Это зона `inference_pipeline`.
- `train_pipeline` не копирует InferenceEngine logic.
- `train_pipeline` не является deprecated wrapper вокруг старого имени модуля.
## Dataset metadata

После `prepare_dataset` и до/во время `train_model` pipeline использует `TrainingDatasetPrepareResult.dataset_identity` и логирует dataset input в MLflow через `mlflow_adapter.log_dataset_input_to_run(...)`. Adapter также сохраняет параметры `dataset.version`, `dataset.version_source`, `dataset.fingerprint`, `dataset.git_commit`, `dataset.git_commit_date`, `dataset.class_name`, `dataset.class_slug`, `dataset.objects`, `dataset.scenes`, `dataset.selected_scenes`, `dataset.train_scenes`, `dataset.val_scenes`, `dataset.split_strategy`, `dataset.annotation_uri`, `dataset.scenes_uri`, `dataset.images_uri`, `dataset.layout_uri`.

Pipeline также логирует dataset artifacts, если они существуют: `dataset_manifest.json`, `inventory_scenes.json`, `scene_matching_report.json`, `split_summary.json`.

## MLflow metrics

Основная таблица модели использует только:
- `model_metrics/f1_pixel`;
- `model_metrics/epochs_total`;
- `model_metrics/epoch_time_sec`;
- `model_metrics/training_time_sec`.

Diagnostics metrics логируются только из whitelist `diagnostics/*`; pixel counters, threshold sweeps and debug/microtiming payloads остаются artifacts.
