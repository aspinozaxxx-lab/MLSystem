# Модуль train

## Назначение

`mlsystem/src/train` отвечает только за обучение модели сегментации.

Модуль получает batch sources из `tile_preparation`, запускает обучение на CPU/GPU, владеет model architecture, loss, optimizer, scheduler, checkpoint и history. Результат работы - `TrainResult` и payload для MLflow. Сам `train` напрямую в MLflow не пишет.

## Public API

```python
from mlsystem.src.train.api import list_supported_models, train_model
```

### `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult`

- `request` - параметры обучения, dataloader источники для train/validation batch и директория для checkpoint/history.
- `progress_sink` - необязательный sink для событий прогресса обучения. Используется только для передачи `TrainProgressEvent`, не владеет lifecycle pipeline run.
- Возвращает `TrainResult` с метриками, checkpoint artifact, history и `mlflow_params`/`mlflow_metrics`/`mlflow_artifacts` для внешнего логирования.

### `list_supported_models() -> list[ModelSpec]`

- Возвращает список поддерживаемых архитектур моделей.

## Контракты

`contracts.py` содержит только:

- `TrainRequest`;
- `TrainResult`;
- `TrainConfig`;
- `ModelSpec`;
- `CheckpointArtifact`;
- `EpochMetrics`;
- `TrainProgressEvent`;
- `TrainProgressSink`;
- `TrainError`.

## Запрещенные пересечения

- `train` не подготавливает dataset input и split. Это зона `dataset_preparing`.
- `train` не создает tile datasets, dataloaders и augmentation. Это зона `tile_preparation`.
- `train` не запускает pipeline lifecycle, не пишет run/status/stage файлы и не владеет orchestration. Это зона `pipeline_runner`.
- `train` не импортирует и не вызывает MLflow. Он возвращает payload, который логирует `pipeline_runner` через `mlflow_adapter`.
- `train` не выполняет pseudolabeling, inference, vectorization и postprocess.
- `train` не импортирует FastAPI и pipeline runner internals.
