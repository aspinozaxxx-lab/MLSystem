# Модуль train

## Назначение

`mlsystem/src/train` отвечает только за обучение модели сегментации.

Модуль получает batch sources из `tile_preparation`, запускает обучение на CPU/GPU, владеет model architecture, loss, optimizer, scheduler, checkpoint и history. Результат работы - `TrainResult` с сырыми результатами обучения: history, diagnostics, checkpoint, best/last metrics и training summary. Сам `train` напрямую в MLflow не пишет и не знает, что будет записано в MLflow UI.

## Public API

```python
from mlsystem.src.train.api import list_supported_models, train_model
```

### `train_model(request: TrainRequest, progress_sink: TrainProgressSink | None = None) -> TrainResult`

- `request` - параметры обучения, dataloader источники для train/validation batch и директория для checkpoint/history.
- `progress_sink` - необязательный sink для событий прогресса обучения. Используется только для передачи `TrainProgressEvent`, не владеет lifecycle pipeline run.
- Возвращает `TrainResult` с сырыми результатами обучения: history, diagnostics, checkpoint, best/last metrics. Интерпретация этих данных для MLflow выполняется только `mlflow_adapter`.

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
- `train` не запускает pipeline lifecycle, не пишет run/status/stage файлы и не владеет orchestration. Это зона `train_pipeline`.
- `train` не импортирует и не вызывает MLflow, не принимает решений о params/metrics/artifacts MLflow и не возвращает готовые MLflow table metrics.
- Он возвращает training result. `train_pipeline` передает его в `mlflow_adapter`, а `mlflow_adapter` решает, что писать в MLflow metrics/params/artifacts.
- `train` не выполняет pseudolabeling, inference, vectorization и postprocess.
- `train` не импортирует FastAPI и `train_pipeline` internals.
