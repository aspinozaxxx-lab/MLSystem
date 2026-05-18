# Модуль inference_pipeline

## Назначение

`mlsystem/src/inference_pipeline` - единственный оркестратор создания псевдоразметки.

Модуль создает pseudolabel run, хранит trace, управляет worker/status/log/progress/cancel/final summary и вызывает InferenceEngine для реальной inference-работы. Он не рефакторит InferenceEngine и не копирует его tiling, stitching, vectorization или postprocess.

## Public API

```python
from mlsystem.src.inference_pipeline.api import (
    InferencePipelineRunner,
    PseudolabelRun,
    PseudolabelRunRequest,
    cancel_pseudolabel_run,
    get_pseudolabel_run,
    start_pseudolabel_run,
    tail_pseudolabel_log,
)
```

### `PseudolabelRunRequest`

- `experiment_id: str` - человекочитаемый id эксперимента.
- `model_ref: str` - ссылка на модель для InferenceEngine.
- `images_uri: str` - URI набора снимков.
- `class_name: str | None = None` - имя класса объектов.
- `run_id: str | None = None` - необязательный id run.
- `scenes: list[str] | None = None` - необязательный список сцен.
- `layout_uri: str | None = None` - URI layout/annotation context.
- `dry_run: bool = False` - создать run и payload без отправки job в InferenceEngine.

### `start_pseudolabel_run(request: PseudolabelRunRequest) -> PseudolabelRun`

- Создает run directory.
- Сохраняет request/trace.
- Запускает worker process.
- Возвращает текущий `PseudolabelRun`.

### `get_pseudolabel_run(run_id: str) -> PseudolabelRun`

- Возвращает текущий статус run.

### `cancel_pseudolabel_run(run_id: str) -> PseudolabelRun`

- Запрашивает отмену worker/run и возвращает обновленный статус.

### `tail_pseudolabel_log(run_id: str, max_chars: int = 20000) -> str`

- Возвращает хвост pseudolabel log.

### `InferencePipelineRunner`

- Public runner для CLI/API/integration-кода.
- Использует только contracts и внутренние реализации `inference_pipeline`.

## Lifecycle

1. API/CLI принимает `PseudolabelRunRequest`.
2. `inference_pipeline` создает run directory.
3. Модуль сохраняет request и trace.
4. Runner запускает worker process.
5. Worker строит payload для InferenceEngine.
6. Worker отправляет job в InferenceEngine.
7. Worker polling-ом ждет завершения InferenceEngine job.
8. Worker валидирует artifact payload.
9. Worker пишет status, logs, progress и final summary.
10. Worker логирует metadata/artifacts в MLflow только через `mlflow_adapter`.

## Вход

- `PseudolabelRunRequest`.
- InferenceEngine HTTP/API boundary.
- Тонкие настройки tile size, stride, thresholds, vectorization params, postprocess params, polling interval, timeout, compatibility artifact names и retry policy являются внутренними defaults модуля.

## Выход

- `PseudolabelRun`.
- Run directory с `request.json`, `trace.json`, `status.json`, `summary.json`, `logs/pseudolabel.log`.
- InferenceEngine payload и artifact summary.

## MLflow logging

`inference_pipeline` не импортирует `mlflow` напрямую. Metadata и artifacts логируются только через `mlflow_adapter`.

## Граница с InferenceEngine

`inference_pipeline` использует InferenceEngine как внешний исполнитель через API/client. InferenceEngine владеет inference tiling/windows, probability map/stitching, vectorization, postprocess и pseudolabel export.

## Запрещенные пересечения

- Не выполнять tiling/vectorization/postprocess внутри `inference_pipeline`.
- Не копировать implementation-код InferenceEngine.
- Не импортировать FastAPI.
- Не импортировать `train` internals.
- Не импортировать `train_pipeline` internals.
- Не импортировать `mlflow` напрямую.
- Не оставлять compatibility/deprecated wrappers.
