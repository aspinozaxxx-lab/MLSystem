# Модуль tile_preparation

## Назначение

`mlsystem/src/tile_preparation` владеет подготовкой training/validation tile samples из raster и vector annotation.

Training loop не должен знать, как читать raster, rasterize mask, clip valid area, делать mosaic, augmentation или normalization.

## Публичный фасад

Основная точка входа:

```python
from mlsystem.src.tile_preparation import TilePreparationFacade
```

Public API:

- `TilePreparationFacade.build_datasets(...)`;
- `TilePreparationFacade.train_dataloader(...)`;
- `TilePreparationFacade.val_dataloader(...)`;
- compatibility wrapper для старого sync iterator, если нужен старому коду.

Расширение public API требует согласования.

## Вход

- train scenes: список `SceneInput`;
- val scenes: список `SceneInput`;
- annotation path;
- `tile_size`;
- `stride`;
- augmentation level;
- normalization mode;
- DataLoader параметры workers/prefetch/pin/persistent.

## Выход

DataLoader возвращает batch:

```python
indices: list[int]
x: torch.Tensor  # [B,C,H,W]
y: torch.Tensor  # [B,1,H,W]
```

## Внутреннее поведение

Модуль отвечает за:

- построение scene records;
- virtual train records;
- fixed validation grid;
- lazy rasterio open per worker;
- raster window read;
- valid mask read and clipping;
- mask rasterization;
- geometry STRtree index and fallback filtering;
- mosaic fill;
- train augmentation;
- normalization;
- DataLoader collate;
- lightweight profiling.

## Запрещено выносить наружу

- raster window read в `real_train.py`;
- mask rasterization в training loop;
- valid clipping в training loop;
- mosaic в training loop;
- augmentation в training loop;
- MLflow logging;
- FastAPI routes;
- train/val split logic;
- pipeline business decisions.

## Validation

Validation всегда:

- fixed grid;
- no augmentation;
- no random jitter;
- no virtual repeats;
- no train-like oversampling.

## Пример вызова

```python
bundle = TilePreparationFacade.build_datasets(...)
train_loader = TilePreparationFacade.train_dataloader(bundle, batch_size=4, workers=4)
val_loader = TilePreparationFacade.val_dataloader(bundle, batch_size=4, workers=4)
```
