# Модуль tile_preparation

## Назначение

`mlsystem/src/tile_preparation` владеет подготовкой training/validation tile samples из raster и vector annotation. Создан для быстрой выдачи тайлов на обучение, реализуя torch.dataloader и torch.dataset . Проводит аугментацию заданного урвовня. Параллелит обработку для утулизации ресурсов.

## Публичный фасад
```python
from mlsystem.src.tile_preparation import TilePreparationFacade
```
Public API:
- `TilePreparationFacade.build_datasets(...)`;
- `TilePreparationFacade.train_dataloader(...)`;
- `TilePreparationFacade.val_dataloader(...)`;

## Вход
- train scenes: список `SceneInput`;
- val scenes: список `SceneInput`;
- annotation path;
- `tile_size`;
- `stride`;
- augmentation level;
- normalization mode;
- batch size для DataLoader.

DataLoader workers, prefetch, pin memory и persistent worker policy являются внутренними defaults модуля. Они не должны задаваться как параметры experiment trace. Расширение списка параметров доступных из вне должно быть отдельно согласовано.

## Выход

DataLoader возвращает batch:

```python
indices: list[int]
x: torch.Tensor  # [B,C,H,W]
y: torch.Tensor  # [B,1,H,W]
```

## Модуль отвечает за:
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

## Validation всегда:
- fixed grid;
- no augmentation;
- no random jitter;
- no virtual repeats;
- no train-like oversampling.