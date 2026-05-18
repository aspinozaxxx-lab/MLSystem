# Модуль tile_preparation

## Назначение

`mlsystem/src/tile_preparation` владеет подготовкой training/validation tile datasets из raster и vector annotation. Модуль строит tile records, torch datasets, torch dataloaders, fixed validation grid и training augmentation.

## Public API

```python
from mlsystem.src.tile_preparation.api import build_datasets, train_dataloader, val_dataloader
```

### `build_datasets(request: TileDatasetRequest) -> TilePreparationBundle`

- `request.train_scenes` - список training scenes.
- `request.val_scenes` - список validation scenes.
- `request.annotation_path` - путь к annotation GeoJSON.
- `request.tile_size` - размер tile.
- `request.stride` - шаг fixed grid.
- `request.augmentation_level` - уровень train augmentation.
- `request.mosaic_enabled` - включает mosaic fill, если задано.
- `request.normalization_mode` - режим нормализации raster batch.

### `train_dataloader(bundle: TilePreparationBundle, *, batch_size: int, **kwargs) -> Any`

- `bundle` - результат `build_datasets`.
- `batch_size` - размер batch.
- `kwargs` - технические параметры DataLoader.

### `val_dataloader(bundle: TilePreparationBundle, *, batch_size: int, **kwargs) -> Any`

- `bundle` - результат `build_datasets`.
- `batch_size` - размер batch.
- `kwargs` - технические параметры DataLoader.

## Выход batch

DataLoader возвращает:

```python
indices: list[int]
x: torch.Tensor  # [B,C,H,W]
y: torch.Tensor  # [B,1,H,W]
```

## Запрещенные пересечения

- `tile_preparation` не обучает модель.
- `tile_preparation` не пишет в MLflow.
- `tile_preparation` не владеет pipeline lifecycle.
- `tile_preparation` не выполняет inference, pseudolabeling, vectorization или postprocess.
- Validation всегда fixed grid без augmentation, jitter, virtual repeats и train-like oversampling.
