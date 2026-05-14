# Модуль подготовки тайлов

`mlsystem/src/tile_preparation/` - отдельный модуль подготовки training/validation тайлов для сегментации. Он не зависит от Airflow, FastAPI и `real_train.py`: входом являются пути к GeoTIFF, GeoJSON и txt scene list, выходом - ленивый `Dataset`/batch iterator с готовыми `image`, `mask` и lightweight metadata.

## Рекомендуемая точка входа

Используйте фасад:

```python
from pathlib import Path
from mlsystem.src.tile_preparation import TilePreparationFacade

bundle = TilePreparationFacade.from_scene_list(
    images_root=Path("E:/Projects/NSPD/Images/test"),
    scene_list_path=Path("E:/Projects/NSPD/Images/test/deforestation.txt"),
    annotation_path=Path("E:/Projects/NSPD/Images/test/deforestation.geojson"),
    tile_size=512,
    augmentation_level=1,
    mosaic_mode="auto",
    seed=42,
)

for indices, x, y in TilePreparationFacade.train_dataloader(bundle, batch_size=2):
    # x: [B, C, H, W], y: [B, 1, H, W]
    break
```

Публичный API пакета ограничен: `TilePreparationFacade`, `TilePreparationSimpleParams`, `TilePreparationBundle`, `TilePreparationConfig`, `SceneInput`, `AnnotationInput`, `TrainingTileDataset`, `ReadyTileSample`. Низкоуровневые функции из `windows`, `mask_rasterizer`, `validity`, `mosaic`, `iterator` считаются internal API и импортируются напрямую только в тестах/разработке.

## Высокоуровневые параметры

- `tile_size`: размер квадратного тайла.
- `stride`: шаг сетки; если не задан, считается как `tile_size * stride_ratio`.
- `stride_ratio`: fallback для `stride`.
- `augmentation_level`: агрегированный уровень аугментации и виртуального расширения train set.
- `mosaic_mode`: `off`, `auto`, `force`.
- `max_empty_tile_share`: явный override доли empty/hard-negative тайлов.
- `seed`: seed для split, shuffle и аугментаций.
- `input_bands`: список raster bands, если нужны не все каналы.

## augmentation_level

| level | augmentations | repeat factors | dense stride | max empty share |
|---|---|---|---|---|
| 0 | none | positive=1, hard_negative=1, negative=1 | all 1.0 | none |
| 1 | flips, rot90 | positive=2, hard_negative=1, negative=1 | positive=1.0, hard_negative=1.0, negative=1.0 | 0.5 |
| 2 | flips, rot90, brightness_contrast, gamma, noise, blur | positive=4, hard_negative=2, negative=1 | positive=0.5, hard_negative=0.5, negative=1.0 | 0.35 |
| 3 | all including cutout/coarse_dropout | positive=6, hard_negative=3, negative=1 | positive=0.25, hard_negative=0.5, negative=1.0 | 0.25 |

Если `max_empty_tile_share` передан явно, он перекрывает default уровня.

## Class-aware repeats

Виртуальные повторы не материализуют изображения на диск. Модуль хранит lightweight `TileSampleRecord` и повторяет индексы/records:

- `positive` и `partial_positive`: самый сильный repeat;
- `hard_negative`: умеренный repeat;
- `negative`: обычно без repeat.

Validation всегда строится отдельно: без repeat, без random augmentation, без dense train stride и без train-like oversampling.

## Valid data clipping

Для каждого tile читается `valid_data_mask`. Режимы: `auto`, `dataset_mask`, `alpha`, `nodata`, `nonzero_any`, `nonzero_all`. Для Kanopus с чёрным прямоугольным background важен fallback `nonzero_any`.

Training mask строится так:

1. GeoJSON растеризуется в window grid.
2. Если `clip_mask_to_valid_data=True`, mask умножается на `valid_data_mask`.
3. `positive_pixels` и классификация считаются после clipping.

На чёрной области без данных training mask должна быть равна 0.

## Mosaic fill

Если `mosaic_mode=auto` и в split есть несколько сцен, либо `mosaic_mode=force`, invalid pixels anchor-сцены могут заполняться соседними снимками. Маска после этого клипится уже по union valid mask. Если сосед не покрывает область, пиксели остаются invalid, а mask там зануляется.

## Cutout/dropout

Default production behavior: `cutout_mask_mode="erase"`.

Если cutout или coarse dropout попадает внутрь объекта:

- image pixels зануляются;
- соответствующие пиксели binary mask тоже зануляются;
- metadata пишет `mask_erased_pixels`, `cutout_intersected_positive`, `cutout_boxes`.

Это нужно, чтобы чёрный квадрат внутри объекта не считался объектом. В HTML-отчёте красный пунктирный контур после cutout показывает и внешнюю границу объекта, и внутреннюю границу вырезанной дырки.

## Training loop

Обучение использует Python `TrainingTileDataset`/batch iterator напрямую. FastAPI не участвует в training path и нужен только для lightweight debug preview.

## Debug

Annotated report:

```powershell
python scripts\debug_annotated_tile_report.py `
  --input-dir E:\Projects\NSPD\Images\test `
  --cases single,two_scene `
  --anchor-scene KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN03.tif `
  --annotation deforestation.geojson `
  --mosaic-enabled `
  --tile-size 768 `
  --stride 512 `
  --augmentation-mode all `
  --cutout-mask-mode erase
```

Train smoke:

```powershell
python scripts\debug_tile_preparation_train_smoke.py `
  --images-root E:\Projects\NSPD\Images\test `
  --scene-list E:\Projects\NSPD\Images\test\deforestation.txt `
  --annotation E:\Projects\NSPD\Images\test\deforestation.geojson `
  --tile-size 512 `
  --augmentation-level 1 `
  --mosaic-mode auto `
  --batch-size 2 `
  --max-train-batches 2 `
  --max-val-batches 1 `
  --device cpu
```

Debug outputs и реальные GeoTIFF/GeoJSON не коммитятся.

## Правила разработки

- Новый пользовательский entrypoint сначала добавляется в `TilePreparationFacade`.
- Низкоуровневые функции можно тестировать напрямую, но не продвигать как публичный API.
- Training и validation должны использовать один source of truth для window grid, rasterization, valid clipping, mosaic fill и augmentation semantics.
- FastAPI endpoint не должен возвращать raster/mask arrays.
