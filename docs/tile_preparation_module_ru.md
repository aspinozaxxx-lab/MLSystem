# Модуль подготовки тайлов

`mlsystem/src/tile_preparation/` - отдельный модуль подготовки training/validation тайлов для сегментации. Он не зависит от orchestration layer, FastAPI и внутренних деталей `real_train.py`: входом являются уже разделённые списки сцен и путь к GeoJSON, выходом - ленивые `Dataset`/batch iterator с готовыми `image`, `mask` и lightweight metadata.

## Главное правило

Модуль не делает train/val split. Split делает стадия `prepare_dataset`, а `tile_preparation` получает уже готовые `train_scenes` и `val_scenes`.

## Рекомендуемая точка входа

```python
from pathlib import Path
from mlsystem.src.tile_preparation import SceneInput, TilePreparationFacade

bundle = TilePreparationFacade.build_datasets(
    train_scenes=[
        SceneInput(image_path=Path("E:/Projects/NSPD/Images/test/train_scene.tif"), scene_id="train_scene"),
    ],
    val_scenes=[
        SceneInput(image_path=Path("E:/Projects/NSPD/Images/test/val_scene.tif"), scene_id="val_scene"),
    ],
    annotation_path=Path("E:/Projects/NSPD/Images/test/deforestation.geojson"),
    tile_size=512,
    stride=512,
    augmentation_level=1,
)

for indices, x, y in TilePreparationFacade.train_dataloader(bundle, batch_size=2):
    # x: torch.Tensor [B, C, H, W]
    # y: torch.Tensor [B, 1, H, W], значения 0/1
    break
```

Публичный API пакета:

- `TilePreparationFacade`;
- `TilePreparationBundle`;
- `TilePreparationConfig`;
- `SceneInput`;
- `AnnotationInput`;
- `TrainingTileDataset`;
- `ReadyTileSample`.

Низкоуровневые функции из `windows`, `mask_rasterizer`, `validity`, `mosaic`, `iterator` считаются internal API. Их можно импортировать напрямую в тестах и при разработке, но не использовать как пользовательский entrypoint.

## Публичные параметры фасада

`TilePreparationFacade.default_config` принимает только:

- `tile_size`;
- `stride`;
- `augmentation_level`.

`TilePreparationFacade.build_datasets` принимает только:

- `train_scenes`;
- `val_scenes`;
- `annotation_path`;
- `tile_size`;
- `stride`;
- `augmentation_level`.

Удалённые из публичного API параметры: `from_scene_list`, `train_fraction`, `stride_ratio`, `mosaic_mode`, `seed`, `input_bands`, `max_empty_tile_share`. Они не должны управляться снаружи фасада.

## Внутренние defaults

В `facade.py` закреплены внутренние значения:

- `DEFAULT_SEED = 42`;
- `DEFAULT_MOSAIC_MODE = "auto"`;
- `DEFAULT_ANNOTATION_CRS = "auto"`;
- `DEFAULT_ALLOW_INFERRED_ANNOTATION_CRS = True`;
- `DEFAULT_INPUT_BANDS = None`.

`stride` всегда задаётся явно. `DEFAULT_INPUT_BANDS = None` означает чтение всех доступных raster bands.

## CRS и GeoJSON

GeoJSON загружается как единый источник разметки, но геометрии приводятся к CRS каждого raster отдельно. В `scene_reports` и JSON-отчётах для каждой сцены пишутся:

- `raster_crs`;
- `annotation_crs`;
- `annotation_crs_source`;
- `transformed_to_raster_crs`.

Если GeoJSON не содержит CRS и inference выключен на low-level API, модуль падает с понятной ошибкой. Фасад использует debug-friendly `annotation_crs="auto"` и `allow_inferred_annotation_crs=True`; inference сопровождается warning.

## augmentation_level

`augmentation_level` управляет и аугментациями, и виртуальным расширением train set.

| level | augmentations | repeat factors | dense stride | max empty share |
|---|---|---|---|---|
| 0 | none | positive=1, hard_negative=1, negative=1 | all 1.0 | none |
| 1 | flips, rot90 | positive=2, hard_negative=1, negative=1 | all 1.0 | 0.5 |
| 2 | flips, rot90, brightness_contrast, gamma, noise, blur | positive=4, hard_negative=2, negative=1 | positive=0.5, hard_negative=0.5, negative=1.0 | 0.35 |
| 3 | all including cutout/coarse_dropout | positive=6, hard_negative=3, negative=1 | positive=0.25, hard_negative=0.5, negative=1.0 | 0.25 |

## Class-aware repeats

Виртуальные повторы не материализуют изображения на диск. Модуль хранит lightweight `TileSampleRecord` и повторяет records:

- `positive` и `partial_positive`: самый сильный repeat;
- `hard_negative`: умеренный repeat;
- `negative`: обычно без repeat.

Validation строится отдельно: без repeat, без random augmentation, без dense train stride и без train-like oversampling.

## Valid data clipping

Для каждого tile читается `valid_data_mask`. Режимы internal config: `auto`, `dataset_mask`, `alpha`, `nodata`, `nonzero_any`, `nonzero_all`. Для Kanopus с чёрным прямоугольным background важен fallback `nonzero_any`.

Training mask строится так:

1. GeoJSON растеризуется в window grid.
2. Если `clip_mask_to_valid_data=True`, mask умножается на `valid_data_mask`.
3. `positive_pixels` и классификация считаются после clipping.

На чёрной области без данных training mask равна 0. Полностью невалидные тайлы не попадают в records никогда, если `drop_fully_invalid_tiles=True` (default). В summary пишутся `skipped_fully_invalid_tiles`, `skipped_low_valid_share_tiles`, `min_valid_pixel_share`.

## Mosaic fill

Mosaic работает автоматически и только внутри своего split:

- train может использовать соседей только из `train_scenes`;
- val может использовать соседей только из `val_scenes`;
- train никогда не использует val-сцены как mosaic neighbors;
- val никогда не использует train-сцены как mosaic neighbors.

Если в split больше одной сцены, invalid pixels anchor-сцены могут заполняться соседними снимками. Маска после этого клипится по union valid mask. Если сосед не покрывает область, пиксели остаются invalid, а mask там зануляется.

Перед чтением соседей используется footprint index: модуль проверяет пересечение bounds tile и bounds соседнего raster. Для разных CRS bounds соседей приводятся через `transform_bounds`. В mosaic summary пишутся `candidate_neighbors`, `intersecting_neighbors`, `actually_used_neighbors`, `skipped_non_intersecting_neighbors`.

## Cutout/dropout

Internal default: `cutout_mask_mode="erase"`.

Если cutout или coarse dropout попадает внутрь объекта:

- image pixels зануляются;
- соответствующие пиксели binary mask тоже зануляются;
- metadata пишет `mask_erased_pixels`, `cutout_intersected_positive`, `cutout_boxes`.

Это нужно, чтобы чёрный квадрат внутри объекта не считался объектом. В HTML-отчёте красный пунктирный контур после cutout показывает и внешнюю границу объекта, и внутреннюю границу вырезанной области.

## Training loop

Обучение использует Python `TrainingTileDataset`/batch iterator напрямую. FastAPI не участвует в training path и нужен только для lightweight debug preview.

`ReadyTileSample` содержит:

- `sample.image`: `np.ndarray [C, H, W]`, `float32`;
- `sample.mask`: `np.ndarray [1, H, W]`, `float32`, значения 0/1;
- `sample.record`: lightweight `TileSampleRecord`;
- `sample.metadata`: valid clipping, mosaic и augmentation metadata.

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

Train smoke с готовыми split lists:

```powershell
python scripts\debug_tile_preparation_train_smoke.py `
  --images-root E:\Projects\NSPD\Images\test `
  --train-scene-list E:\Projects\NSPD\Images\test\train.txt `
  --val-scene-list E:\Projects\NSPD\Images\test\val.txt `
  --annotation E:\Projects\NSPD\Images\test\deforestation.geojson `
  --tile-size 512 `
  --stride 512 `
  --augmentation-level 1 `
  --batch-size 2 `
  --max-train-batches 2 `
  --max-val-batches 1 `
  --device cpu
```

Если локально есть только общий txt, script поддерживает debug-only fallback `--scene-list ... --debug-split-first-n-train ...`. Это split внутри debug script, не внутри `tile_preparation`.

Debug outputs и реальные GeoTIFF/GeoJSON не коммитятся.

## Правила разработки

- Новый пользовательский entrypoint сначала добавляется в `TilePreparationFacade`.
- Низкоуровневые функции можно тестировать напрямую, но не продвигать как публичный API.
- Training и validation должны использовать один source of truth для window grid, rasterization, valid clipping, mosaic fill и augmentation semantics.
- FastAPI endpoint не должен возвращать raster/mask arrays.
# DataLoader and normalization update

Recommended training entrypoint is `TilePreparationFacade.train_dataloader(...)` / `val_dataloader(...)`.
These methods return `torch.utils.data.DataLoader` and accept `workers`, `prefetch_factor`, `pin_memory`, `persistent_workers`.
On Windows the default `workers=None` resolves to `0`. On Linux it resolves to `min(8, os.cpu_count() // 2)`.
The returned batch format stays compatible with the old iterator: `indices: list[int]`, `x: torch.Tensor [B,C,H,W]`, `y: torch.Tensor [B,1,H,W]`.
`iter_dataset_batches()` remains as a compatibility fallback, but production training should use the facade DataLoader path.

`TilePreparationConfig.normalization_mode` controls image normalization:

- `uint8_255` - fast path, `float32(arr) / 255.0`;
- `tile_percentile` - legacy per-tile percentile normalization;
- `scene_percentile` - approximate per-scene percentile stats cached per worker.

Default is `uint8_255`.
