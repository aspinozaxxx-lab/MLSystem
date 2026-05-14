# Virtual Tile Sampling Debug

Основная инструкция по текущему модулю находится в [`docs/tile_preparation_module_ru.md`](tile_preparation_module_ru.md). Этот документ оставлен как краткая справка по debug workflow и совместимости старой идеи virtual sampling.

## Текущая модель

`tile_preparation` больше не делает train/val split. Split создаёт стадия `prepare_dataset`, а training получает готовые `train_scenes` и `val_scenes`.

Единственная рекомендуемая точка входа:

```python
from pathlib import Path
from mlsystem.src.tile_preparation import SceneInput, TilePreparationFacade

bundle = TilePreparationFacade.build_datasets(
    train_scenes=[SceneInput(Path("train_scene.tif"), "train_scene")],
    val_scenes=[SceneInput(Path("val_scene.tif"), "val_scene")],
    annotation_path=Path("deforestation.geojson"),
    tile_size=768,
    stride=512,
    augmentation_level=2,
)
```

Публичные параметры фасада: `tile_size`, `stride`, `augmentation_level`. Параметры `from_scene_list`, `train_fraction`, `stride_ratio`, `mosaic_mode`, `seed`, `input_bands`, `max_empty_tile_share` не являются публичным API фасада.

## augmentation_level

`augmentation_level` агрегирует dense stride, repeats, empty limiting и train augmentation:

| level | смысл |
|---|---|
| 0 | без train augmentation и без virtual expansion |
| 1 | лёгкий режим: flips/rot90, небольшой repeat positive tiles |
| 2 | основной режим для маленького датасета: photometric/noise/blur, dense positive/hard-negative grid, class-aware repeats |
| 3 | агрессивный режим: более плотные positive windows, cutout/coarse_dropout, stronger repeats |

Validation всегда строится честно: fixed grid, без random augmentation, без repeats, без dense train stride.

## Valid clipping и mosaic

Training mask всегда клипится по valid data mask. Полностью невалидные тайлы не попадают в records. Если в split больше одной сцены, mosaic включается автоматически внутри этого split:

- train использует только `train_scenes`;
- val использует только `val_scenes`;
- train и val не используются друг для друга как mosaic neighbors.

FastAPI не используется в training path.

## Annotated report

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

Отчёт показывает:

- красный пунктирный contour clipped training mask;
- valid/invalid raster diagnostics;
- mosaic fill diagnostics;
- cutout/dropout holes with mask erase;
- augmentation checks for image+mask consistency.

## Train smoke

Smoke script читает txt списки только как debug tool. Это не часть публичного API `tile_preparation`.

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

Если локально есть только общий txt, можно использовать debug-only fallback:

```powershell
python scripts\debug_tile_preparation_train_smoke.py `
  --images-root E:\Projects\NSPD\Images\test `
  --scene-list E:\Projects\NSPD\Images\test\deforestation.txt `
  --debug-split-first-n-train 1 `
  --annotation E:\Projects\NSPD\Images\test\deforestation.geojson `
  --tile-size 512 `
  --stride 512 `
  --augmentation-level 1 `
  --batch-size 2 `
  --max-train-batches 2 `
  --max-val-batches 1 `
  --device cpu
```

## FastAPI debug

FastAPI endpoints are debug-only and registered only with:

```powershell
$env:MLSYSTEM_DEBUG_DATASET_ENDPOINTS="1"
python -m uvicorn mlsystem.src.api.app:app --host 127.0.0.1 --port 8088
```

Endpoints must return lightweight summaries only: no raster arrays, no mask arrays, no training tiles. Training calls `TilePreparationFacade`/`TrainingTileDataset` directly.

## Git hygiene

Do not commit GeoTIFF, GeoJSON, generated HTML/PNG/JSON reports, `outputs/`, or local debug folders.
