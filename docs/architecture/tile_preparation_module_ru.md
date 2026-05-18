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

Расширение public API требует согласования.

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

## Внутреннее поведение

Модуль отвечает за:

Быструю выдачу тайлов на обучение, реализуя torch.dataloader и torch.dataset . Проводит аугментацию заданного урвовня. Параллелит обработку для утулизации ресурсов.
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
train_loader = TilePreparationFacade.train_dataloader(bundle, batch_size=4)
val_loader = TilePreparationFacade.val_dataloader(bundle, batch_size=4)
```

По умолчанию на Linux `tile_preparation` использует 16 DataLoader workers, `prefetch_factor=2`, `pin_memory=True`.
На Windows default workers падает до 0, чтобы не ломать локальную отладку multiprocessing.
Для аварийной диагностики есть env override `MLSYSTEM_TILE_DATALOADER_WORKERS` и `MLSYSTEM_TILE_DATALOADER_PREFETCH_FACTOR`, но это не публичные параметры эксперимента.

## Локальный profiler

`scripts/profile_tile_training_local.py` использует тот же `TrainingTileDataset`, но включает debug collate с metadata, чтобы увидеть внутренние тайминги даже при `workers > 0`.
Это не расширяет публичный фасад: production DataLoader по-прежнему возвращает `indices, x, y`.

Per-sample timings живут в `ReadyTileSample.metadata["tile_prep_profile"]` только при `MLSYSTEM_TILE_PREP_PROFILE=1`.
Основные поля: `read_valid_mask_sec`, `read_image_sec`, `normalize_sec`, `geometry_index_query_sec`, `rasterize_sec`, `augmentation_sec`, `mosaic_sec`, `total_getitem_sec`.

## SceneFootprint и grid по покрытию

`tile_preparation` строит internal `SceneFootprint` один раз на сцену. Footprint описывает фактическое покрытие raster в pixel coordinates и raster CRS.

Источники footprint:

- `dataset_mask`;
- alpha band;
- nodata/read masks;
- fallback `nonzero_any` для снимков с черным фоном.

Если mask source не дает полезный контур, модуль переходит к `nonzero_any`. Для больших raster footprint читается как downsample/overview и затем переводится обратно в pixel coordinates.

Window grid остается регулярным, но records создаются только для окон, которые пересекают `SceneFootprint.polygon_pixel`.
Окна полностью вне footprint не создаются и не доходят до `TrainingTileDataset`.

Boundary tile сохраняется, если пересекает footprint. Для такого окна valid mask строится rasterization footprint polygon в grid окна. Raster pixels для valid mask не читаются.
Fully-inside tile считается полностью валидным и не читает valid mask.

Per-window `read_valid_data_mask_with_source` в build records является только fallback для старых records без footprint metadata.

Scene reports и metadata содержат:

- `candidate_windows_rectangular`;
- `windows_intersecting_footprint`;
- `skipped_outside_footprint`;
- `fully_inside_footprint_windows`;
- `boundary_footprint_windows`;
- `build_footprint_sec`;
- `build_records_sec`;
- `read_valid_mask_calls`.

Параллельный build scene records включается внутри модуля через `MLSYSTEM_TILE_RECORD_WORKERS`.
На Windows default последовательный; на Linux default ограничен количеством сцен и CPU, максимум 16.

## SceneAdjacency и TileMosaicPlan

`tile_preparation` заранее строит internal `SceneAdjacencyIndex` по `SceneFootprint` всех сцен. Индекс описывает только геометрически полезных соседей:

- `overlap` - footprint соседней сцены пересекает anchor footprint;
- `touch` - footprint касается anchor в пределах допуска;
- `near` - зарезервировано для малого расстояния, если оно включено;
- `disjoint` наружу не попадает и не становится candidate для mosaic.

Для каждого record создается internal `TileMosaicPlan`. Он не расширяет публичный фасад, а записывает служебные поля в `TileSampleRecord.metadata`:

- `mosaic_needed`;
- `mosaic_reason`;
- `mosaic_side`;
- `mosaic_candidate_scene_ids`;
- `mosaic_estimated_gap_share`;
- `mosaic_estimated_neighbor_cover_share`;
- `mosaic_overlap_record`.

Если tile полностью внутри footprint anchor-сцены, mosaic не вызывается. Если boundary tile имеет gap, но сосед не покрывает этот gap геометрически, mosaic также не вызывается. `read_mosaic_window` вызывается только для records, где `TileMosaicPlan.needed=True`, и получает только candidate neighbors из плана.

Overlap policy намеренно простая: neighbor используется только для пикселей, где anchor invalid. Valid-valid blending не реализован, чтобы не менять семантику данных и не тратить время на ненужное смешивание снимков.

Mosaic counters доступны в build metadata, scene reports и sample metadata:

- `mosaic_attempted`;
- `mosaic_skipped_fully_inside`;
- `mosaic_skipped_no_gap`;
- `mosaic_skipped_no_candidates`;
- `mosaic_candidate_neighbors_total`;
- `mosaic_intersecting_neighbors_total`;
- `mosaic_warped_vrt_calls`;
- `mosaic_used_neighbors`;
- `mosaic_filled_pixels`;
- `mosaic_zero_fill_attempts`;
- `mosaic_boundary_records`;
- `mosaic_overlap_records`.
