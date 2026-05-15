# Inventory tile/preprocessing code

## tile_preparation

| Файл | Что делает | Training | Inference | Решение |
|---|---|---:|---:|---|
| `tile_preparation/facade.py` | Public facade для build datasets и DataLoader | да | возможно позже | keep |
| `tile_preparation/dataset.py` | `TrainingTileDataset`, lazy rasterio, records, getitem | да | нет | keep |
| `tile_preparation/dataloader.py` | DataLoader/collate/workers defaults | да | нет | keep |
| `tile_preparation/raster_reader.py` | raster window read, normalization | да | потенциально | keep |
| `tile_preparation/mask_rasterizer.py` | rasterize annotation mask | да | нет | keep |
| `tile_preparation/validity.py` | valid data mask and clipping | да | потенциально | keep |
| `tile_preparation/mosaic.py` | mosaic fill for train tiles | да | потенциально | keep |
| `tile_preparation/augmentations.py` | train augmentation | да | нет | keep |
| `tile_preparation/geometry_index.py` | STRtree/fallback geometry query | да | потенциально | keep |
| `tile_preparation/windows.py` | tile grid helpers | да | потенциально | keep |
| `tile_preparation/records.py` | train/val tile records | да | нет | keep |
| `tile_preparation/profiling.py` | aggregate tile prep timings | да | нет | keep |
| `tile_preparation/report.py` | debug/report support | да | нет | keep |
| `tile_preparation/summary.py` | record summaries | да | нет | keep |
| `tile_preparation/annotations.py` | annotation load/filter helpers | да | нет | keep |
| `tile_preparation/iterator.py` | compatibility iterator | legacy | нет | keep as deprecated wrapper |

## tiling

| Файл | Что делает | Training | Inference | Пересечение | Решение |
|---|---|---:|---:|---|---|
| `tiling/windows.py` | generic window grid helpers | indirectly old code | да | overlaps `tile_preparation/windows.py` | keep for inference compatibility; do not use for new training |
| `tiling/tile_reader.py` | old tile reader helpers | нет для нового path | возможно old inference/debug | overlaps raster read | deprecate for training; migrate callers later |
| `tiling/stitching.py` | prediction stitching | нет | да | no direct train overlap | keep |
| `tiling/debug_tiling.py` | debug helpers | нет | debug | overlaps reports | keep/debug only |

## preprocessing

| Файл | Что делает | Training | Inference | Пересечение | Решение |
|---|---|---:|---:|---|---|
| `preprocessing/normalization.py` | older normalization helpers | no new training | possible legacy | overlaps `tile_preparation/raster_reader.py` | deprecate for training |
| `preprocessing/mask_rasterizer.py` | older mask rasterizer | no new training | possible legacy | overlaps `tile_preparation/mask_rasterizer.py` | deprecate wrapper or migrate |
| `preprocessing/preprocess_inventory.py` | inventory/preprocess utility | no direct tile samples | possible pipeline utility | no direct overlap | keep if still used |

## Решение

- Training tile windows, raster reads, valid clipping, mosaic, mask rasterization, augmentation и normalization выполняются только через `tile_preparation`.
- `tiling` остаётся для inference/stitching compatibility.
- `preprocessing` не должен получать новую логику подготовки training tiles.
- Расширение `tile_preparation` для inference нужно согласовать отдельным public API.
