# SegFormer Tiling Artifact Debug

Дата проверки: 2026-04-27.

## Вывод

Причина тайловых артефактов найдена в stitching pseudolabel inference для SegFormer.

Jobs `deforest_full_exp02_segformer_b0_t1024_aug` и `deforest_full_exp03_segformer_b1_t1024_aug` задавали `sample_size: 768` и `context_bounds: 128`, но код вставлял в `prob_map` весь prediction тайла `1024x1024`. Для SegFormer это оставляло в итоговой карте нестабильные края тайлов. После threshold/vectorization они визуально проявлялись как прямоугольные/тайловые блоки.

Исправление: для SegFormer при наличии `sample_size/context_bounds` теперь используется `crop_mode: center`.

- внутренний tile `1024x1024` вставляет центральную область `768x768`;
- координата вставки смещается на `+128/+128`;
- edge tiles сохраняют внешнюю сторону, чтобы не создавать дыр по краям сцены;
- CNN-модели без `center_size` продолжают использовать full-tile insertion.

## Debug Run

MLflow run:

http://172.26.12.169:5000/#/experiments/1/runs/e8165d57bfcd47eb83100c3abeadc419

Локальная директория артефактов на сервере:

`/data/mlsystem/storage/experiments/segformer_tiling_artifact_debug__20260427T091137750657Z`

## Проверка На Одной Сцене

| Параметр | Значение |
|---|---:|
| model | `segformer_b0` |
| checkpoint | `deforest_full_exp02.../segformer_b0.pt` |
| tile_size | 1024 |
| stride | 768 |
| crop_mode | `center` |
| center_size | 768 |
| context_bounds | 128 |
| expected windows | 240 |
| predicted windows | 202 |
| mean coverage | 0.839758 |
| min coverage | 0.839758 |
| accepted objects | 500 |
| accepted.geojson | 4.258 MB |
| accepted.gpkg | 2.172 MB |

Пример insert для верхнего ряда:

`x=4608, y=0, tile=1024x1024 -> insert_x=4736, insert_y=0, insert=768x896`

Это корректно: по X применяется смещение `+128`, по Y верхний край сохраняется, чтобы не потерять покрытие края изображения.

## Debug Artifacts

- `segformer_tiling_debug.json`
- `tile_insert_debug.geojson`
- `windows_preview.geojson`
- `probability_preview_*.png`
- `center_crop_preview_*.png`
- `accepted_debug.geojson`
- `coverage_report.json`
- `postprocess_debug.json`

## Ограничения

Debug сделан на одной сцене, чтобы не запускать тяжелую обработку поверх текущих серверных jobs. Для финальной проверки нужны новые train jobs с исправленной логикой и последующей full-scene pseudolabel.
