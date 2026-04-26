# Deforest control pseudolabel full

Дата запуска: 2026-04-26.

## Статус

Контрольный job `deforest_control_pseudolabel_full` завершился успешно через очередь GitHub Actions/MLSystem.

- MLflow run: http://172.26.12.169:5000/#/experiments/1/runs/4616080884d040219bfa9a771482bb15
- Run name: `CTRL-pseudo-full`
- Task: `predict`
- Training: не запускался
- Device: CPU
- Duration: 1081.795 sec

## Checkpoint

Использован готовый checkpoint:

`/data/mlsystem/storage/experiments/deforest_exp04_r34_t512__20260425T202857249530Z/unet_resnet34.pt`

Причина выбора: это лучший доступный checkpoint по prior summaries, `best_val_iou = 0.44836788483030987`.

## Scene matching

- `scenes.txt` entries: 24
- matched scenes: 7
- ambiguous scenes: 0
- missing scenes: 17

Matched scenes:

| Scene | Status |
|---|---|
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN07.tif` | processed |
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN04.tif` | processed |
| `KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN03.tif` | processed |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN06.tif` | processed |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN05.tif` | processed |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN04.tif` | processed |
| `KVI_33499_32578-00_KANOPUS_20230729_035151_12.L2.PMS.SCN03.tif` | processed |

## Coverage

| Metric | Value |
|---|---:|
| scenes_processed | 7 |
| scenes_failed | 0 |
| total_expected_windows | 252 |
| total_predicted_windows | 248 |
| total_skipped_windows | 4 |
| mean_coverage_fraction | 0.984172 |
| min_coverage_fraction | 0.972240 |

Per scene:

| Scene | Expected | Predicted | Skipped | Coverage | Objects before filter | Objects after filter |
|---|---:|---:|---:|---:|---:|---:|
| SCN07 | 36 | 36 | 0 | 1.000000 | 3363 | 1117 |
| SCN04 | 36 | 36 | 0 | 1.000000 | 6461 | 2063 |
| SCN03 | 36 | 36 | 0 | 1.000000 | 3197 | 993 |
| KVI SCN06 | 36 | 35 | 1 | 0.972240 | 4082 | 1105 |
| KVI SCN05 | 36 | 35 | 1 | 0.972287 | 4022 | 1150 |
| KVI SCN04 | 36 | 35 | 1 | 0.972321 | 5392 | 1593 |
| KVI SCN03 | 36 | 35 | 1 | 0.972358 | 8720 | 2298 |

Покрытие больше не похоже на один тайл: по каждой сцене обработано 35-36 окон из 36 ожидаемых. Пропущенные 4 окна были all-zero/nodata.

## Postprocess

| Metric | Value |
|---|---:|
| threshold | 0.55 |
| min_area_m2 | 2000 |
| simplify_tolerance_m | 10 |
| objects_before_filter | 35237 |
| objects_after_filter | 10319 |
| objects_after_top500 | 500 |
| accepted_geojson_mb | 4.760598 |
| accepted_geojson_gz_mb | 1.454805 |
| accepted_gpkg_mb | 2.367188 |
| max_geojson_mb | 20 |

Top-500 применен после full-scene inference и полной векторизации. Итоговый GeoJSON уложился в лимит 20 MB.

## MLflow artifacts

В MLflow сохранены:

- `job.yml`
- `scenes_match_report.json`
- `coverage_report.json`
- `tiling_debug.json`
- `windows_preview.geojson`
- `postprocess_debug.json`
- `pseudolabel_summary.json`
- `accepted.geojson`
- `accepted.geojson.gz`
- `accepted.gpkg`
- `run_summary.json`
- `codex_summary.json`

## Вывод

Контрольный запуск подтверждает, что после фикса псевдоразметка строится по всей площади matched scenes, а не по одному или двум тайлам. Для текущих данных обработаны все 7 сопоставленных сцен, среднее покрытие составило 98.4%.

Остающиеся ограничения:

- 17 сцен из `scenes.txt` не найдены среди загруженных S3 снимков.
- Сервер CPU-only, поэтому полный inference по ResNet34 занял около 18 минут.
- Heavy artifacts остаются в MLflow/S3; в repo сохранен только легкий summary/report.
