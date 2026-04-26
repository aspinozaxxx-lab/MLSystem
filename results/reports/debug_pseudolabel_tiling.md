# Debug pseudolabel tiling

Дата проверки: 2026-04-26.

## Причина ошибки

Псевдоразметка строилась не по всей сцене. В `mlsystem/src/real_train.py` функция `_write_pseudolabel_outputs()` брала только `max_windows_per_scene` случайных окон на снимок. Для серии deforest jobs в YAML было задано `max_windows_per_scene: 2`, поэтому итоговый `accepted.geojson` отражал только 1-2 тайла, а не всю площадь снимка.

## Исправление

Псевдоразметка теперь строится через полный grid окон по сцене:

- формируются все окна с учетом `tile_size` и `stride`;
- edge windows добавляются так, чтобы закрыть правый и нижний край сцены;
- модель предсказывает каждое ненулевое окно;
- вероятности склеиваются в full-scene probability map через накопление `prob_sum/prob_count`;
- векторизация идет по полной карте вероятностей с `ds.transform`, а не по transform отдельного tile;
- фильтр мелких объектов, simplify и top-500 применяются после полной векторизации.

## Debug job

- Job: `debug_pseudolabel_tiling`
- Task: `predict`
- Model: `tiny_unet_4ch`
- Checkpoint: `/data/mlsystem/storage/experiments/deforest_exp09_tiny_t1024__20260425T202940527033Z/tiny_unet_4ch.pt`
- Scene: `s3://mlsystems/images/incoming/irkutsk/KV3_30861_31739-00_KANOPUS_20230826_035108_20.L2.PMS.SCN07.tif`
- MLflow run: http://172.26.12.169:5000/#/experiments/1/runs/897edd79e6514cbca315dd205d85630f

## До и после

| Проверка | До фикса по коду | После фикса |
|---|---:|---:|
| expected windows | 144 | 144 |
| predicted windows | максимум 2 | 115 |
| skipped windows | не диагностировалось | 29 all_zero |
| coverage fraction | до 0.0148 без учета overlap/skip | 0.8222 |
| probability bbox | только случайные tile | весь bbox снимка |
| objects before filter | не диагностировалось | 14261 |
| objects after min-area | не диагностировалось | 5747 |
| objects after top-500 | до 500 из неполного покрытия | 500 из full-scene покрытия |
| GeoJSON size MB | не диагностировалось | 3.435 |

29 окон были пропущены как `all_zero`, поэтому покрытие 82.2% считается ожидаемым для этой сцены: ненулевая probability map покрывает весь bbox снимка, где есть данные.

## Postprocess

- threshold: `0.5`
- min area: `500 m2`
- simplify tolerance: `5 m`
- max objects: `500`
- GeoJSON limit: `20 MB`
- result size: `3.435 MB`

Лимит 20 MB достигнут корректно: сначала выполнен full-scene inference, затем векторизация, фильтр мелких объектов, simplify и top-500 по площади.

## Artifacts

В MLflow run сохранены:

- `tiling_debug.json`
- `windows_preview.geojson`
- `probability_preview_kv3308613173900kanopus2023082603510820l2pmsscn07.png`
- `accepted_debug.geojson`
- `accepted.geojson`
- `accepted.geojson.gz`
- `accepted.gpkg`
- `postprocess_debug.json`
- `pseudolabel_summary.json`
- `run_summary.json`
- `codex_summary.json`
- `job.yml`

## Ограничения

- Проверка выполнена на одной сцене и CPU-only.
- В текущем S3 наборе 17 строк из `scenes.txt` не сопоставились со снимками.
- Для полного pseudolabel по всем снимкам CPU будет медленным; корректность tiling исправлена, но большой прогон лучше запускать отдельным job с лимитом времени.
