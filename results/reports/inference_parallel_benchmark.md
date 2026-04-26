# Inference parallel benchmark

Дата проверки: 2026-04-26.

Сравнивались два контрольных pseudolabel запуска на одинаковом наборе из 7 matched scenes, с одним checkpoint `unet_resnet34`, `tile_size=2048`, `stride=2048`.

## Runs

| Run | Mode | max_workers | torch_threads_per_worker | Duration sec | CPU observation |
|---|---:|---:|---:|---:|---|
| `deforest_control_pseudolabel_full` | sequential | 1 | default | 1075.356 pseudolabel / 1081.795 total | около 50% общей CPU загрузки по наблюдению |
| `deforest_control_pseudolabel_full_v2` | scene threads | 2 | 4 | 1137.049 pseudolabel / 1143.741 total | процесс доходил примерно до 480-550% CPU |

MLflow v2: http://172.26.12.169:5000/#/experiments/1/runs/23757081db134064873d93e46b5d900b

## Результат

Thread-level parallel by scene безопасно отработал и не изменил покрытие/размеры, но не ускорил wall-clock:

- speedup vs sequential: `0.9466`, то есть v2 был примерно на 5.6% медленнее;
- вероятные причины: конкуренция PyTorch CPU threads, GDAL/rasterio S3 reads и overhead thread scheduling;
- память оставалась в безопасной зоне.

## Coverage consistency

| Metric | Sequential | Parallel v2 |
|---|---:|---:|
| scenes_processed | 7 | 7 |
| total_expected_windows | 252 | 252 |
| total_predicted_windows | 248 | 248 |
| mean_coverage_fraction | 0.984172 | 0.984172 |
| min_coverage_fraction | 0.972240 | 0.972240 |
| accepted_objects_total | 500 | 500 |
| accepted_geojson_mb | 4.760598 | 4.760601 |
| accepted_gpkg_mb | 2.367188 | 2.367188 |

## Вывод

Параллельный режим `max_workers=2` включен и логируется, но для текущего CPU-only сервера и ResNet34 он не дает ускорения. Для следующих проверок лучше:

- попробовать `max_workers=2`, `torch_threads_per_worker=2`;
- попробовать batching windows внутри одной сцены;
- вынести тяжелый inference на GPU-сервер;
- не увеличивать `max_workers` выше 2 без отдельного RAM/CPU теста.
