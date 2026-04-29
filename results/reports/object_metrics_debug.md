# Object F1 Local Debug

Метод: ЧТЗ, Приложение Г. Объект считается TP, если one-to-one match с GT имеет `IoU > 0.5`.

- GT: `E:\Projects\NSPD\Images\Разметка для обучения\Вырубки от 2026-04-24\layout.geojson`
- Prediction: `E:\Projects\razmetka\deforest_control_pseudolabel_full_v2.geojson`
- GT CRS: `urn:ogc:def:crs:EPSG::3857`
- Prediction CRS: `urn:ogc:def:crs:EPSG::3857`
- CRS note: Both GeoJSON files declare the same CRS; no reprojection was needed in this local debug.
- Matching: `greedy_descending_iou`

## Metrics

| metric | value |
|---|---:|
| object_tp | 42 |
| object_fp | 458 |
| object_fn | 258 |
| object_precision | 0.084000 |
| object_recall | 0.140000 |
| object_f1 | 0.105000 |
| object_iou_threshold | 0.5 |

`pixel_f1` не рассчитан: в локальной проверке не создавались растровые маски.

## Artifacts

- `results/reports/object_metrics_debug/object_metrics.json`
- `results/reports/object_metrics_debug/object_matches.csv`
- `results/reports/object_metrics_debug/object_tp.geojson`
- `results/reports/object_metrics_debug/object_fp.geojson`
- `results/reports/object_metrics_debug/object_fn.geojson`
- `results/reports/object_metrics_debug/object_metrics_preview.svg`
