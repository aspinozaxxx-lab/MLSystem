# Metrics Debug Report

Дата: 2026-05-08.

Базовый класс: `deforest` / `cuttings` / `вырубки`.

## Source of truth

Production validation loop формирует один `production_metrics_snapshot` на каждую эпоху. Этот snapshot используется для:

- MLflow logging;
- `epoch_summary.json`;
- `production_metrics_snapshot.json`;
- compact debug report folder;
- последующих stage artifacts/XCom через `training_result.json`.

`metrics_recompute_check.json` не является альтернативным источником истины. Он только проверяет, что значения из production snapshot пересчитываются из per-sample TP/FP/FN/TN с допуском `1e-6`.

## Что исправлено

- `val/pixel_f1`, `val/pixel_iou`, `val/precision`, `val/recall`, `val/pixel_accuracy` считаются из глобальных TP/FP/FN/TN по всей validation выборке.
- `val/pixel_tp`, `val/pixel_fp`, `val/pixel_fn`, `val/pixel_tn` и `val/threshold` пишутся в MLflow/history/training_result.
- `train/loss_total`, `train/loss_bce`, `train/loss_dice`, `val/loss_total`, `val/loss_bce`, `val/loss_dice` логируются отдельно и усредняются по количеству samples, а не как last batch.
- Object matching имеет deterministic tie-break: `(-iou, pred_index, gt_index)`.
- Metrics debug dump пишет artifacts по каждой завершенной эпохе.
- Compact report folder не копирует исходные raster/image датасета; в нем остаются lightweight masks, overlays and GeoJSON.
- Добавлен production-safe wall-clock limit: `train.max_wallclock_seconds`. Если параметр не задан, новый limiter отключен. Если задан, training останавливается только после безопасной точки между завершенными эпохами.

## Как включить Airflow debug report

Пример `dag_run.conf` для `mlsystem_experiment_pipeline`:

```json
{
  "experiment_id": "metrics_debug_cuttings_airflow_YYYYMMDD_HHMMSS",
  "class_name": "deforest",
  "task": "train_predict_pseudolabel",
  "images_uri": "s3://mlsystems/images/",
  "layout_uri": "s3://mlsystems/layouts/deforest/",
  "scenes_file": "scenes.txt",
  "annotation_file": "auto",
  "preprocess": {
    "use_all_matched_scenes": true,
    "use_full_dataset_tiles": true,
    "train_fraction": 0.75,
    "stratify_positive_validation": true,
    "max_empty_tile_share": 0.25
  },
  "train": {
    "enabled": true,
    "max_epochs": 100,
    "max_wallclock_seconds": 600,
    "seed": 20260508,
    "metric_threshold": 0.5,
    "require_gpu": true,
    "cache_samples_on_gpu": true
  },
  "params": {
    "class_name": "cuttings",
    "metrics_debug": {
      "enabled": true,
      "class_name": "вырубки",
      "class_id": 1,
      "save_every_epoch": true,
      "save_all_val_samples": true,
      "save_arrays": true,
      "save_png": true,
      "save_object_matching": true,
      "report_enabled": true,
      "upload_to_mlflow": false
    }
  }
}
```

Не задавать `max_train_batches`, `max_val_batches`, `max_train_tiles`, `max_val_tiles`, `max_scenes`, `dataset_limit`, `sample_size` для финального full-dataset debug run.

## Raw debug artifacts

Raw artifacts пишутся в:

```text
<experiment_dir>/metrics_debug/<run_id>/epoch_0001/
<experiment_dir>/metrics_debug/<run_id>/epoch_0002/
```

На каждую эпоху:

- `epoch_summary.json`
- `production_metrics_snapshot.json`
- `metrics.md`
- `per_sample_metrics.csv`
- `val_manifest_snapshot.json`
- `metrics_recompute_check.json`
- `mlflow_logged_metrics.json`
- `artifacts_manifest.csv`
- `geojson/gt_objects.geojson`
- `geojson/pred_objects.geojson`
- `geojson/object_matches.geojson`
- `samples/<sample_id>/metadata.json`
- `samples/<sample_id>/gt_mask.png`
- `samples/<sample_id>/pred_mask.png`
- `samples/<sample_id>/overlay_gt_pred.png`
- `samples/<sample_id>/confusion_map.png`
- `samples/<sample_id>/objects_gt.geojson`
- `samples/<sample_id>/objects_pred.geojson`
- `samples/<sample_id>/object_matches.json`
- `samples/<sample_id>/object_iou_matrix.csv`

Raw debug dir может содержать `image.png` и `pred_prob.npz` для глубокой диагностики, но compact report folder их не копирует.

## Compact report folder

Training создает compact report under:

```text
<experiment_dir>/metrics_debug_reports/<report_name>/
```

Структура:

```text
summary.md
run_metadata.json
metrics_timeseries.csv
metrics_timeseries.json
airflow/dag_run.json
airflow/task_statuses.json
airflow/airflow_url.txt
airflow/dag_conf.json
checks/source_of_truth_check.json
checks/mlflow_consistency_check.json
checks/report_structure_check.json
checks/dataset_full_run_check.json
epochs/epoch_0001/...
```

Эту папку можно копировать в `E:\Projects\reports\metrics_debug_cuttings_airflow_<airflow_run_id>_<timestamp>` после Airflow run.

## Required consistency checks

- `production_metrics_snapshot.json` equals values used by `epoch_summary.json`.
- MLflow metrics for the epoch match production snapshot values with `delta <= 1e-6`.
- Recompute from `per_sample_metrics.csv` matches production pixel metrics with `delta <= 1e-6`.
- `val_manifest_snapshot.json` is stable unless config intentionally changes validation data.
- `checks/report_structure_check.json` confirms raw source images were not copied into the compact report.

## Local verification

```text
python -m unittest discover -s tests
python -m unittest discover -s frontend/tests
python -m compileall -q mlsystem airflow frontend tests
git diff --check
```
