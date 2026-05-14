# Metrics Debug Report

Р”Р°С‚Р°: 2026-05-08.

Р‘Р°Р·РѕРІС‹Р№ РєР»Р°СЃСЃ: `deforest` / `cuttings` / `РІС‹СЂСѓР±РєРё`.

## Source of truth

Production validation loop С„РѕСЂРјРёСЂСѓРµС‚ РѕРґРёРЅ `production_metrics_snapshot` РЅР° РєР°Р¶РґСѓСЋ СЌРїРѕС…Сѓ. Р­С‚РѕС‚ snapshot РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РґР»СЏ:

- MLflow logging;
- `epoch_summary.json`;
- `production_metrics_snapshot.json`;
- compact debug report folder;
- РїРѕСЃР»РµРґСѓСЋС‰РёС… stage artifacts/stage reports С‡РµСЂРµР· `training_result.json`.

`metrics_recompute_check.json` РЅРµ СЏРІР»СЏРµС‚СЃСЏ Р°Р»СЊС‚РµСЂРЅР°С‚РёРІРЅС‹Рј РёСЃС‚РѕС‡РЅРёРєРѕРј РёСЃС‚РёРЅС‹. РћРЅ С‚РѕР»СЊРєРѕ РїСЂРѕРІРµСЂСЏРµС‚, С‡С‚Рѕ Р·РЅР°С‡РµРЅРёСЏ РёР· production snapshot РїРµСЂРµСЃС‡РёС‚С‹РІР°СЋС‚СЃСЏ РёР· per-sample TP/FP/FN/TN СЃ РґРѕРїСѓСЃРєРѕРј `1e-6`.

## Р§С‚Рѕ РёСЃРїСЂР°РІР»РµРЅРѕ

- `val/pixel_f1`, `val/pixel_iou`, `val/precision`, `val/recall`, `val/pixel_accuracy` СЃС‡РёС‚Р°СЋС‚СЃСЏ РёР· РіР»РѕР±Р°Р»СЊРЅС‹С… TP/FP/FN/TN РїРѕ РІСЃРµР№ validation РІС‹Р±РѕСЂРєРµ.
- `val/pixel_tp`, `val/pixel_fp`, `val/pixel_fn`, `val/pixel_tn` Рё `val/threshold` РїРёС€СѓС‚СЃСЏ РІ MLflow/history/training_result.
- `train/loss_total`, `train/loss_bce`, `train/loss_dice`, `val/loss_total`, `val/loss_bce`, `val/loss_dice` Р»РѕРіРёСЂСѓСЋС‚СЃСЏ РѕС‚РґРµР»СЊРЅРѕ Рё СѓСЃСЂРµРґРЅСЏСЋС‚СЃСЏ РїРѕ РєРѕР»РёС‡РµСЃС‚РІСѓ samples, Р° РЅРµ РєР°Рє last batch.
- Object matching РёРјРµРµС‚ deterministic tie-break: `(-iou, pred_index, gt_index)`.
- Metrics debug dump РїРёС€РµС‚ artifacts РїРѕ РєР°Р¶РґРѕР№ Р·Р°РІРµСЂС€РµРЅРЅРѕР№ СЌРїРѕС…Рµ.
- Compact report folder РЅРµ РєРѕРїРёСЂСѓРµС‚ РёСЃС…РѕРґРЅС‹Рµ raster/image РґР°С‚Р°СЃРµС‚Р°; РІ РЅРµРј РѕСЃС‚Р°СЋС‚СЃСЏ lightweight masks, overlays and GeoJSON.
- Р”РѕР±Р°РІР»РµРЅ production-safe wall-clock limit: `train.max_wallclock_seconds`. Р•СЃР»Рё РїР°СЂР°РјРµС‚СЂ РЅРµ Р·Р°РґР°РЅ, РЅРѕРІС‹Р№ limiter РѕС‚РєР»СЋС‡РµРЅ. Р•СЃР»Рё Р·Р°РґР°РЅ, training РѕСЃС‚Р°РЅР°РІР»РёРІР°РµС‚СЃСЏ С‚РѕР»СЊРєРѕ РїРѕСЃР»Рµ Р±РµР·РѕРїР°СЃРЅРѕР№ С‚РѕС‡РєРё РјРµР¶РґСѓ Р·Р°РІРµСЂС€РµРЅРЅС‹РјРё СЌРїРѕС…Р°РјРё.

## РљР°Рє РІРєР»СЋС‡РёС‚СЊ pipeline debug report

РџСЂРёРјРµСЂ `dag_run.conf` РґР»СЏ `mlsystem_experiment_pipeline`:

```json
{
  "experiment_id": "metrics_debug_cuttings_pipeline_YYYYMMDD_HHMMSS",
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
      "class_name": "РІС‹СЂСѓР±РєРё",
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

РќРµ Р·Р°РґР°РІР°С‚СЊ `max_train_batches`, `max_val_batches`, `max_train_tiles`, `max_val_tiles`, `max_scenes`, `dataset_limit`, `sample_size` РґР»СЏ С„РёРЅР°Р»СЊРЅРѕРіРѕ full-dataset debug run.

## Raw debug artifacts

Raw artifacts РїРёС€СѓС‚СЃСЏ РІ:

```text
<experiment_dir>/metrics_debug/<run_id>/epoch_0001/
<experiment_dir>/metrics_debug/<run_id>/epoch_0002/
```

РќР° РєР°Р¶РґСѓСЋ СЌРїРѕС…Сѓ:

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

Raw debug dir РјРѕР¶РµС‚ СЃРѕРґРµСЂР¶Р°С‚СЊ `image.png` Рё `pred_prob.npz` РґР»СЏ РіР»СѓР±РѕРєРѕР№ РґРёР°РіРЅРѕСЃС‚РёРєРё, РЅРѕ compact report folder РёС… РЅРµ РєРѕРїРёСЂСѓРµС‚.

## Compact report folder

Training СЃРѕР·РґР°РµС‚ compact report under:

```text
<experiment_dir>/metrics_debug_reports/<report_name>/
```

РЎС‚СЂСѓРєС‚СѓСЂР°:

```text
summary.md
run_metadata.json
metrics_timeseries.csv
metrics_timeseries.json
pipeline/run.json
pipeline/stage_statuses.json
pipeline/run_url.txt
pipeline/trace.json
checks/source_of_truth_check.json
checks/mlflow_consistency_check.json
checks/report_structure_check.json
checks/dataset_full_run_check.json
epochs/epoch_0001/...
```

Р­С‚Сѓ РїР°РїРєСѓ РјРѕР¶РЅРѕ РєРѕРїРёСЂРѕРІР°С‚СЊ РІ `E:\Projects\reports\metrics_debug_cuttings_pipeline_<pipeline_run_id>_<timestamp>` РїРѕСЃР»Рµ pipeline run.

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
python -m compileall -q mlsystem frontend tests
git diff --check
```
