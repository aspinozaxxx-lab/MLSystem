# Airflow Real ML Acceptance

Date: 2026-04-29

## Result

Airflow run `AIR-real-mini-deforest-r5` completed successfully and executed real MLSystem logic, not synthetic smoke only.

- Airflow UI: http://172.26.12.169:8081
- DAG run: http://172.26.12.169:8081/dags/mlsystem_experiment_pipeline/grid?dag_run_id=AIR-real-mini-deforest-r5
- MLflow run: http://172.26.12.169:5000/#/experiments/1/runs/6ba597b9d1e34200a5894a96cbd3873c
- MLflow experiment: `mlsystem-deforest`
- MLflow run id: `6ba597b9d1e34200a5894a96cbd3873c`

## Stage Status

| task_id | status | implementation |
|---|---:|---|
| validate_experiment_config | success | real config validation |
| check_s3_layout | success | real S3/MinIO layout check |
| match_scenes | success | real S3 image listing and scene matching |
| validate_scene_matching | success | real matched-scene validation |
| inventory_images | success | wrapper over S3 inventory already gathered by matching |
| prepare_dataset_manifest | success | real Airflow manifest from matched scenes |
| prepare_train_tiles_or_windows | success | real config gate; physical sampling runs in training pipeline |
| validate_dataset | success | real manifest validation |
| create_mlflow_run | success | real MLflow run creation |
| train_model | success | real `TrainingPipeline -> real_train.run_real_train` |
| evaluate_pixel_metrics | success | real metrics read from training result |
| predict_validation_scenes | success | real inference result validation |
| vectorize_validation_predictions | success | real vectorization artifact validation |
| compute_object_f1 | success | real CHTZ object F1 from `object_metrics.json` |
| predict_pseudolabel_scenes | success | real coverage report validation |
| stitch_probability_maps | success | real stitching coverage validation |
| vectorize_pseudolabel | success | real accepted GeoJSON validation |
| postprocess_pseudolabel | success | real postprocess metrics validation |
| export_pseudolabel_artifacts | success | real artifact validation |
| generate_prediction_examples | success | real HTML artifact validation |
| log_mlflow_artifacts | success | real MLflow artifact logging |
| write_codex_api_summary | success | real summary JSON writing |
| finalize_mlflow_run | success | real MLflow finalization |

## Metrics

- `train/loss`: `1.7090254426002502`
- `train/iou`: `0.07215563456252842`
- `val/iou`: `3.814697265623545e-13`
- `val/dice`: `3.814697265623545e-13`
- `val/pixel_f1`: `1.0000075293933967e-07`
- `val/object_f1`: `0.0`
- `val/object_precision`: `0.0`
- `val/object_recall`: `0.0`
- `val/object_tp`: `0.0`
- `val/object_fp`: `1.0`
- `val/object_fn`: `300.0`
- `accepted_objects`: `1`
- `coverage_fraction`: `0.7723794744217091`

## Artifacts

Required artifacts were present in MLflow:

- `train_scenes.txt`
- `pseudolabel_scenes.txt`
- `AIR-real-mini-deforest-r5.accepted.geojson`
- `object_metrics.json`
- `coverage_report.json`
- `prediction_examples.html`
- `run_summary.json`
- `codex_summary.json`
- `history.json`
- `history.csv`

Forbidden legacy artifacts were not logged to MLflow:

- `accepted.geojson.gz`
- `accepted.gpkg`
- `accepted_debug.geojson`
- `windows_preview.geojson`

Some debug/local-only files are still created in the Airflow run directory and intentionally filtered from MLflow. `accepted.geojson.gz` and `windows_preview.geojson` were present locally but excluded from MLflow logging.

## XCom

XCom stayed small. Largest observed XCom row was `2784` bytes (`check_s3_layout`). Probability maps, masks, rasters and GeoJSON payloads were not passed through XCom.

## Remaining Limitations

- The heavy ML implementation is still physically executed inside `train_model` through `TrainingPipeline -> real_train.run_real_train`.
- Downstream Airflow stages are real validation/reporting gates over produced outputs, but not yet separate compute tasks.
- A mini-run fallback is enabled for Airflow configs so tiny scene subsets can create validation samples from training holdout when the selected validation scene has no usable samples.
- `coverage_fraction` was `0.7723` because all-zero windows are skipped; this is current MLSystem behavior and should be reviewed separately from Airflow orchestration.
