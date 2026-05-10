# InferenceEngine Inventory

## Moved To InferenceEngine

Source of truth for pseudolabel domain logic is now under `InferenceEngine/src/inference_engine`.

| Old mlsystem path | New InferenceEngine path | Notes |
|---|---|---|
| `mlsystem/src/tiling/windows.py` | `tiling/windows.py` | Tile origins, grid, crop/insert/weight windows. |
| `mlsystem/src/inference/probability_map.py` | `probability/map.py` | ProbabilityMap accumulator and coverage stats. |
| `mlsystem/src/inference/triton_client.py` | `triton/client.py` | Triton HTTP client using `INPUT__0`/`OUTPUT__0`. |
| `mlsystem/src/inference/scene_inference.py` | `workers/scene_inference.py`, `workers/tile_workers.py` | Scene inference split into planning, preprocessing, Triton batch/persist. |
| `mlsystem/src/vectorization/*` | `vectorization/*`, `workers/block_worker.py` | block/core/halo plan, vectorize expanded block, clip to core, final merge. |
| `mlsystem/src/postprocessing/vectorization.py` | `postprocessing/vectorization.py` | Raster-to-vector conversion. |
| `mlsystem/src/postprocessing/filtering.py` | `postprocessing/filtering.py` | Area/top-N filtering helpers. |
| `mlsystem/src/postprocessing/simplification.py` | `postprocessing/simplification.py` | Geometry simplification. |
| `mlsystem/src/postprocessing/thresholding.py` | `postprocessing/thresholding.py` | Probability thresholding. |
| `mlsystem/src/postprocessing/pseudolabel_export.py` | `postprocessing/pseudolabel_export.py`, `workers/finalizer.py` | GeoJSON/GZip compatibility artifacts. |

## Thin Wrappers Remaining In mlsystem

These modules now re-export InferenceEngine implementations for backwards compatibility:

- `mlsystem/src/tiling/windows.py`
- `mlsystem/src/inference/probability_map.py`
- `mlsystem/src/inference/triton_client.py`
- `mlsystem/src/inference/scene_inference.py`
- `mlsystem/src/vectorization/*.py`
- `mlsystem/src/postprocessing/*.py`

Training, MLflow run creation/finalization, pixel/object metrics, dataset preparation, and validation prediction semantics remain in `mlsystem`.

## Compatibility Artifacts

InferenceEngine finalizer writes artifacts expected by the old Airflow pipeline into `run_dir`:

- `<experiment_id>.accepted.geojson`
- `accepted.geojson.gz`
- `coverage_report.json`
- `pseudolabel_summary.json`
- `postprocess_summary.json`
- `vectorization_summary.json`
- `pseudolabel_scene_results_manifest.json`
- `probability_maps_index.json`
- `inference_results.json`
- `inference_timing_report.json`
- `pseudolabel_scenes.txt`
- `prediction_examples.html`

Runtime arrays and intermediate queue artifacts stay under `/data/mlsystem/inference-engine/...` and are not committed.

## Validate-Only Stages

When `pseudolabel.source=inference_engine`:

- `run_pseudolabel_inference` starts an InferenceEngine job and polls it.
- `validate_probability_maps` validates `coverage_report.json` and probability index.
- `vectorize_pseudolabel` validates InferenceEngine vectorization artifacts and skips CPU vectorization in `mlsystem`.
- `postprocess_pseudolabel` validates `postprocess_summary.json` and skips heavy postprocess.
- `export_pseudolabel_artifacts` validates required compatibility artifacts.

## Deferred From Training

No training/MLflow semantics are changed. The requested b2 SegFormer MLflow run id is used only as InferenceEngine model metadata and by `scripts/export_mlflow_run_to_triton.py`.
