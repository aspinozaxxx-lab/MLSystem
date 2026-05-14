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

InferenceEngine finalizer writes artifacts expected by the MLSystem pipeline runner into `run_dir`:

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
Probability tile artifacts remain tile-backed in the InferenceEngine job directory. The scene-level `npz_path` entries kept for legacy validators are compact placeholders in `source=inference_engine` mode; mlsystem downstream stages validate summaries and accepted vectors instead of rebuilding probability mosaics.

## Production Pipeline Boundary

The production pipeline runner has one pseudolabel stage:

- `inference_engine_pipeline` starts an InferenceEngine job over HTTP, polls it, and validates compatibility artifacts.

The old stage modules are not registered in `DEFAULT_PIPELINE_STAGES` or the production stage registry:

- `prepare_inference_scenes`
- `run_pseudolabel_inference`
- `validate_probability_maps`
- `vectorize_pseudolabel`
- `postprocess_pseudolabel`
- `export_pseudolabel_artifacts`

The retired `run/validate/vectorize/postprocess/export` stage modules have been removed. `prepare_inference_scenes.py` remains only as a small manifest helper used by `inference_engine_pipeline` before HTTP submission.

## Deferred From Training

No training/MLflow semantics are changed. The requested b2 SegFormer MLflow run id is used only as InferenceEngine model metadata and by `scripts/export_mlflow_run_to_triton.py`.
