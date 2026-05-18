# dataset_preparing

## Purpose

`dataset_preparing` owns training dataset inspection and preparation before tile creation.

It checks dataset inputs, matches scene rows to images, counts annotation objects, builds train/val split, writes dataset manifests, and computes dataset identity for MLflow/reporting.

## Public API

```python
from mlsystem.src.dataset_preparing.api import (
    DatasetInspectionRequest,
    DatasetInspectionResult,
    TrainingDatasetPrepareRequest,
    TrainingDatasetPrepareResult,
    inspect_dataset,
    prepare_training_dataset,
)
```

### `inspect_dataset(request: DatasetInspectionRequest) -> DatasetInspectionResult`

- `request.experiment_id`: human-readable experiment id.
- `request.images_uri`: image collection URI.
- `request.layout_uri`: layout/annotation URI.
- `request.scenes_file`: scenes list file name.
- `request.annotation_file`: annotation file name or `auto`.
- `request.annotations`: annotation source metadata.
- `request.preprocess`: dataset inspection options.
- `request.output_dir`: optional directory for inspection artifacts.
- `request.runtime_config`: optional already-loaded runtime config.

### `prepare_training_dataset(request: TrainingDatasetPrepareRequest) -> TrainingDatasetPrepareResult`

- `request.run_id`: pipeline run id.
- `request.experiment_id`: experiment id.
- `request.output_dir`: run directory where manifest/report artifacts are written.
- `request.class_name`: optional class name for dataset identity.
- `request.class_slug`: optional class slug for dataset identity.
- `request.inventory`: inspection inventory payload.
- `request.matching`: scene matching payload.
- `request.preprocess`: split/count options.
- `request.raw_preprocess`: raw trace preprocess options used for audit.
- `request.annotations`: annotation source metadata.
- `request.schema_version`: optional pipeline schema version.
- `request.runtime_config`: optional already-loaded runtime config.

## Forbidden Overlaps

- No tile datasets, dataloaders, or augmentation: owned by `tile_preparation`.
- No model training: owned by `train`.
- No pipeline lifecycle or MLflow run ownership: owned by `train_pipeline`.
- No inference, vectorization, postprocess, or pseudolabel export: owned by InferenceEngine boundary.
- Low-level helpers such as scene matching, split functions, report writers, and dataset identity payload conversion are internal.
