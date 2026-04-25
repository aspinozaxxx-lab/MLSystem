# S3 layout

Existing server buckets observed during MVP work:

- `mlflow-artifacts`
- `ml-datasets`
- `dvc-storage`
- `triton-models`

Recommended logical layout:

```text
s3://ml-datasets/
  images/
  layouts/<class_name>/
  cache/

s3://mlflow-artifacts/
  mlflow/
  experiments/
```

Repository rules:

- Git stores code, configs, job YAML, docs, and small JSON summaries only.
- TIFF, checkpoints, probability maps, large GeoJSON/GPKG, reports, and caches stay in MinIO/S3.
- S3 credentials are read from environment or existing server config only.
- Do not write S3 credentials to YAML, README, workflow files, or logs.
