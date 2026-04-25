# MLSystem server MVP

Lightweight server-side MVP for experiment orchestration on roskadastr-ml.

This project intentionally does not copy large TIFF files and does not run training by default. Large datasets, caches, and artifacts should live in MinIO/S3 or on a dedicated mounted data disk.

Quick start:

```bash
cd /home/worker/mlsystem
source /home/worker/ml-training/.venv/bin/activate
python -m src.cli status
python -m src.cli check-mlflow
python -m src.cli check-s3
python -m src.cli enqueue configs/job.example.yaml
python -m src.cli run-once
python -m src.web_app
```

Default web port: 8010.

Deployment is split between the code, infrastructure, queue, and results GitHub Actions workflows.
