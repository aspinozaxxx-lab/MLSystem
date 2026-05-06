# MLSystem frontend

FastAPI BFF for the MLSystem web UI.

The annotation check page accepts a GeoJSON annotation file and a TXT scene list,
uploads them to a temporary MinIO prefix, and runs the same production
`mlsystem-api` stages used by Airflow:

- `inventory_scenes`
- `prepare_dataset`

The frontend does not count objects, match scenes, or split train/val itself.
It only renders stage reports and artifacts from
`/data/mlsystem/airflow/status/<run_id>/`.

Public URL after deploy:

```text
http://31.192.104.147/
```

The container listens on `8090`, but external access is through the managed
port 80 reverse proxy.

Required environment:

- `MLSYSTEM_FRONTEND_USER`
- `MLSYSTEM_FRONTEND_PASSWORD`
- `MLSYSTEM_FRONTEND_SESSION_SECRET`
- `MLSYSTEM_API_BASE_URL`
- `MLSYSTEM_API_TOKEN`
- `MLSYSTEM_FRONTEND_UPLOAD_ROOT`
- `MLSYSTEM_AIRFLOW_STATUS_ROOT`

Runtime uploads are stored outside git under
`/data/mlsystem/frontend/uploads/<run_id>/`.
