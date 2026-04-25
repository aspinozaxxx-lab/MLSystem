# Codex result summaries

`sync-results.yml` writes one small JSON file per job here.

Expected shape:

```json
{
  "schema_version": 1,
  "job_id": "deforest-r34-001",
  "status": "done",
  "task": "train_predict_pseudolabel",
  "server": "roskadastr-ml",
  "mlflow_run_url": "http://127.0.0.1:5000/#/experiments/6/runs/...",
  "s3_artifacts": {
    "accepted_geojson": "s3://...",
    "report_html": "s3://..."
  },
  "metrics": {
    "best_val_iou": null,
    "accepted_objects": null,
    "accepted_geojson_mb": null
  },
  "warnings": [],
  "errors": []
}
```
