# Desertification Manual Tuning Journal

This is a human-readable journal for manually operated Airflow experiments on `Опустынивание` / `desertification`.
It is not an automation script and must not be used to launch or monitor experiments.

Policy:
- Launch one Airflow DAG run manually.
- Wait for completion.
- Verify MLflow run and metrics.
- Analyze pixel-level F1, precision, recall, stability, and checkpoint.
- Decide and launch the next experiment manually.
- Keep pseudo-labeling disabled.

