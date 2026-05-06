# MLSystem Python package

This package contains production code used by `mlsystem-api` and Airflow stages.

The current server path does not use a filesystem job queue or local CLI executor. Airflow starts stages through the API, and the API persists job state outside the repository.

Useful local checks:

```bash
python -m unittest discover -s tests
python -m compileall -q mlsystem airflow tests
```
