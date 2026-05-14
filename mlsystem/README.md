# MLSystem Python package

This package contains production code used by `mlsystem-api` and the MLSystem pipeline runner.

The current server path starts complete pipeline runs through FastAPI and persists run state outside the repository.

Useful local checks:

```bash
python -m unittest discover -s tests
python -m compileall -q mlsystem tests
```
