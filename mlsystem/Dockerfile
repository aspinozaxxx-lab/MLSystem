ARG MLSYSTEM_BASE_IMAGE=python:3.11-slim
FROM ${MLSYSTEM_BASE_IMAGE}

COPY mlsystem/requirements-runtime.txt /tmp/mlsystem-requirements-runtime.txt

RUN python -m pip install --no-cache-dir -r /tmp/mlsystem-requirements-runtime.txt \
    && rm -f /tmp/mlsystem-requirements-runtime.txt
