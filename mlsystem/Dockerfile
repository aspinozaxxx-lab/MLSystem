ARG AIRFLOW_BASE_IMAGE=apache/airflow:2.9.3-python3.11
FROM ${AIRFLOW_BASE_IMAGE}

COPY mlsystem/requirements-runtime.txt /tmp/mlsystem-requirements-runtime.txt

RUN python -m pip install --no-cache-dir -r /tmp/mlsystem-requirements-runtime.txt \
    && rm -f /tmp/mlsystem-requirements-runtime.txt
