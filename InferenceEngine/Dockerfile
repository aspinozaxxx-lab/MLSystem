FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl gdal-bin libgdal-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY InferenceEngine/pyproject.toml /app/InferenceEngine/pyproject.toml
COPY InferenceEngine/src /app/InferenceEngine/src
RUN pip install --no-cache-dir -e /app/InferenceEngine

EXPOSE 8095
CMD ["uvicorn", "inference_engine.api.app:app", "--host", "0.0.0.0", "--port", "8095"]
