FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY monitoring/exporter/exporter.py /app/exporter.py
CMD ["python", "/app/exporter.py"]
