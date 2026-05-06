FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY frontend/requirements.txt /tmp/frontend-requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/frontend-requirements.txt

COPY frontend /app/frontend
COPY docs/frontend.md /app/docs/frontend.md

EXPOSE 8090

CMD ["python", "-m", "uvicorn", "frontend.app.main:app", "--host", "0.0.0.0", "--port", "8090"]
