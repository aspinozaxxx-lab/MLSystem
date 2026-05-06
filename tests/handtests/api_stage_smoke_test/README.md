# Ручной тест api_stage_smoke_test

Проверяет локальный API/job-store слой без запуска сервера, секретов и тяжелых stages.

`run.bat`:

- переходит в корень репозитория;
- очищает `output`;
- вызывает production helpers из `mlsystem/src/api`;
- записывает список stages и dry-run job response.

Тест не требует FastAPI, Docker, S3, MLflow или Triton.

Основные файлы:

- `output/stages.json`
- `output/dry_run_job.json`
