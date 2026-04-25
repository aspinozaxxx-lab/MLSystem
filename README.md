# MLSystem

Репозиторий для безопасного развертывания серверного MVP-конвейера ML-экспериментов.

Код конвейера живет в `mlsystem/`, серверная установка по умолчанию находится в `/home/worker/mlsystem`, тяжелые данные и артефакты должны храниться вне Git в MinIO/S3.

Основные команды на сервере:

```bash
cd /home/worker/mlsystem
source /home/worker/ml-training/.venv/bin/activate
python -m src.cli status
python -m src.cli run-once
python -m src.web_app
```

GitHub Actions не запускают обучение напрямую. Workflow только валидируют код/jobs, деплоят легкий код, кладут YAML-задания в серверную очередь и синхронизируют маленькие JSON-summary.
