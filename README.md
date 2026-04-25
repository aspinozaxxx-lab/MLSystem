# MLSystem

Репозиторий для серверного MVP-конвейера ML-экспериментов.

Серверная установка по умолчанию:

```bash
/home/worker/mlsystem
```

Тяжелые данные, снимки, кэши, модели и большие артефакты хранятся вне Git в MinIO/S3. Активный bucket:

```text
s3://mlsystems/
```

Основные команды на сервере:

```bash
cd /home/worker/mlsystem
source /home/worker/ml-training/.venv/bin/activate
python -m src.cli status
python -m src.cli check-s3
python -m src.cli s3-layout
python -m src.cli preprocess-once
python -m src.cli queue-list
```

Загрузка поставки снимков:

```bash
mc cp --recursive ./batch_001/ mlplatform/mlsystems/images/incoming/batch_001/
```

Загрузка разметки:

```bash
mc cp ./layout.geojson mlplatform/mlsystems/layouts/deforest/layout.geojson
mc cp ./scenes.txt mlplatform/mlsystems/layouts/deforest/scenes.txt
```

GitHub Actions не запускают тяжелое обучение напрямую. Они деплоят код/инфраструктуру через Ansible, ставят YAML jobs в серверную очередь и синхронизируют только маленькие JSON summaries.
