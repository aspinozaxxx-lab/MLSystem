# S3 Layout

Активный bucket для тяжелых данных:

```text
s3://mlsystems/
```

Целевая структура:

```text
s3://mlsystems/images/incoming/<delivery_name>/<files>
s3://mlsystems/images/kanopus/<delivery_name>/<scene_id>/metadata.json
s3://mlsystems/images/kanopus/<delivery_name>/<scene_id>/source.json
s3://mlsystems/images/kanopus/<delivery_name>/<scene_id>/previews/
s3://mlsystems/layouts/<class_name>/
s3://mlsystems/datasets/
s3://mlsystems/cache/
s3://mlsystems/models/
s3://mlsystems/predictions/
s3://mlsystems/pseudolabels/
s3://mlsystems/reports/
s3://mlsystems/experiments/
s3://mlsystems/system/manifests/images_manifest.json
s3://mlsystems/system/manifests/deliveries_manifest.json
```

`incoming` не удаляется и не перемещается автоматически. Preprocess daemon только регистрирует снимки, читает metadata и пишет маленькие JSON pointer-файлы в `images/kanopus/...`.

Загрузка поставки:

```bash
mc cp --recursive ./batch_001/ mlplatform/mlsystems/images/incoming/batch_001/
```

Загрузка разметки:

```bash
mc cp ./layout.geojson mlplatform/mlsystems/layouts/deforest/layout.geojson
mc cp ./scenes.txt mlplatform/mlsystems/layouts/deforest/scenes.txt
```

Проверка:

```bash
cd /home/worker/mlsystem
source /home/worker/ml-training/.venv/bin/activate
python -m src.cli s3-layout
python -m src.cli preprocess-once
```

Локальные копии маленьких manifests:

```text
/home/worker/mlsystem/storage/system/images_manifest.json
/home/worker/mlsystem/storage/system/deliveries_manifest.json
/home/worker/mlsystem/storage/system/preprocess_status.json
```

В Git не кладутся TIFF, checkpoints, probability maps, большие GeoJSON/GPKG, reports и caches. Секреты S3 читаются из окружения или существующего `mc` alias на сервере.
