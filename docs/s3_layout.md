# S3 Layout

Active bucket for heavy ML data:

```text
s3://mlsystems/
```

Target structure:

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

`incoming` is never deleted or moved automatically. The preprocess daemon registers images, reads metadata, and writes small JSON pointer files under `images/kanopus/...`.

Upload an image delivery:

```bash
mc cp --recursive ./batch_001/ mlplatform/mlsystems/images/incoming/batch_001/
```

Upload labels:

```bash
mc cp ./layout.geojson mlplatform/mlsystems/layouts/deforest/layout.geojson
mc cp ./scenes.txt mlplatform/mlsystems/layouts/deforest/scenes.txt
```

Check S3 and rebuild manifests:

```bash
cd /home/worker/mlsystem
source /home/worker/ml-training/.venv/bin/activate
python -m src.cli s3-layout
python -m src.cli preprocess-once
```

Local copies of small manifests live on `/data`:

```text
/data/mlsystem/storage/system/images_manifest.json
/data/mlsystem/storage/system/deliveries_manifest.json
/data/mlsystem/storage/system/preprocess_status.json
```

Compatibility symlink:

```text
/home/worker/mlsystem/storage -> /data/mlsystem/storage
```

Git must not store TIFF, checkpoints, probability maps, large GeoJSON/GPKG, reports, caches, or secrets.
