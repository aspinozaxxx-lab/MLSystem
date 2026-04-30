# QGIS Kanopus Map Service

Картосервис публикует все COG-снимки Канопуса из MinIO/S3 одним общим XYZ-слоем через lightweight gateway на базе rio-tiler.

## QGIS Layers

### Common NRG 4/1/2

Добавить как **XYZ Tiles**:

```text
http://31.192.104.147:8082/kanopus/tiles/nrg_412/{z}/{x}/{y}.png
```

Каналы: `R=band4`, `G=band1`, `B=band2`.

### Common RGB 1/2/3

Добавить как **XYZ Tiles**:

```text
http://31.192.104.147:8082/kanopus/tiles/rgb_123/{z}/{x}/{y}.png
```

Каналы: `R=band1`, `G=band2`, `B=band3`.

### Footprints

Добавить как **Vector Layer URL**:

```text
http://31.192.104.147:8082/static/kanopus_footprints.geojson
```

### Labels

Добавить как **Vector Layer URL**:

```text
http://31.192.104.147:8082/static/kanopus_labels.geojson
```

Включить подписи слоя по полю `label`.

## Catalog

HTML/JSON-каталог:

```text
http://31.192.104.147:8082/static/qgis_kanopus_layers.html
http://31.192.104.147:8082/static/qgis_kanopus_layers.json
```

## Implementation Notes

- COG лежат в `s3://mlsystems/images/kanopus/<delivery_folder>/*.tif`.
- Основной слой не использует тяжелый all-MosaicJSON TiTiler endpoint.
- Tile lookup идет через SQLite RTree spatial index: `/data/mlsystem/mapservice/index/kanopus_footprints.sqlite`.
- nginx кеширует tiles в `/data/mlsystem/mapservice/cache/nginx`.
- Запросы с очень широким fan-out на низких zoom защищены быстрым `204`, чтобы QGIS не перегружал сервер при первом подключении.
- XYZ tiles являются RGB/PNG-визуализацией для просмотра и разметки, а не 4-band raster delivery.
- Для production-доступа порт `8082` нужно держать за VPN или закрыть basic auth/reverse proxy policy.
