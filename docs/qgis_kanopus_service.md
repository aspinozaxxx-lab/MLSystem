# QGIS Kanopus Map Service

Картосервис на GPU-сервере публикует COG-снимки Канопуса из MinIO/S3 через TiTiler.

## S3 Layout

Рабочая структура снимков:

```text
s3://mlsystems/images/kanopus/<delivery_folder>/*.tif
```

`images/incoming` больше не является рабочим местом для картографов. Автоматическая предобработка MLSystem не должна перекладывать COG из этой структуры.

## QGIS Layers

### Layer Index

The all-Kanopus XYZ mosaic is intentionally not recommended for QGIS. QGIS requests low-zoom world tiles when a layer is added, and the single mosaic of all deliveries can overload TiTiler before a useful map view is reached.

Use the per-delivery XYZ URLs from:

```text
http://31.192.104.147:8082/static/qgis_kanopus_layers.json
```

### XYZ: NRG 4/1/2 Per Delivery

Example for `irkutsk`, add as **XYZ Tiles**:

```text
http://31.192.104.147:8082/mosaicjson/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=file%3A%2F%2F%2Fdata%2Fmlsystem%2Fmapservice%2Fmosaics%2Firkutsk_mosaic.json&bidx=4&bidx=1&bidx=2&rescale=1%2C255&rescale=1%2C255&rescale=1%2C255
```

Каналы: `R=band4`, `G=band1`, `B=band2`.

### XYZ: RGB 1/2/3 Per Delivery

Example for `irkutsk`, add as **XYZ Tiles**:

```text
http://31.192.104.147:8082/mosaicjson/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=file%3A%2F%2F%2Fdata%2Fmlsystem%2Fmapservice%2Fmosaics%2Firkutsk_mosaic.json&bidx=1&bidx=2&bidx=3&rescale=1%2C255&rescale=1%2C255&rescale=1%2C255
```

Каналы переключаются порядком повторяющихся параметров `bidx`; порядок соответствует выходным `R,G,B`.

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

## Notes

- XYZ tiles являются RGB/PNG-визуализацией для просмотра и разметки, а не 4-band raster delivery.
- Все валидные COG имеют overviews; массовая пересборка COG для скорости сейчас не требуется.
- Use footprints first, zoom to the delivery area, then enable the matching delivery XYZ layer.
- Для production-доступа порт `8082` нужно держать за VPN или закрыть basic auth/reverse proxy policy.
