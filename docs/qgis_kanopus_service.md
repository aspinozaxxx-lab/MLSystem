# QGIS Kanopus Map Service

Картосервис на GPU-сервере публикует COG-снимки Канопуса из MinIO/S3 через TiTiler.

## S3 Layout

Рабочая структура снимков:

```text
s3://mlsystems/images/kanopus/<delivery_folder>/*.tif
```

`images/incoming` больше не является рабочим местом для картографов. Автоматическая предобработка MLSystem не должна перекладывать COG из этой структуры.

## QGIS Layers

### XYZ: NRG 4/1/2

Добавить как **XYZ Tiles**:

```text
http://31.192.104.147:8082/mosaicjson/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=file%3A%2F%2F%2Fdata%2Fmlsystem%2Fmapservice%2Fkanopus_mosaic.json&bidx=4&bidx=1&bidx=2&rescale=1%2C255&rescale=1%2C255&rescale=1%2C255&tilesize=512
```

Каналы: `R=band4`, `G=band1`, `B=band2`.

### XYZ: RGB 1/2/3

Добавить как **XYZ Tiles**:

```text
http://31.192.104.147:8082/mosaicjson/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=file%3A%2F%2F%2Fdata%2Fmlsystem%2Fmapservice%2Fkanopus_mosaic.json&bidx=1&bidx=2&bidx=3&rescale=1%2C255&rescale=1%2C255&rescale=1%2C255&tilesize=512
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
- Для production-доступа порт `8082` нужно держать за VPN или закрыть basic auth/reverse proxy policy.
