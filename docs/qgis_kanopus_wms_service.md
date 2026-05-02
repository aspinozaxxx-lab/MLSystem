# QGIS: Kanopus WMS/WMTS Service

## Recommended Connection

Use WMTS for raster imagery. It is cached and is the stable path for QGIS.

WMTS GetCapabilities:

```text
http://31.192.104.147:8082/mapcache/wmts/1.0.0/WMTSCapabilities.xml
```

Layers:

- `kanopus_all_nrg`: all Kanopus scenes, NRG pseudo-color, R=band 4, G=band 1, B=band 2.
- `kanopus_all_rgb`: all Kanopus scenes, RGB, R=band 1, G=band 2, B=band 3.

## WMS Fallback

Use WMS if QGIS needs arbitrary BBOX/CRS requests instead of cached WebMercator tiles.

WMS GetCapabilities:

```text
http://31.192.104.147:8082/cgi-bin/mapserv?map=/config/kanopus.map&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities
```

WMS layer names:

- `kanopus_all_nrg`
- `kanopus_all_rgb`
- `kanopus_footprints`
- `kanopus_labels`

## Overlays

Footprints GeoJSON:

```text
http://31.192.104.147:8082/static/kanopus_footprints.geojson
```

Labels GeoJSON:

```text
http://31.192.104.147:8082/static/kanopus_labels.geojson
```

In QGIS, add the labels GeoJSON as a vector layer, open layer properties, enable labels, and use the `label` field.

## QGIS Steps

1. Open `Layer` -> `Add Layer` -> `Add WMS/WMTS Layer`.
2. Create a new connection with the WMTS URL above.
3. Connect and add `kanopus_all_nrg` or `kanopus_all_rgb`.
4. Add footprints as a vector URL layer from the GeoJSON URL.
5. Add labels as a vector URL layer from the GeoJSON URL.
6. Enable labels on the labels layer using the `label` field.

## Server Notes

- Stack: MapServer 8.0.1 + Apache CGI + MapCache WMTS disk cache.
- Source data: `s3://mlsystems/images/kanopus/**/*.tif`.
- Source COG files are not modified or copied locally.
- Spatial index: `/data/mlsystem/mapservice2/index/kanopus_tileindex.shp` with `.qix`.
- Cache directory: `/data/mlsystem/mapservice2/cache`.
- Report: `/data/mlsystem/mapservice2/reports/wms_service_report.md`.

Current limitation: the service has no authentication and port `8082` is exposed. Put it behind VPN, basic auth, or an IP allowlist before wider access.
