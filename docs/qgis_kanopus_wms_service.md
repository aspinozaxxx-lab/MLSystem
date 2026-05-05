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
- Delivery layers are also published as WMTS tilesets:
  `Hilokskij_nrg`, `Hilokskij_rgb`, `Kachugskij_nrg`, `Kachugskij_rgb`,
  `Ohinskij_nrg`, `Ohinskij_rgb`, `Oktyabrskij_nrg`, `Oktyabrskij_rgb`,
  `Olhonskij_nrg`, `Olhonskij_rgb`, `Olskij_nrg`, `Olskij_rgb`,
  `Toguchinskij_nrg`, `Toguchinskij_rgb`, `irkutsk_nrg`, `irkutsk_rgb`,
  `wave_2_Upload_01_nrg`, `wave_2_Upload_01_rgb`.

## WMS

Use WMS for the full layer tree, including per-scene checkbox layers.

WMS GetCapabilities:

```text
http://31.192.104.147:8082/cgi-bin/mapserv?map=/config/kanopus.map&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities
```

WMS layer names:

- `kanopus_all_nrg`
- `kanopus_all_rgb`
- the delivery layers listed above
- 995 per-scene NRG layers named `scene_<safe_scene_id>_nrg`
- `kanopus_footprints_full`
- `kanopus_labels`
- `kanopus_overlap_zones`

WMS groups:

- `/Kanopus/All`
- `/Kanopus/Deliveries/<delivery>/NRG`
- `/Kanopus/Deliveries/<delivery>/RGB`
- `/Kanopus/Scenes/<delivery>/NRG`
- `/Kanopus/Overlays`

Per-scene RGB layers are not published. RGB is available for all and delivery layers.

## QGIS Project

Ready-to-open project and layer definition:

```text
http://31.192.104.147:8082/static/qgis_kanopus_project.qgs
http://31.192.104.147:8082/static/qgis_kanopus_layers.qlr
```

The project groups layers as:

```text
Kanopus
  Common layers
  Footprints and labels
  Deliveries
  Scenes
```

`All NRG` and `Footprints full` are enabled by default. Per-scene layers are unchecked by default, so QGIS does not request all scenes at startup.

## Overlays

Full footprints GeoJSON:

```text
http://31.192.104.147:8082/static/kanopus_footprints_full.geojson
```

These footprints are built from the COG valid-data mask, not from rectangular COG bounds. They include all 995 published valid scenes and are simplified for QGIS use.

Labels GeoJSON:

```text
http://31.192.104.147:8082/static/kanopus_labels.geojson
```

Labels are representative points inside the valid footprint. In QGIS, add the labels GeoJSON as a vector layer, open layer properties, enable labels, and use the `label` field.

Overlap zones GeoJSON:

```text
http://31.192.104.147:8082/static/kanopus_overlap_zones.geojson
```

Overlap zones are calculated from valid-data footprints.

Scene catalog:

```text
http://31.192.104.147:8082/static/kanopus_scene_catalog.html
http://31.192.104.147:8082/static/kanopus_scene_catalog.json
```

## QGIS Steps

Fast manual setup:

1. Open `Layer` -> `Add Layer` -> `Add WMS/WMTS Layer`.
2. Create a new connection with the WMS URL above.
3. Connect and add `kanopus_all_nrg`, `kanopus_all_rgb`, delivery layers, or individual `scene_*_nrg` layers.
4. To switch deliveries independently, add the corresponding `*_nrg` or `*_rgb` delivery layer.
5. Add full footprints as a vector URL layer from the GeoJSON URL.
6. Add labels as a vector URL layer from the GeoJSON URL.
7. Optionally add overlap zones as a vector URL layer.
8. Enable labels on the labels layer using the `label` field.
9. To determine which scene is under a point, use QGIS `Identify Features` on the full footprints layer and read `delivery`, `file_name`, `scene_id`, and `s3_uri`.

Recommended setup for cartographers:

1. Open `qgis_kanopus_project.qgs`, or add `qgis_kanopus_layers.qlr`.
2. Keep `All NRG` on for context.
3. Expand `Scenes` -> delivery name.
4. Enable specific scene checkboxes as needed.
5. Use `Identify Features` on `Footprints full` to see exact scene metadata.

## Server Notes

- Stack: MapServer 8.0.1 + Apache CGI + MapCache WMTS disk cache.
- Source data: `s3://mlsystems/images/kanopus/**/*.tif`.
- Source COG files are not modified or copied locally.
- Spatial index: `/data/mlsystem/mapservice2/index/kanopus_tileindex.shp` with `.qix`.
- Delivery spatial indexes: `/data/mlsystem/mapservice2/index/deliveries/*_tileindex.shp`.
- Cache directory: `/data/mlsystem/mapservice2/cache`.
- Cache backend: one MapCache disk root, separated internally by tileset/layer directories such as `kanopus_all_nrg/`, `kanopus_all_rgb/`, `Kachugskij_nrg/`.
- Per-scene layers are WMS only. They are not preseeded and are not added as MapCache tilesets.
- Report: `/data/mlsystem/mapservice2/reports/wms_service_report.md`.
- Valid footprints report: `/data/mlsystem/mapservice2/reports/kanopus_footprints_valid_report.md`.
- Full footprints report: `/data/mlsystem/mapservice2/reports/footprints_full_report.md`.
- Overlap report: `/data/mlsystem/mapservice2/reports/kanopus_overlap_report.md`.
- Missing TIFF reindex report: `/data/mlsystem/mapservice2/reports/missing_tifs_reindex_report.md`.
- WMS GetCapabilities with scene layers is about 1.32 MB and responded in about 6.5 seconds on the server test after publishing 995 scene layers.

Current data note: S3 currently has 996 TIFF objects under `images/kanopus`, but one Kachugskij object is not published because GDAL/rasterio does not recognize it as a supported TIFF. The service publishes the 995 valid COGs.

Current limitation: the service has no authentication and port `8082` is exposed. Put it behind VPN, basic auth, or an IP allowlist before wider access.
