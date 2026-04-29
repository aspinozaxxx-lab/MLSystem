from __future__ import annotations

from typing import Any

import geopandas as gpd
from shapely.geometry import shape

from ..pipeline_config import PipelineConfig
from ..storage.s3 import read_s3_json


def load_shapes(config: PipelineConfig, annotation_uri: str) -> list[Any]:
    payload = read_s3_json(config, annotation_uri)
    return [shape(feature["geometry"]) for feature in payload.get("features", []) if feature.get("geometry")]


def load_geojson_geodataframe(path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(path)
