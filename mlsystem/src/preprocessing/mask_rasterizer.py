from __future__ import annotations

from typing import Any

from rasterio.features import rasterize


def rasterize_shapes(shapes: list[Any], out_shape: tuple[int, int], transform: Any, dtype: str = "uint8") -> Any:
    return rasterize([(geom, 1) for geom in shapes], out_shape=out_shape, transform=transform, fill=0, dtype=dtype)
