from __future__ import annotations

from typing import Any


def read_raster_window(ds: Any, bands: list[int], window: Any, *, boundless: bool = True, fill_value: int = 0) -> Any:
    return ds.read(bands, window=window, boundless=boundless, fill_value=fill_value)
