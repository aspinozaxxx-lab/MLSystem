from __future__ import annotations

from typing import Any


def read_dataset_metadata(ds: Any) -> dict[str, Any]:
    return {
        "driver": ds.driver,
        "crs": str(ds.crs) if ds.crs else None,
        "width": ds.width,
        "height": ds.height,
        "band_count": ds.count,
        "dtypes": list(ds.dtypes),
        "nodata": ds.nodata,
        "bounds": list(ds.bounds),
    }
