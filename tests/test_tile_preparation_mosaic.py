from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
from shapely.geometry import Polygon

from mlsystem.src.tile_preparation.config import TilePreparationConfig
from mlsystem.src.tile_preparation.mask_rasterizer import rasterize_mask_for_window
from mlsystem.src.tile_preparation.mosaic import read_mosaic_window
from mlsystem.src.tile_preparation.records import TileWindow
from mlsystem.src.tile_preparation.validity import read_valid_data_mask_with_source

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationMosaicTests(unittest.TestCase):
    def test_neighbor_fills_invalid_anchor_pixels_and_mask_uses_union_validity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor_path = root / "anchor.tif"
            neighbor_path = root / "neighbor.tif"
            _write_raster(anchor_path, left_value=90, right_value=0)
            _write_raster(neighbor_path, left_value=0, right_value=160)
            config = TilePreparationConfig(tile_size=64, stride=64, valid_pixel_mode="nonzero_any", mosaic_enabled=True)
            window = TileWindow("anchor", 0, 0, 64, 64, 64, 64)
            polygon = Polygon([(0, 64), (64, 64), (64, 0), (0, 0)])

            with rasterio.open(anchor_path) as anchor, rasterio.open(neighbor_path) as neighbor:
                anchor_valid = read_valid_data_mask_with_source(anchor, window, mode="nonzero_any")
                anchor_mask = rasterize_mask_for_window(anchor, [polygon], window, valid_mask=anchor_valid.mask)
                mosaic = read_mosaic_window(anchor, [("neighbor", neighbor)], window, config)
                mosaic_mask = rasterize_mask_for_window(anchor, [polygon], window, valid_mask=mosaic.valid_mask)

            self.assertAlmostEqual(anchor_valid.valid_pixel_share, 0.5)
            self.assertAlmostEqual(mosaic.final_valid_pixel_share, 1.0)
            self.assertGreater(mosaic.filled_pixel_count, 0)
            self.assertGreater(mosaic_mask.clipped_positive_pixels, anchor_mask.clipped_positive_pixels)
            self.assertEqual(mosaic.unfilled_pixel_count, 0)
            self.assertIn("neighbor", mosaic.source_scenes)


def _write_raster(path: Path, *, left_value: int, right_value: int) -> None:
    data = np.zeros((3, 64, 64), dtype="uint8")
    data[:, :, :32] = left_value
    data[:, :, 32:] = right_value
    with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 64, 1, 1)) as ds:
        ds.write(data)


if __name__ == "__main__":
    unittest.main()
