from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from shapely.geometry import box

from mlsystem.src.tile_preparation.config import TilePreparationConfig
from mlsystem.src.tile_preparation.footprint import SceneFootprint
from mlsystem.src.tile_preparation.mosaic import read_mosaic_window
from mlsystem.src.tile_preparation.records import TileWindow

try:
    import rasterio
    from rasterio.transform import from_origin

    HAS_RASTERIO = True
except Exception:  # noqa: BLE001
    HAS_RASTERIO = False


@unittest.skipUnless(HAS_RASTERIO, "rasterio is required")
class TilePreparationMosaicOptimizedTests(unittest.TestCase):
    def test_no_candidates_does_not_create_warped_vrt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            anchor_path = Path(tmp) / "anchor.tif"
            _write_raster(anchor_path, left_value=90, right_value=0)
            config = TilePreparationConfig(tile_size=100, stride=100, valid_pixel_mode="nonzero_any", mosaic_enabled=True)
            window = TileWindow("anchor", 0, 0, 100, 100, 100, 100)

            with rasterio.open(anchor_path) as anchor:
                with mock.patch("mlsystem.src.tile_preparation.mosaic.WarpedVRT", side_effect=AssertionError("WarpedVRT should not be created")):
                    result = read_mosaic_window(anchor, [], window, config, anchor_footprint=_footprint("anchor", box(0, 0, 50, 100)))

            self.assertEqual(result.candidate_neighbors, 0)
            self.assertTrue(result.skipped_no_candidates)
            self.assertEqual(result.warped_vrt_calls, 0)
            self.assertEqual(result.filled_pixel_count, 0)

    def test_neighbor_footprint_mask_is_used_before_pixel_validity_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor_path = root / "anchor.tif"
            neighbor_path = root / "neighbor.tif"
            _write_raster(anchor_path, left_value=90, right_value=0)
            _write_raster(neighbor_path, left_value=160, right_value=160)
            config = TilePreparationConfig(tile_size=100, stride=100, valid_pixel_mode="nonzero_any", mosaic_enabled=True)
            window = TileWindow("anchor", 0, 0, 100, 100, 100, 100)
            anchor_footprint = _footprint("anchor", box(0, 0, 50, 100))
            neighbor_footprint = _footprint("neighbor", box(50, 0, 100, 100))

            with rasterio.open(anchor_path) as anchor, rasterio.open(neighbor_path) as neighbor:
                with mock.patch("mlsystem.src.tile_preparation.mosaic.read_valid_data_mask_with_source", side_effect=AssertionError("pixel valid fallback should not be used")):
                    result = read_mosaic_window(
                        anchor,
                        [("neighbor", neighbor)],
                        window,
                        config,
                        anchor_footprint=anchor_footprint,
                        neighbor_footprints={"neighbor": neighbor_footprint},
                    )

            self.assertEqual(result.intersecting_neighbors, 1)
            self.assertEqual(result.warped_vrt_calls, 1)
            self.assertEqual(result.neighbor_valid_from_footprint, 1)
            self.assertEqual(result.neighbor_valid_from_pixels_fallback, 0)
            self.assertEqual(result.filled_pixel_count, 5000)
            self.assertEqual(result.unfilled_pixel_count, 0)
            self.assertEqual(result.actually_used_neighbors, ["neighbor"])


def _write_raster(path: Path, *, left_value: int, right_value: int) -> None:
    data = np.zeros((3, 100, 100), dtype="uint8")
    data[:, :, :50] = left_value
    data[:, :, 50:] = right_value
    with rasterio.open(path, "w", driver="GTiff", width=100, height=100, count=3, dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 100, 1, 1)) as ds:
        ds.write(data)


def _footprint(scene_id: str, polygon) -> SceneFootprint:
    minx, miny, maxx, maxy = polygon.bounds
    return SceneFootprint(
        scene_id=scene_id,
        image_path=f"{scene_id}.tif",
        raster_width=100,
        raster_height=100,
        raster_crs="EPSG:3857",
        source="test",
        polygon_raster_crs=polygon,
        polygon_pixel=polygon,
        bounds_pixel=(int(minx), int(miny), int(maxx), int(maxy)),
        area_pixels_estimated=float(polygon.area),
        valid_pixel_share_estimated=float(polygon.area) / 10000.0,
        warnings=[],
        raster_transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


if __name__ == "__main__":
    unittest.main()
