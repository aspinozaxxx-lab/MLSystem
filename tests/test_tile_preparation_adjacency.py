from __future__ import annotations

import unittest

from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from mlsystem.src.tile_preparation.adjacency import build_scene_adjacency_index
from mlsystem.src.tile_preparation.footprint import SceneFootprint


class TilePreparationAdjacencyTests(unittest.TestCase):
    def test_adjacent_sides_and_overlap_are_classified(self) -> None:
        footprints = {
            "anchor": _footprint("anchor", box(0, 0, 100, 100)),
            "left": _footprint("left", box(-100, 0, 0, 100)),
            "right": _footprint("right", box(100, 0, 200, 100)),
            "top": _footprint("top", box(0, 100, 100, 200)),
            "bottom": _footprint("bottom", box(0, -100, 100, 0)),
            "overlap": _footprint("overlap", box(50, 0, 150, 100)),
            "far": _footprint("far", box(1000, 0, 1100, 100)),
        }

        index = build_scene_adjacency_index(footprints)
        by_neighbor = {item.neighbor_scene_id: item for item in index.neighbors_for("anchor")}

        self.assertEqual(by_neighbor["left"].relation, "touch")
        self.assertEqual(by_neighbor["left"].side, "left")
        self.assertEqual(by_neighbor["right"].side, "right")
        self.assertEqual(by_neighbor["top"].side, "top")
        self.assertEqual(by_neighbor["bottom"].side, "bottom")
        self.assertEqual(by_neighbor["overlap"].relation, "overlap")
        self.assertEqual(by_neighbor["overlap"].side, "overlap")
        self.assertNotIn("far", by_neighbor)

    def test_different_crs_neighbor_is_transformed(self) -> None:
        try:
            from pyproj import Transformer
        except Exception:  # noqa: BLE001
            self.skipTest("pyproj is required")

        anchor = _footprint("anchor", box(0, 0, 1000, 1000), crs="EPSG:3857")
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        neighbor_polygon = shapely_transform(transformer.transform, box(100, 100, 900, 900))
        neighbor = _footprint("neighbor", neighbor_polygon, crs="EPSG:4326")

        index = build_scene_adjacency_index({"anchor": anchor, "neighbor": neighbor})
        neighbors = index.neighbors_for("anchor")

        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0].neighbor_scene_id, "neighbor")
        self.assertEqual(neighbors[0].relation, "overlap")


def _footprint(scene_id: str, polygon, *, crs: str = "EPSG:3857") -> SceneFootprint:
    minx, miny, maxx, maxy = polygon.bounds
    return SceneFootprint(
        scene_id=scene_id,
        image_path=f"{scene_id}.tif",
        raster_width=1000,
        raster_height=1000,
        raster_crs=crs,
        source="test",
        polygon_raster_crs=polygon,
        polygon_pixel=polygon,
        bounds_pixel=(int(minx), int(miny), int(maxx), int(maxy)),
        area_pixels_estimated=float(polygon.area),
        valid_pixel_share_estimated=1.0,
        warnings=[],
        raster_transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


if __name__ == "__main__":
    unittest.main()
