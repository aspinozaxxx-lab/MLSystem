from __future__ import annotations

import unittest

try:
    from shapely.geometry import Polygon, box

    from mlsystem.src.tile_preparation.geometry_index import GeometryWindowIndex, filter_geometries_linear

    HAS_SHAPELY = True
except Exception:  # noqa: BLE001
    HAS_SHAPELY = False


@unittest.skipUnless(HAS_SHAPELY, "shapely is required")
class TilePreparationGeometryIndexTests(unittest.TestCase):
    def test_strtree_query_matches_legacy_linear_filter(self) -> None:
        geometries = [
            box(0, 0, 10, 10),
            box(20, 20, 30, 30),
            box(5, 5, 25, 25),
            Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)]),
        ]
        window = box(8, 8, 22, 22)
        index = GeometryWindowIndex.build(geometries)

        expected = filter_geometries_linear(geometries, window)
        actual = index.query(window)

        self.assertEqual([geom.wkt for geom in actual], [geom.wkt for geom in expected])

    def test_empty_window_returns_empty_list(self) -> None:
        geometries = [box(0, 0, 10, 10), box(20, 20, 30, 30)]
        index = GeometryWindowIndex.build(geometries)
        self.assertEqual(index.query(box(100, 100, 110, 110)), [])

    def test_invalid_geometries_do_not_break_index(self) -> None:
        invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1), (0, 0)])
        geometries = [invalid, box(10, 10, 20, 20)]
        index = GeometryWindowIndex.build(geometries)
        self.assertEqual([geom.wkt for geom in index.query(box(12, 12, 13, 13))], [geometries[1].wkt])


if __name__ == "__main__":
    unittest.main()
