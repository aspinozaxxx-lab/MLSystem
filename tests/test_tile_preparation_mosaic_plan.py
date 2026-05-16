from __future__ import annotations

import unittest

from shapely.geometry import box

from mlsystem.src.tile_preparation.adjacency import build_scene_adjacency_index
from mlsystem.src.tile_preparation.footprint import SceneFootprint
from mlsystem.src.tile_preparation.mosaic_plan import build_tile_mosaic_plan
from mlsystem.src.tile_preparation.records import TileSampleRecord


class TilePreparationMosaicPlanTests(unittest.TestCase):
    def test_fully_inside_record_does_not_need_mosaic(self) -> None:
        anchor = _footprint("anchor", box(0, 0, 100, 100))
        record = _record(metadata={"footprint_fully_inside": True})
        index = build_scene_adjacency_index({"anchor": anchor})

        plan = build_tile_mosaic_plan(record, anchor, {"anchor": anchor}, index)

        self.assertFalse(plan.needed)
        self.assertEqual(plan.reason, "fully_inside_footprint")
        self.assertEqual(plan.candidate_scene_ids, [])

    def test_boundary_without_neighbor_does_not_need_mosaic(self) -> None:
        anchor = _footprint("anchor", box(0, 0, 50, 100))
        record = _record(metadata={"footprint_boundary": True})
        index = build_scene_adjacency_index({"anchor": anchor})

        plan = build_tile_mosaic_plan(record, anchor, {"anchor": anchor}, index)

        self.assertFalse(plan.needed)
        self.assertEqual(plan.reason, "no_neighbor_for_gap")
        self.assertGreater(plan.estimated_gap_share, 0.0)

    def test_boundary_neighbor_covering_gap_needs_mosaic(self) -> None:
        anchor = _footprint("anchor", box(0, 0, 50, 100))
        neighbor = _footprint("neighbor", box(50, 0, 100, 100))
        record = _record(metadata={"footprint_boundary": True})
        footprints = {"anchor": anchor, "neighbor": neighbor}
        index = build_scene_adjacency_index(footprints)

        plan = build_tile_mosaic_plan(record, anchor, footprints, index)

        self.assertTrue(plan.needed)
        self.assertEqual(plan.reason, "neighbor_covers_gap")
        self.assertEqual(plan.candidate_scene_ids, ["neighbor"])
        self.assertGreater(plan.estimated_neighbor_cover_share, 0.0)

    def test_overlap_does_not_replace_fully_valid_anchor_tile(self) -> None:
        anchor = _footprint("anchor", box(0, 0, 100, 100))
        neighbor = _footprint("neighbor", box(50, 0, 150, 100))
        record = _record(metadata={"footprint_fully_inside": True})
        footprints = {"anchor": anchor, "neighbor": neighbor}
        index = build_scene_adjacency_index(footprints)

        plan = build_tile_mosaic_plan(record, anchor, footprints, index)

        self.assertFalse(plan.needed)
        self.assertEqual(plan.reason, "fully_inside_footprint")


def _record(*, metadata: dict[str, object] | None = None) -> TileSampleRecord:
    return TileSampleRecord(
        scene_id="anchor",
        image_path="anchor.tif",
        x=0,
        y=0,
        width=100,
        height=100,
        tile_size=100,
        stride=100,
        kind="negative",
        positive_pixels=0,
        source="test",
        metadata=dict(metadata or {}),
    )


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
