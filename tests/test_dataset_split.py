from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlsystem.src.data.dataset_split import (
    SceneObjectCount,
    clean_scene_list_file,
    count_objects_per_scene,
    filter_existing_scenes,
    index_image_files,
    read_scene_list,
    split_train_val_by_object_counts,
)


class DatasetSplitTests(unittest.TestCase):
    def test_read_scene_list_ignores_empty_lines_comments_and_keeps_first_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenes.txt"
            path.write_text("\n# comment\nscene_a.tif\tmetadata\n  scene_b  \n", encoding="utf-8")
            self.assertEqual(read_scene_list(path), ["scene_a.tif", "scene_b"])

    def test_filter_existing_scenes_matches_filename_stem_and_casefold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            images.mkdir()
            (images / "SCENE_A.tif").write_text("", encoding="utf-8")
            (images / "scene_b.tiff").write_text("", encoding="utf-8")
            result = filter_existing_scenes(["SCENE_A.tif", "scene_b", "missing"], index_image_files(images))
            self.assertEqual(result.existing_scenes, ["SCENE_A.tif", "scene_b"])
            self.assertEqual(result.missing_scenes, ["missing"])
            self.assertEqual(result.scene_to_image["scene_b"].name, "scene_b.tiff")

    def test_index_image_files_is_recursive_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            nested = images / "nested"
            nested.mkdir(parents=True)
            (nested / "scene_nested.tif").write_text("", encoding="utf-8")
            image_index = index_image_files(images)
            self.assertEqual([path.name for path in image_index["paths"]], ["scene_nested.tif"])
            result = filter_existing_scenes(["scene_nested"], image_index)
            self.assertEqual(result.existing_scenes, ["scene_nested"])

    def test_count_objects_per_scene_uses_feature_properties(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotation = root / "ann.geojson"
            annotation.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            self._feature("scene_a.tif"),
                            self._feature("scene_a.tif"),
                            self._feature("scene_b"),
                            self._feature("not_in_list"),
                        ],
                    }
                ),
                encoding="utf-8",
            )
            rows = count_objects_per_scene(
                ["scene_a.tif", "scene_b", "scene_c"],
                {"scene_a.tif": root / "scene_a.tif", "scene_b": root / "scene_b.tif", "scene_c": root / "scene_c.tif"},
                annotation,
                count_mode="property",
            )
            counts = {row.scene_name: row.object_count for row in rows}
            self.assertEqual(counts, {"scene_a.tif": 2, "scene_b": 1, "scene_c": 0})

    def test_split_is_deterministic_and_has_no_overlap(self) -> None:
        rows = [
            SceneObjectCount("a", None, 10, "property"),
            SceneObjectCount("b", None, 5, "property"),
            SceneObjectCount("c", None, 1, "property"),
            SceneObjectCount("d", None, 0, "property"),
            SceneObjectCount("e", None, 0, "property"),
        ]
        first = split_train_val_by_object_counts(rows, target_val_fraction=0.2, seed=7)
        second = split_train_val_by_object_counts(rows, target_val_fraction=0.2, seed=7)
        self.assertEqual([item.scene_name for item in first.train], [item.scene_name for item in second.train])
        self.assertEqual([item.scene_name for item in first.val], [item.scene_name for item in second.val])
        train_names = {item.scene_name for item in first.train}
        val_names = {item.scene_name for item in first.val}
        self.assertFalse(train_names & val_names)

    def test_split_summary_counts_files_objects_and_keeps_zero_scenes(self) -> None:
        rows = [
            SceneObjectCount("a", None, 3, "property"),
            SceneObjectCount("b", None, 0, "property"),
            SceneObjectCount("c", None, 0, "property"),
        ]
        split = split_train_val_by_object_counts(rows, target_val_fraction=0.34, seed=1)
        all_names = {item.scene_name for item in split.train + split.val}
        self.assertEqual(all_names, {"a", "b", "c"})
        self.assertEqual(split.summary["total_files"], 3)
        self.assertEqual(split.summary["total_objects"], 3)
        self.assertEqual(split.summary["train_files"] + split.summary["val_files"], 3)
        self.assertEqual(split.summary["train_objects"] + split.summary["val_objects"], 3)

    def test_object_balanced_split_never_leaves_validation_without_objects(self) -> None:
        rows = [
            SceneObjectCount("large", None, 200, "property"),
            SceneObjectCount("medium", None, 120, "property"),
            SceneObjectCount("small_a", None, 90, "property"),
            SceneObjectCount("small_b", None, 80, "property"),
            SceneObjectCount("zero_a", None, 0, "property"),
            SceneObjectCount("zero_b", None, 0, "property"),
        ]
        split = split_train_val_by_object_counts(rows, target_val_fraction=0.1, seed=12)
        self.assertGreater(split.summary["val_objects"], 0)
        self.assertGreater(split.summary["train_objects"], 0)
        self.assertFalse({item.scene_name for item in split.train} & {item.scene_name for item in split.val})

    def test_clean_scene_list_file_creates_backup_report_and_removes_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            (images / "scene_a.tif").write_text("", encoding="utf-8")
            scene_list = root / "deforestation.txt"
            scene_list.write_text("scene_a\nmissing_scene\n", encoding="utf-8")
            report = root / "removed.txt"
            result = clean_scene_list_file(scene_list, images, output_report=report, in_place=True, backup=True)
            self.assertEqual(read_scene_list(scene_list), ["scene_a"])
            self.assertIsNotNone(result.backup_path)
            self.assertTrue(result.backup_path.exists())
            self.assertTrue(report.exists())
            self.assertIn("missing_scene", report.read_text(encoding="utf-8"))

    @staticmethod
    def _feature(scene_name: str) -> dict:
        return {
            "type": "Feature",
            "properties": {"image_name": scene_name},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        }


if __name__ == "__main__":
    unittest.main()
