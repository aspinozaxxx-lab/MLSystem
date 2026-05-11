from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frontend.app.training_report.dataset_inventory import ClassSpec, count_geojson_objects, count_scene_files, inventory_class


class TrainingReportDatasetInventoryTests(unittest.TestCase):
    def test_txt_with_files_counts_one_scene_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            class_dir = root / "Озера"
            class_dir.mkdir()
            scenes = class_dir / "lakes.txt"
            scenes.write_text("a.tif\nb.tiff\n# skip\n\n", encoding="utf-8")
            count, source, warnings = count_scene_files([scenes], class_dir=class_dir, mlmarkup_path=root)
            self.assertEqual(count, 2)
            self.assertEqual(source, "txt_files")
            self.assertEqual(warnings, [])

    def test_txt_with_folder_counts_tif_inside_folder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            class_dir = root / "Озера"
            class_dir.mkdir()
            folder = class_dir / "scene_folder"
            folder.mkdir()
            (folder / "one.tif").write_text("", encoding="utf-8")
            (folder / "two.tiff").write_text("", encoding="utf-8")
            (folder / "ignore.jpg").write_text("", encoding="utf-8")
            scenes = class_dir / "lakes.txt"
            scenes.write_text("scene_folder\n", encoding="utf-8")
            count, source, warnings = count_scene_files([scenes], class_dir=class_dir, mlmarkup_path=root)
            self.assertEqual(count, 2)
            self.assertEqual(source, "txt_dirs_expanded")
            self.assertEqual(warnings, [])

    def test_geojson_object_count_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            class_dir = root / "Озера"
            class_dir.mkdir()
            geojson = class_dir / "lakes.geojson"
            geojson.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {}},
                            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [1, 1]}, "properties": {}},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (class_dir / "lakes.txt").write_text("a.tif\n", encoding="utf-8")
            self.assertEqual(count_geojson_objects(geojson), 2)
            payload = inventory_class(root, ClassSpec("Озера", "lakes"))
            self.assertEqual(payload["objects_count"], 2)
            self.assertEqual(payload["scenes_count"], 1)
            self.assertEqual(payload["scenes_count_source"], "txt_files")


if __name__ == "__main__":
    unittest.main()

