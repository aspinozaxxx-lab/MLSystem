from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mlsystem.src.io_utils import write_json
from mlsystem.src.real_train import SceneMatch, _load_prepared_dataset_split


class RealTrainPreparedSplitTests(unittest.TestCase):
    def test_load_prepared_dataset_split_matches_manifest_to_current_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            write_json(
                manifest,
                {
                    "split_strategy": "object_balanced",
                    "train_scenes": [{"entry": "scene_a.tif", "name": "scene_a.tif", "key": "images/scene_a.tif"}],
                    "val_scenes": [{"entry": "scene_b.tif", "name": "scene_b.tif", "key": "images/scene_b.tif"}],
                    "split_summary": {"total_files": 2},
                },
            )
            matches = [
                SceneMatch(entry="scene_a.tif", name="scene_a.tif", key="images/scene_a.tif", score=1.0),
                SceneMatch(entry="scene_b.tif", name="scene_b.tif", key="images/scene_b.tif", score=1.0),
            ]
            job = SimpleNamespace(preprocess={"prepared_dataset_manifest": str(manifest)})
            split = _load_prepared_dataset_split(root, matches, job)
            self.assertIsNotNone(split)
            train, val, metadata = split
            self.assertEqual([item.name for item in train], ["scene_a.tif"])
            self.assertEqual([item.name for item in val], ["scene_b.tif"])
            self.assertEqual(metadata["split_strategy"], "object_balanced")

    def test_load_prepared_dataset_split_fails_on_lost_scene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dataset_manifest.json"
            write_json(
                manifest,
                {
                    "train_scenes": [{"entry": "scene_a.tif", "name": "scene_a.tif", "key": "images/scene_a.tif"}],
                    "val_scenes": [{"entry": "scene_b.tif", "name": "scene_b.tif", "key": "images/scene_b.tif"}],
                },
            )
            matches = [
                SceneMatch(entry="scene_a.tif", name="scene_a.tif", key="images/scene_a.tif", score=1.0),
                SceneMatch(entry="scene_b.tif", name="scene_b.tif", key="images/scene_b.tif", score=1.0),
                SceneMatch(entry="scene_c.tif", name="scene_c.tif", key="images/scene_c.tif", score=1.0),
            ]
            job = SimpleNamespace(preprocess={"prepared_dataset_manifest": str(manifest)})
            with self.assertRaisesRegex(RuntimeError, "lost 1 matched scenes"):
                _load_prepared_dataset_split(root, matches, job)


if __name__ == "__main__":
    unittest.main()
