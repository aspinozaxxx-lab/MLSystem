from __future__ import annotations

import unittest
from pathlib import Path


class RealTrainIntegrationTests(unittest.TestCase):
    def test_real_train_uses_tile_preparation_directly(self) -> None:
        text = Path("mlsystem/src/real_train.py").read_text(encoding="utf-8")
        self.assertIn("TilePreparationFacade", text)
        self.assertIn("TrainingTileDataset", text)
        self.assertIn("iter_dataset_batches", text)
        self.assertIn("TilePreparationFacade.build_datasets", text)
        self.assertNotIn("resolve_tile_preparation_config", text)
        self.assertNotIn("from_scene_list", text)
        self.assertNotIn("data.virtual_tile_sampling", text)
        self.assertNotIn("from .data.virtual_tile_sampling import", text)
        self.assertNotIn("FastAPI", text)
        self.assertNotIn("/api/debug/annotated-tile-report/preview", text)
        self.assertNotIn("def _apply_train_augmentations", text)
        self.assertNotIn("build_virtual_train_records", text)
        self.assertNotIn("build_validation_records", text)


if __name__ == "__main__":
    unittest.main()
