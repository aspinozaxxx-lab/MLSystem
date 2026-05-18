from __future__ import annotations

import inspect
import unittest

from mlsystem.src.tile_preparation.api import build_datasets, train_dataloader, val_dataloader


class TilePreparationApiBoundaryTests(unittest.TestCase):
    def test_public_api_has_minimal_entrypoints(self) -> None:
        self.assertEqual(list(inspect.signature(build_datasets).parameters), ["request"])
        self.assertIn("batch_size", inspect.signature(train_dataloader).parameters)
        self.assertIn("batch_size", inspect.signature(val_dataloader).parameters)


if __name__ == "__main__":
    unittest.main()
