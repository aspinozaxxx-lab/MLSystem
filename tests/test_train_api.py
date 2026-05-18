from __future__ import annotations

import unittest

from mlsystem.src.train.api import list_supported_models


class TrainApiTests(unittest.TestCase):
    def test_public_api_lists_supported_models(self) -> None:
        names = {spec.name for spec in list_supported_models()}
        self.assertIn("tiny_unet_4ch", names)


if __name__ == "__main__":
    unittest.main()
