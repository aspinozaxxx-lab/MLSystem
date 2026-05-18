from __future__ import annotations

import unittest

import pytest

torch = pytest.importorskip("torch")

from mlsystem.src.train._losses import segmentation_loss_components


class TrainLossesTests(unittest.TestCase):
    def test_focal_dice_uses_configured_weights(self) -> None:
        logits = torch.tensor([[[[0.0, 1.0], [-1.0, 0.5]]]], dtype=torch.float32)
        target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32)
        components = segmentation_loss_components(logits, target, {"name": "focal_dice", "focal_weight": 0.25, "dice_weight": 0.75})
        expected = components["loss_focal"] * 0.25 + components["loss_dice"] * 0.75
        self.assertTrue(torch.allclose(components["loss_total"], expected))


if __name__ == "__main__":
    unittest.main()
