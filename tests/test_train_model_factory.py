from __future__ import annotations

import unittest

import pytest

torch = pytest.importorskip("torch")

from mlsystem.src.train._checkpoints import load_initial_checkpoint
from mlsystem.src.train._models import TinyUNet, build_model, configure_dropout
from mlsystem.src.train._optimizers import build_optimizer
from mlsystem.src.train._schedulers import build_scheduler


class TrainModelFactoryTests(unittest.TestCase):
    def test_tiny_unet_forward_shape(self) -> None:
        model = build_model("tiny_unet_4ch", 4, 1, 4)
        out = model(torch.zeros((2, 4, 16, 16), dtype=torch.float32))
        self.assertEqual(tuple(out.shape), (2, 1, 16, 16))

    def test_optimizer_scheduler_and_dropout(self) -> None:
        model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Dropout2d(p=0.5))
        self.assertEqual(configure_dropout(model, 0.15), 1)
        optimizer = build_optimizer(model, {"optimizer": "adam", "learning_rate": 1e-4, "weight_decay": 0.01})
        self.assertIsInstance(optimizer, torch.optim.Adam)
        scheduler = build_scheduler(optimizer, {"scheduler": {"name": "cosine", "t_max": 3}}, epochs=5)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_load_initial_checkpoint_restores_model_weights(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            source = TinyUNet(in_channels=4, out_channels=1, base_channels=4)
            with torch.no_grad():
                for param in source.parameters():
                    param.fill_(0.25)
            torch.save({"model_state_dict": source.state_dict()}, path)
            target = TinyUNet(in_channels=4, out_channels=1, base_channels=4)
            info = load_initial_checkpoint(target, path, device=torch.device("cpu"), strict=True)
            self.assertEqual(info["missing_keys"], [])
            for param in target.parameters():
                self.assertTrue(torch.allclose(param, torch.full_like(param, 0.25)))


if __name__ == "__main__":
    unittest.main()
