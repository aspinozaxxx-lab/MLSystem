from __future__ import annotations

import unittest

from mlsystem.src.train_pipeline.api import StageStartRequest
from mlsystem.src.api.security import mask_secrets, mask_text


class ApiModelsTests(unittest.TestCase):
    def test_stage_start_request_defaults(self) -> None:
        request = StageStartRequest(experiment_config={"experiment_id": "unit"})
        self.assertEqual(request.status_root, "/data/mlsystem/runs")
        self.assertEqual(request.source, "api")

    def test_mask_secrets_masks_nested_values(self) -> None:
        payload = {
            "token": "abc",
            "nested": {"AWS_SECRET_ACCESS_KEY": "secret", "safe": "value"},
            "items": [{"password": "pw"}],
        }
        masked = mask_secrets(payload)
        self.assertEqual(masked["token"], "***")
        self.assertEqual(masked["nested"]["AWS_SECRET_ACCESS_KEY"], "***")
        self.assertEqual(masked["nested"]["safe"], "value")
        self.assertEqual(masked["items"][0]["password"], "***")

    def test_mask_text_masks_secret_env_values(self) -> None:
        import os

        old_value = os.environ.get("MLSYSTEM_API_TOKEN")
        os.environ["MLSYSTEM_API_TOKEN"] = "very-secret-token"
        try:
            self.assertEqual(mask_text("failed with very-secret-token"), "failed with ***")
        finally:
            if old_value is None:
                os.environ.pop("MLSYSTEM_API_TOKEN", None)
            else:
                os.environ["MLSYSTEM_API_TOKEN"] = old_value


if __name__ == "__main__":
    unittest.main()
