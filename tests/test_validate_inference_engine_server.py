from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import validate_inference_engine_server as validate


class InferenceEngineServerValidationTests(unittest.TestCase):
    def test_pipeline_submit_helper_posts_trace_to_mlsystem_api(self) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_post_json(url: str, payload: dict, **_kwargs):
            calls.append((url, payload))
            return {"run_id": payload["trace"]["experiment_id"], "state": "queued"}

        config = {
            "experiment_id": "ie_pipeline_real_2_unit",
            "pseudolabel": {"source": "inference_engine", "max_scenes": 2},
        }
        with patch.object(validate, "_post_json", side_effect=fake_post_json):
            validate._submit_pipeline_run(mlsystem_api="http://mlsystem", token="token", trace=config)
        url, payload = calls[0]
        self.assertEqual(url, "http://mlsystem/api/v1/pipeline-runs")
        self.assertEqual(payload["trace"], config)
        self.assertFalse(payload["dry_run"])

    def test_pipeline_validation_config_is_inference_engine_only(self) -> None:
        config = validate._mlsystem_config("ie_pipeline_real_2_unit", max_scenes=2)
        self.assertEqual(config["train"], {"enabled": False})
        self.assertEqual(config["predict"], {"enabled": False})
        self.assertEqual(config["pseudolabel"]["source"], "inference_engine")
        self.assertEqual(config["pseudolabel"]["max_scenes"], 2)


if __name__ == "__main__":
    unittest.main()
