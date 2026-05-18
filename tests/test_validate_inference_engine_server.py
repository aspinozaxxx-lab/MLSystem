from __future__ import annotations

import unittest
from unittest.mock import patch

from scripts import validate_inference_engine_server as validate


class InferenceEngineServerValidationTests(unittest.TestCase):
    def test_pseudolabel_submit_helper_posts_request_to_mlsystem_api(self) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_post_json(url: str, payload: dict, **_kwargs):
            calls.append((url, payload))
            return {"run_id": payload["run_id"], "state": "queued"}

        request = {
            "run_id": "ie_pipeline_real_2_unit",
            "experiment_id": "ie_pipeline_real_2_unit",
            "model_ref": "model",
            "images_uri": "s3://bucket/images/",
        }
        with patch.object(validate, "_post_json", side_effect=fake_post_json):
            validate._submit_pseudolabel_run(mlsystem_api="http://mlsystem", token="token", request=request)
        url, payload = calls[0]
        self.assertEqual(url, "http://mlsystem/api/v1/pseudolabel-runs")
        self.assertEqual(payload, request)

    def test_manifest_scene_names_extracts_strings(self) -> None:
        with patch.object(validate, "_read_json", return_value={"scenes": [{"entry": "a.tif"}, {"name": "b.tif"}, "c.tif"]}):
            self.assertEqual(validate._manifest_scene_names(validate.Path("manifest.json"), limit=2), ["a.tif", "b.tif"])


if __name__ == "__main__":
    unittest.main()
