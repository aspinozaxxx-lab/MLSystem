from __future__ import annotations

import unittest

from pydantic import ValidationError

from mlsystem.src.inference_pipeline.api import PseudolabelRunRequest


class InferencePipelineContractsTests(unittest.TestCase):
    def test_pseudolabel_request_minimal_fields(self) -> None:
        request = PseudolabelRunRequest(experiment_id="exp", model_ref="model", images_uri="s3://bucket/images/")
        self.assertEqual(request.experiment_id, "exp")
        self.assertIsNone(request.class_name)
        self.assertFalse(request.dry_run)

    def test_pseudolabel_request_rejects_public_tuning_knobs(self) -> None:
        with self.assertRaises(ValidationError):
            PseudolabelRunRequest(experiment_id="exp", model_ref="model", images_uri="s3://bucket/images/", tile_size=512)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
