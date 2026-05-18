from __future__ import annotations

import unittest

from pydantic import ValidationError

from mlsystem.src.train_pipeline.api import JobSpec


class JobSchemaTests(unittest.TestCase):
    def test_minimal_train_job_defaults(self) -> None:
        job = JobSpec.model_validate({"job_id": "exp_001", "task": "train"})
        self.assertEqual(job.schema_version, 1)
        self.assertEqual(job.params, {})
        self.assertEqual(job.evaluate, {})
        self.assertFalse(job.resources.requires_gpu)

    def test_invalid_job_id_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            JobSpec.model_validate({"job_id": "bad/id", "task": "train"})

    def test_existing_task_enum_kept(self) -> None:
        job = JobSpec.model_validate({"job_id": "predict-001", "task": "predict"})
        self.assertEqual(job.task, "predict")


if __name__ == "__main__":
    unittest.main()
