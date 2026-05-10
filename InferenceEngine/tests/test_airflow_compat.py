from __future__ import annotations

import json
import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mlsystem.src.pipeline.stages.context import StageContext
from mlsystem.src.pipeline.stages.vectorize_pseudolabel import run as run_vectorize


class _Store:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def update_summary(self, **_updates):
        return {}


class AirflowCompatibilityTests(unittest.TestCase):
    def test_vectorize_stage_is_validate_only_for_inference_engine_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "exp.accepted.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
            (run_dir / "vectorization_summary.json").write_text(json.dumps({"source": "inference_engine", "pseudolabel": {"accepted_objects": 0}}), encoding="utf-8")
            ctx = StageContext(
                stage_id="vectorize_pseudolabel",
                run_id="run",
                config=SimpleNamespace(experiment_id="exp", pseudolabel={"enabled": True, "source": "inference_engine"}, smoke=False, postprocess={}),
                raw_conf={},
                status_dir=run_dir,
                store=_Store(run_dir),
                logger=logging.getLogger("test"),
            )
            report = run_vectorize(ctx)
            self.assertEqual(report.status, "success")
            self.assertEqual(report.counters["vectorization_mode"], "inference_engine_validate_only")


if __name__ == "__main__":
    unittest.main()
