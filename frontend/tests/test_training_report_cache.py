from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frontend.app.config import FrontendConfig
from frontend.app.training_report.cache import TrainingReportCache
from frontend.app.training_report.collector import TrainingReportCollector


class TrainingReportCacheTests(unittest.TestCase):
    def test_atomic_write_and_read_index(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = TrainingReportCache(Path(td))
            payload = {"status": "ok", "classes": [{"class_name": "Озера"}]}
            cache.write_index(payload)
            self.assertEqual(cache.read_index(), payload)
            self.assertTrue(cache.index_path.exists())

    def test_incremental_state_tracks_known_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = FrontendConfig(
                session_secret="unit",
                training_report_root=Path(td) / "cache",
                mlmarkup_path=Path(td) / "missing",
                training_report_background_enabled=False,
            )
            cache = TrainingReportCache(config.training_report_root)
            collector = TrainingReportCollector(config, cache)
            fake_runs = [
                {"run_id": "r2", "experiment_id": "1", "class_name": "Озера", "class_slug": "lakes", "pixel_f1": 0.2},
                {"run_id": "r1", "experiment_id": "1", "class_name": "Абразия", "class_slug": "abrasion", "pixel_f1": 0.1},
            ]
            with patch("frontend.app.training_report.collector.MLflowReader.read_class_runs", return_value=fake_runs):
                collector.collect()
            self.assertEqual(cache.read_state()["known_run_ids"], ["r1", "r2"])


if __name__ == "__main__":
    unittest.main()

