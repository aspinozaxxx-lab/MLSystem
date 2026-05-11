from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from scripts import validate_inference_engine_server as validate


class InferenceEngineServerValidationTests(unittest.TestCase):
    def test_airflow_trigger_helper_uses_real_dag_run(self) -> None:
        commands: list[str] = []

        def fake_run_text(command: str, **_kwargs):
            commands.append(command)
            return "created"

        config = {
            "experiment_id": "ie_airflow_real_2_unit",
            "pseudolabel": {"source": "inference_engine", "max_scenes": 2},
        }
        with patch.object(validate, "_run_text", side_effect=fake_run_text):
            validate._trigger_airflow_dag_run("ie_airflow_real_2_unit", config)
        command = commands[0]
        self.assertIn("airflow dags trigger mlsystem_experiment_pipeline", command)
        self.assertIn("--run-id ie_airflow_real_2_unit", command)
        self.assertIn("--conf", command)
        self.assertIn(json.dumps(config, ensure_ascii=False, separators=(",", ":")), command)

    def test_airflow_validation_config_is_inference_engine_only(self) -> None:
        config = validate._mlsystem_config("ie_airflow_real_2_unit", max_scenes=2)
        self.assertEqual(config["train"], {"enabled": False})
        self.assertEqual(config["predict"], {"enabled": False})
        self.assertEqual(config["pseudolabel"]["source"], "inference_engine")
        self.assertEqual(config["pseudolabel"]["max_scenes"], 2)


if __name__ == "__main__":
    unittest.main()
