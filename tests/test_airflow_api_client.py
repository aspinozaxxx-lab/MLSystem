from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from mlsystem.src.orchestration.airflow_api_client import AirflowApiStageError, MLSystemApiClient


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class AirflowApiClientTests(unittest.TestCase):
    def test_client_start_stage_posts_json(self) -> None:
        with patch("urllib.request.urlopen", return_value=_Response({"job_id": "job1"})) as opened:
            client = MLSystemApiClient("http://api", token="token")
            response = client.start_stage("run1", "inventory_scenes", {"secret": "value"})
        self.assertEqual(response["job_id"], "job1")
        request = opened.call_args.args[0]
        self.assertEqual(request.headers["Authorization"], "Bearer token")
        self.assertIn(b'"secret": "***"', request.data)

    def test_client_raises_clear_error_on_connection_failure(self) -> None:
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")):
            client = MLSystemApiClient("http://api")
            with self.assertRaises(AirflowApiStageError):
                client.job_status("job1")


if __name__ == "__main__":
    unittest.main()
