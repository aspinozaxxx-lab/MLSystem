from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from frontend.app.config import FrontendConfig
from frontend.app.main import create_app


def test_config(tmp: Path) -> FrontendConfig:
    return FrontendConfig(
        username="mluser",
        password="qazwsxedc",
        session_secret="unit-secret",
        upload_root=tmp / "uploads",
        status_root=tmp / "status",
        api_base_url="http://mlsystem-api:8088",
        api_token="server-token",
    )


class FrontendAuthTests(unittest.TestCase):
    def test_login_success_and_logout(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            client = TestClient(create_app(test_config(Path(td))))
            response = client.post("/login", data={"username": "mluser", "password": "qazwsxedc"}, follow_redirects=False)
            self.assertEqual(response.status_code, 303)
            self.assertEqual(response.headers["location"], "/")
            protected = client.get("/")
            self.assertEqual(protected.status_code, 200)
            logout = client.post("/logout", follow_redirects=False)
            self.assertEqual(logout.status_code, 303)

    def test_login_fail_and_protected_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            client = TestClient(create_app(test_config(Path(td))))
            protected = client.get("/", follow_redirects=False)
            self.assertEqual(protected.status_code, 303)
            response = client.post("/login", data={"username": "mluser", "password": "wrong"})
            self.assertEqual(response.status_code, 401)

    def test_head_login_for_proxy_validation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            client = TestClient(create_app(test_config(Path(td))))
            response = client.head("/login")
            self.assertEqual(response.status_code, 200)

    def test_proxy_check_requires_valid_session(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            client = TestClient(create_app(test_config(Path(td))))
            denied = client.get("/auth/proxy-check")
            self.assertEqual(denied.status_code, 401)
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})
            allowed = client.get("/auth/proxy-check")
            self.assertEqual(allowed.status_code, 204)
            self.assertEqual(allowed.headers["x-remote-user"], "mluser")
            self.assertEqual(allowed.headers["x-mlsystem-user"], "mluser")

    def test_home_has_admin_gateway_cards(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            client = TestClient(create_app(test_config(Path(td))))
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("Airflow", response.text)
            self.assertIn("MLflow", response.text)
            self.assertIn("MinIO artifacts", response.text)
            self.assertIn("Очереди RabbitMQ", response.text)
            self.assertIn("Мониторинг MLSystem", response.text)
            self.assertIn("Отчет об обучении", response.text)
            self.assertIn("Лучшие модели по классам и история экспериментов", response.text)
            self.assertIn("Grafana", response.text)
            self.assertIn("Prometheus", response.text)
            self.assertNotIn('/airflow/"', response.text)
            self.assertIn('/mlflow/"', response.text)
            self.assertIn('/minio-browser/"', response.text)
            self.assertIn('/rabbitmq/"', response.text)
            self.assertIn('/grafana/"', response.text)
            self.assertIn('/prometheus/"', response.text)
            self.assertIn('/grafana/d/mlsystem-overview/mlsystem-overview?orgId=1&amp;kiosk', response.text)

    def test_queue_metrics_proxy_uses_inference_engine_api(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = test_config(Path(td))
            config = FrontendConfig(**{**config.__dict__, "inference_engine_api_url": "http://ie:8095", "inference_engine_api_token": "ie-token"})
            client = TestClient(create_app(config))
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})

            class FakeResponse:
                status = 200

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def read(self, *_args) -> bytes:
                    return json.dumps(
                        {
                            "metrics": [
                                {"name": "ie.tile.infer", "messages_ready": 3, "messages_unacked": 2, "consumers": 1},
                                {"name": "ie.dead_letter", "messages_ready": 0, "messages_unacked": 0, "consumers": 0},
                            ]
                        }
                    ).encode("utf-8")

            seen = {}

            def fake_urlopen(request, timeout=8):
                seen["url"] = request.full_url
                seen["auth"] = request.headers.get("Authorization")
                return FakeResponse()

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                response = client.get("/api/inference-engine/queues")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(seen["url"], "http://ie:8095/queues")
            self.assertEqual(seen["auth"], "Bearer ie-token")
            self.assertEqual(response.json()["ready"], 3)
            self.assertEqual(response.json()["unacked"], 2)

    def test_services_status_uses_internal_upstreams(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = test_config(Path(td))
            config = FrontendConfig(**{**config.__dict__, "inference_engine_api_url": "http://ie:8095", "inference_engine_api_token": "ie-token"})
            client = TestClient(create_app(config))
            client.post("/login", data={"username": "mluser", "password": "qazwsxedc"})

            class FakeResponse:
                status = 200

                def __init__(self, payload: dict[str, object] | str):
                    self.payload = payload

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return None

                def read(self, *_args) -> bytes:
                    if isinstance(self.payload, str):
                        return self.payload.encode("utf-8")
                    return json.dumps(self.payload).encode("utf-8")

            seen: list[str] = []

            def fake_urlopen(request, timeout=8):
                url = getattr(request, "full_url", request)
                seen.append(url)
                if str(url).endswith("/queues"):
                    return FakeResponse({"metrics": [{"name": "ie.tile.infer", "messages_ready": 1, "messages_unacked": 0, "consumers": 1}]})
                if str(url).endswith("/health"):
                    return FakeResponse({"status": "ok"})
                return FakeResponse("ok")

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                response = client.get("/api/services/status")
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["services"]["rabbitmq"]["ready"], 1)
            self.assertFalse(any("airflow" in url.lower() for url in seen))
            self.assertIn("http://mlflow:5000/mlflow/health", seen)
            self.assertIn("http://grafana:3000/api/health", seen)
            self.assertIn("http://prometheus:9090/prometheus/-/ready", seen)


if __name__ == "__main__":
    unittest.main()
