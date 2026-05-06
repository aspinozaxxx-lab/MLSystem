from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()

