from __future__ import annotations

import unittest
from pathlib import Path


class FrontendGatewayConfigTests(unittest.TestCase):
    def test_nginx_template_protects_admin_ui_locations(self) -> None:
        text = Path("ansible/roles/frontend/templates/frontend-nginx.conf.j2").read_text(encoding="utf-8")
        for location in ["location /airflow/", "location /mlflow/", "location /rabbitmq/", "location /minio-browser/"]:
            self.assertIn(location, text)
        self.assertGreaterEqual(text.count("auth_request /auth/proxy-check;"), 4)
        self.assertIn('proxy_set_header Authorization "Basic {{ frontend_rabbitmq_proxy_basic_auth }}"', text)
        self.assertIn("proxy_set_header X-Forwarded-Prefix /mlflow", text)
        self.assertIn("proxy_set_header X-Forwarded-Prefix /rabbitmq", text)

    def test_compose_binds_admin_raw_ports_to_localhost(self) -> None:
        text = Path("deploy/docker-compose.gpu.yml").read_text(encoding="utf-8")
        self.assertIn('"${AIRFLOW_LISTEN_HOST:-127.0.0.1}:${AIRFLOW_PORT:-8081}:8080"', text)
        self.assertIn('"${MLFLOW_LISTEN_HOST:-127.0.0.1}:${MLFLOW_PORT:-5000}:5000"', text)
        self.assertIn('"${MINIO_CONSOLE_LISTEN_HOST:-127.0.0.1}:${MINIO_CONSOLE_PORT:-9001}:9001"', text)
        self.assertIn('"${RABBITMQ_MANAGEMENT_LISTEN_HOST:-127.0.0.1}:${RABBITMQ_MANAGEMENT_PORT:-15672}:15672"', text)

    def test_inference_engine_deploy_preserves_mlsystem_runtime_config(self) -> None:
        text = Path("ansible/playbooks/deploy_inference_engine_code.yml").read_text(encoding="utf-8")
        self.assertIn("pipeline.server.yaml.j2", text)
        self.assertIn("mlsystem/configs/pipeline.server.yaml", text)
        self.assertIn("up -d --force-recreate --no-deps", text)
        self.assertIn("mlsystem-api airflow-webserver airflow-scheduler airflow-triggerer", text)


if __name__ == "__main__":
    unittest.main()
