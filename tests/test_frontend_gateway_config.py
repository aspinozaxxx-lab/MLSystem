from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class FrontendGatewayConfigTests(unittest.TestCase):
    def test_nginx_template_protects_admin_ui_locations(self) -> None:
        text = Path("ansible/roles/frontend/templates/frontend-nginx.conf.j2").read_text(encoding="utf-8")
        for location in [
            "location /mlflow/",
            "location /rabbitmq/",
            "location /grafana/",
            "location /prometheus/",
        ]:
            self.assertIn(location, text)
        self.assertGreaterEqual(text.count("auth_request /auth/proxy-check;"), 5)
        self.assertIn('proxy_set_header Authorization "Basic {{ frontend_rabbitmq_proxy_basic_auth }}"', text)
        self.assertIn("proxy_set_header X-Forwarded-Prefix /mlflow", text)
        self.assertIn("proxy_set_header X-Forwarded-Prefix /rabbitmq", text)
        self.assertIn("proxy_set_header X-Forwarded-Prefix /minio", text)
        self.assertIn('sub_filter \'<base href="/"/>\' \'<base href="/minio/"/>\';', text)
        self.assertIn("proxy_set_header X-Forwarded-Prefix /grafana", text)
        self.assertIn("proxy_set_header X-Forwarded-Prefix /prometheus", text)
        minio_block = text.split("location /minio/ {", 1)[1].split("location /grafana/", 1)[0]
        self.assertNotIn("auth_request /auth/proxy-check;", minio_block)
        self.assertNotIn("X-MLSystem-User", minio_block)

    def test_compose_binds_admin_raw_ports_to_localhost(self) -> None:
        compose = yaml.safe_load(Path("deploy/docker-compose.gpu.yml").read_text(encoding="utf-8"))
        services = compose["services"]
        self.assertIn("${MLFLOW_LISTEN_HOST:-127.0.0.1}:${MLFLOW_PORT:-5000}:5000", services["mlflow"]["ports"])
        self.assertIn("${MINIO_CONSOLE_LISTEN_HOST:-127.0.0.1}:${MINIO_CONSOLE_PORT:-9001}:9001", services["minio"]["ports"])
        self.assertEqual(services["minio"]["environment"]["MINIO_BROWSER_REDIRECT_URL"], "${MINIO_BROWSER_REDIRECT_URL:-http://31.192.104.147/minio/}")
        self.assertIn("MINIO_IMAGES_CONSOLE_USER", services["minio-init"]["environment"])
        self.assertIn("MINIO_IMAGES_CONSOLE_PASSWORD", services["minio-init"]["environment"])
        self.assertIn("MINIO_KANOPUS_READER_USER", services["minio-init"]["environment"])
        self.assertIn("MINIO_KANOPUS_READER_POLICY", services["minio-init"]["environment"])
        self.assertIn("${RABBITMQ_MANAGEMENT_LISTEN_HOST:-127.0.0.1}:${RABBITMQ_MANAGEMENT_PORT:-15672}:15672", services["rabbitmq"]["ports"])
        self.assertIn("${GRAFANA_LISTEN_HOST:-127.0.0.1}:${GRAFANA_PORT:-3000}:3000", services["grafana"]["ports"])
        self.assertIn("${PROMETHEUS_LISTEN_HOST:-127.0.0.1}:${PROMETHEUS_PORT:-9090}:9090", services["prometheus"]["ports"])
        self.assertIn("${RABBITMQ_PROMETHEUS_LISTEN_HOST:-127.0.0.1}:${RABBITMQ_PROMETHEUS_PORT:-15692}:15692", services["rabbitmq"]["ports"])

    def test_mlsystem_api_shm_size_supports_dataloader_prefetch(self) -> None:
        compose = yaml.safe_load(Path("deploy/docker-compose.gpu.yml").read_text(encoding="utf-8"))
        self.assertEqual(compose["services"]["mlsystem-api"].get("shm_size"), "${MLSYSTEM_API_SHM_SIZE:-1gb}")

    def test_inference_engine_deploy_preserves_mlsystem_runtime_config(self) -> None:
        text = Path("ansible/playbooks/deploy_inference_engine_code.yml").read_text(encoding="utf-8")
        self.assertIn("pipeline.server.yaml.j2", text)
        self.assertIn("mlsystem/configs/pipeline.server.yaml", text)
        self.assertIn("up -d --force-recreate --no-deps", text)
        self.assertIn("mlsystem-api", text)
        self.assertNotIn("airflow-webserver", text)


if __name__ == "__main__":
    unittest.main()
