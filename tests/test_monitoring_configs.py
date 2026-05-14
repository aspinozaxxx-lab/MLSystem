from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


class MonitoringConfigTests(unittest.TestCase):
    def test_prometheus_scrape_jobs_cover_mlsystem_sources(self) -> None:
        config = yaml.safe_load(Path("monitoring/prometheus/prometheus.yml").read_text(encoding="utf-8"))
        jobs = {job["job_name"]: job for job in config["scrape_configs"]}
        for name in [
            "node-exporter",
            "cadvisor",
            "dcgm-exporter",
            "rabbitmq",
            "triton",
            "inference-engine",
            "mlsystem-monitor-exporter",
        ]:
            self.assertIn(name, jobs)
        self.assertEqual(jobs["inference-engine"]["metrics_path"], "/metrics/prometheus")

    def test_grafana_dashboards_are_provisioned(self) -> None:
        dashboards = {}
        for path in Path("monitoring/grafana/dashboards").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            dashboards[payload["uid"]] = payload
        self.assertIn("mlsystem-overview", dashboards)
        overview = dashboards["mlsystem-overview"]
        titles = {panel["title"] for panel in overview["panels"]}
        self.assertIn("Machine CPU", titles)
        self.assertIn("GPU Utilization", titles)
        self.assertIn("RabbitMQ Queue Depth", titles)
        self.assertIn("InferenceEngine Progress", titles)
        self.assertIn("Service Health", titles)

    def test_compose_contains_monitoring_services(self) -> None:
        text = Path("deploy/docker-compose.gpu.yml").read_text(encoding="utf-8")
        for service in [
            "prometheus:",
            "grafana:",
            "node-exporter:",
            "cadvisor:",
            "dcgm-exporter:",
            "mlsystem-monitor-exporter:",
        ]:
            self.assertIn(service, text)
        self.assertNotIn("airflow", text.lower())
        self.assertIn("rabbitmq_prometheus", Path("ansible/roles/gpu_platform/tasks/main.yml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
