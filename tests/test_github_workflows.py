from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

ORCHESTRATOR = "cicd.yml"
REUSABLE_WORKFLOWS = {
    "ansible.yml",
    "inference-engine-infra.yml",
    "mlservice.yml",
    "inference-engine-service.yml",
    "frontend-site.yml",
    "frontend-ansible.yml",
    "monitoring.yml",
    "server-diagnostics.yml",
}
EXPECTED_TRIGGER_PATHS = {
    "ansible/**",
    "deploy/**",
    "mlsystem/**",
    "InferenceEngine/**",
    "frontend/**",
    "monitoring/**",
    "configs/**",
    "scripts/**",
    "tests/**",
    ".github/workflows/**",
}


def _workflow_paths() -> list[Path]:
    return sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))


def _load_workflow(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _trigger_config(data: dict) -> dict:
    return data.get("on") or {}


def test_workflow_yaml_files_parse() -> None:
    paths = _workflow_paths()
    assert paths
    for path in paths:
        assert isinstance(_load_workflow(path), dict), path


def test_only_cicd_has_push_and_pull_request_triggers() -> None:
    for path in _workflow_paths():
        triggers = _trigger_config(_load_workflow(path))
        if path.name == ORCHESTRATOR:
            assert "push" in triggers
            assert "pull_request" in triggers
        else:
            assert "push" not in triggers, path.name
            assert "pull_request" not in triggers, path.name


def test_legacy_workflows_are_reusable() -> None:
    for name in REUSABLE_WORKFLOWS:
        path = WORKFLOWS / name
        assert path.exists(), name
        triggers = _trigger_config(_load_workflow(path))
        assert "workflow_call" in triggers, name


def test_workflows_do_not_reference_removed_pipeline_modules() -> None:
    forbidden = [
        "pipeline_runner",
        "mlsystem.src.pipeline",
        "src.pipeline",
        "inference_engine_pipeline",
    ]
    for path in _workflow_paths():
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} references {token}"


def test_cicd_contains_all_previous_path_triggers() -> None:
    triggers = _trigger_config(_load_workflow(WORKFLOWS / ORCHESTRATOR))
    for trigger_name in ["push", "pull_request"]:
        paths = set(triggers[trigger_name]["paths"])
        assert EXPECTED_TRIGGER_PATHS <= paths


def test_cicd_has_lightweight_workflow_check_job() -> None:
    data = _load_workflow(WORKFLOWS / ORCHESTRATOR)
    assert "workflow_checks" in data["jobs"]
    assert data["jobs"]["workflow_checks"]["if"] == "${{ needs.plan.outputs.run_workflow_checks == 'true' }}"


def test_cicd_change_does_not_fan_out_to_all_reusable_checks() -> None:
    text = (WORKFLOWS / ORCHESTRATOR).read_text(encoding="utf-8")
    assert "|^\\.github/workflows/cicd\\.yml" not in text
    assert 'force_all=true' not in text


def test_cicd_test_only_changes_do_not_deploy_services() -> None:
    text = (WORKFLOWS / ORCHESTRATOR).read_text(encoding="utf-8")
    assert 'reason_mlsystem_service="tests changed"' in text
    assert 'reason_inference_service="tests changed"' in text
    assert 'reason_frontend_site="tests changed"' in text
    assert "grep -q 'tests changed'" in text
