from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mlsystem" / "src"


def test_pipeline_runner_api_exists_and_facade_absent() -> None:
    assert (SRC / "pipeline_runner" / "api.py").exists()
    assert not (SRC / "pipeline_runner" / "facade.py").exists()


def test_mlflow_adapter_api_exists_and_facade_absent() -> None:
    assert (SRC / "mlflow_adapter" / "api.py").exists()
    assert not (SRC / "mlflow_adapter" / "facade.py").exists()


def test_train_module_exists_and_legacy_training_modules_absent() -> None:
    assert (SRC / "train" / "api.py").exists()
    assert (SRC / "train" / "contracts.py").exists()
    assert not (SRC / "training").exists()
    assert not (SRC / "tiling").exists()
    assert not (SRC / "workflow").exists()
    assert not (SRC / "orchestration").exists()
    assert not (SRC / "real_train.py").exists()
    assert not (SRC / "pipeline" / "training_pipeline.py").exists()
    assert not (SRC / "tile_preparation" / "facade.py").exists()


def test_production_mlflow_imports_stay_inside_adapter() -> None:
    pattern = re.compile(r"^\s*(import\s+mlflow\b|from\s+mlflow\b)|\bMlflowClient\b|mlflow\.artifacts", re.MULTILINE)
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if rel.parts and rel.parts[0] == "mlflow_adapter":
            continue
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(rel.as_posix())
    assert offenders == []


def test_api_app_uses_pipeline_runner_public_api() -> None:
    text = (SRC / "api" / "app.py").read_text(encoding="utf-8")
    assert "from ..pipeline_runner.api import" in text
    assert "from ..mlflow_adapter.api import" in text
    forbidden = [
        "pipeline_runner.config",
        "pipeline_runner.run_store",
        "pipeline_runner.runner",
        "pipeline_runner.stages",
    ]
    assert [item for item in forbidden if item in text] == []


def test_deprecated_api_lifecycle_wrappers_absent() -> None:
    for name in ["job_runner.py", "job_store.py", "stage_job_worker.py", "stage_routes.py"]:
        assert not (SRC / "api" / name).exists()


def test_experiment_stages_does_not_own_lifecycle_models() -> None:
    text = (SRC / "pipeline" / "experiment_stages.py").read_text(encoding="utf-8")
    forbidden = [
        "class ExperimentStageStore",
        "class ExperimentStageConfig",
        "PipelineRunStore(",
        "StageJobStore(",
        "StageJobRunner(",
        "subprocess.Popen",
    ]
    assert [item for item in forbidden if item in text] == []


def test_train_module_does_not_import_forbidden_boundaries() -> None:
    forbidden = [
        "import mlflow",
        "from mlflow",
        "MlflowClient",
        "mlflow.artifacts",
        "pipeline_runner.",
        "from ..pipeline_runner",
        "FastAPI",
        "status.json",
        "run.json",
        "stage_json_path",
        "write_stage",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (SRC / "train").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden if item in text]
        if hits:
            offenders[path.relative_to(SRC).as_posix()] = hits
    assert offenders == {}


def test_train_contracts_are_pure_contracts() -> None:
    path = SRC / "train" / "contracts.py"
    text = path.read_text(encoding="utf-8")
    forbidden_tokens = [
        "open(",
        ".read_text(",
        ".write_text(",
        ".mkdir(",
        "subprocess",
        "import mlflow",
        "from mlflow",
        "MlflowClient",
        "mlflow.artifacts",
        "from ._",
    ]
    assert [item for item in forbidden_tokens if item in text] == []


def test_pipeline_train_stage_uses_train_api_boundary() -> None:
    text = (SRC / "pipeline" / "experiment_stages.py").read_text(encoding="utf-8")
    assert "from ..train.api import train_model" in text
    assert "train_model(" in text
    assert "from ..train._" not in text
    assert "TrainingPipeline" not in text
    assert "real_train" not in text


def test_contracts_files_are_pure_contracts() -> None:
    forbidden_tokens = [
        "open(",
        ".read_text(",
        ".write_text(",
        ".mkdir(",
        "subprocess",
        "import mlflow",
        "from mlflow",
        "MlflowClient",
        "mlflow.artifacts",
    ]
    forbidden_internal_import = re.compile(r"^\s*from\s+\.(?!contracts\b)[A-Za-z_]", re.MULTILINE)
    forbidden_private_import = re.compile(r"^\s*from\s+\._", re.MULTILINE)
    for path in SRC.rglob("contracts.py"):
        text = path.read_text(encoding="utf-8")
        assert [item for item in forbidden_tokens if item in text] == [], path
        assert not forbidden_internal_import.search(text), path
        assert not forbidden_private_import.search(text), path
