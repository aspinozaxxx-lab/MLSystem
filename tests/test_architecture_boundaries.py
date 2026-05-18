from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "mlsystem" / "src"
ARCH = ROOT / "docs" / "architecture"


def test_train_pipeline_rename_is_complete() -> None:
    assert (SRC / "train_pipeline" / "api.py").exists()
    assert not (SRC / "pipeline_runner").exists()
    assert not (ARCH / "pipeline_runner_module.md").exists()
    assert (ARCH / "train_pipeline_module.md").exists()
    assert "pipeline_runner" not in (ARCH / "architecture.md").read_text(encoding="utf-8")


def test_inference_pipeline_module_exists() -> None:
    assert (SRC / "inference_pipeline" / "api.py").exists()
    assert (SRC / "inference_pipeline" / "contracts.py").exists()
    doc = (ARCH / "inference_pipeline_module.md").read_text(encoding="utf-8")
    assert "Public API" in doc
    assert "Запрещ" in doc or "Forbidden" in doc


def test_train_module_exists_and_legacy_training_modules_absent() -> None:
    assert (SRC / "train" / "api.py").exists()
    assert (SRC / "train" / "contracts.py").exists()
    assert not (SRC / "training").exists()
    assert not (SRC / "tiling").exists()
    assert not (SRC / "workflow").exists()
    assert not (SRC / "orchestration").exists()
    assert not (SRC / "real_train.py").exists()
    assert not (SRC / "models").exists()
    assert not (SRC / "inference").exists()
    assert not (SRC / "pipeline" / "training_pipeline.py").exists()
    assert not (SRC / "tile_preparation" / "facade.py").exists()


def test_legacy_namespaces_and_root_files_absent() -> None:
    for name in [
        "config",
        "contracts",
        "app",
        "debug",
        "vectorization",
        "postprocessing",
        "preprocessing",
        "data",
    ]:
        assert not (SRC / name).exists(), name
    for name in [
        "io_utils.py",
        "object_metrics.py",
        "s3_adapter.py",
        "preprocess_inventory.py",
        "pipeline_config.py",
        "job_schema.py",
    ]:
        assert not (SRC / name).exists(), name


def test_supporting_modules_are_formalized() -> None:
    assert (SRC / "dataset_preparing" / "api.py").exists()
    assert (SRC / "dataset_preparing" / "contracts.py").exists()
    assert (SRC / "metrics" / "api.py").exists()
    assert (SRC / "storage" / "api.py").exists()
    assert (SRC / "storage" / "contracts.py").exists()
    assert (SRC / "settings" / "api.py").exists()
    assert (SRC / "settings" / "contracts.py").exists()
    assert (ARCH / "dataset_preparing_module.md").exists()
    assert (ARCH / "metrics_module.md").exists()
    assert (ARCH / "storage_module.md").exists()
    assert (ARCH / "settings_module.md").exists()


def test_dataset_preparing_public_api_is_narrow() -> None:
    import mlsystem.src.dataset_preparing.api as api

    assert set(api.__all__) == {
        "DatasetIdentity",
        "DatasetInspectionRequest",
        "DatasetInspectionResult",
        "DatasetPreparingError",
        "TrainingDatasetPrepareRequest",
        "TrainingDatasetPrepareResult",
        "inspect_dataset",
        "prepare_training_dataset",
    }


def test_metrics_public_api_is_narrow() -> None:
    import mlsystem.src.metrics.api as api

    assert set(api.__all__) == {
        "MetricsError",
        "PixelMetricAccumulator",
        "compute_pixel_metrics_from_logits",
    }
    forbidden = {
        "clean_geometries",
        "compute_pairwise_iou",
        "match_objects_by_iou",
        "prepare_geometries",
        "probabilities_from_logits",
        "safe_divide",
        "write_object_metrics_artifacts",
    }
    assert forbidden.isdisjoint(set(api.__all__))


def test_train_pipeline_uses_dataset_preparing_public_api_only() -> None:
    offenders: list[str] = []
    forbidden = [
        "dataset_preparing._dataset_split",
        "dataset_preparing._scene_matching",
        "dataset_preparing._identity",
        "compute_dataset_identity",
        "dataset_identity_mlflow_payload",
        "build_scene_matching_report",
        "split_train_val_by_object_counts",
        "count_objects_per_scene",
    ]
    for path in (SRC / "train_pipeline").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden if item in text]
        if hits:
            offenders.append(f"{path.relative_to(SRC).as_posix()}: {hits}")
    assert offenders == []


def test_mlflow_adapter_dataset_input_helper_exists() -> None:
    import mlsystem.src.mlflow_adapter.api as api

    assert "log_dataset_input_to_run" in api.__all__


def test_pipeline_package_absent_or_formal_module() -> None:
    pipeline = SRC / "pipeline"
    if not pipeline.exists():
        return
    assert (pipeline / "api.py").exists()
    assert (pipeline / "contracts.py").exists()
    assert any(path.name == "pipeline_module.md" for path in ARCH.glob("*pipeline_module.md"))


def test_api_app_uses_public_pipeline_apis_only() -> None:
    text = (SRC / "api" / "app.py").read_text(encoding="utf-8")
    assert "from ..train_pipeline.api import" in text
    assert "from ..inference_pipeline.api import" in text
    assert "from ..mlflow_adapter.api import" in text
    forbidden = [
        "train_pipeline.config",
        "train_pipeline.run_store",
        "train_pipeline.runner",
        "train_pipeline.stages",
        "inference_pipeline._",
    ]
    assert [item for item in forbidden if item in text] == []


def test_no_star_import_wrappers() -> None:
    pattern = re.compile(r"^\s*from\s+[\.\w]+.*import\s+\*", re.MULTILINE)
    offenders = [path.relative_to(SRC).as_posix() for path in SRC.rglob("*.py") if pattern.search(path.read_text(encoding="utf-8"))]
    assert offenders == []


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


def test_production_does_not_import_inference_engine_internals() -> None:
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        rel = path.relative_to(SRC)
        if rel.parts and rel.parts[0] == "inference_pipeline":
            continue
        text = path.read_text(encoding="utf-8")
        if "InferenceEngine.src." in text:
            offenders.append(rel.as_posix())
    assert offenders == []


def test_inference_pipeline_does_not_import_forbidden_boundaries() -> None:
    forbidden = [
        "FastAPI",
        "from fastapi",
        "from ..train._",
        "from ..train_pipeline.",
        "train_pipeline._",
    ]
    offenders: dict[str, list[str]] = {}
    for path in (SRC / "inference_pipeline").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [item for item in forbidden if item in text]
        if hits:
            offenders[path.relative_to(SRC).as_posix()] = hits
    assert offenders == {}


def test_train_module_does_not_import_forbidden_boundaries() -> None:
    forbidden = [
        "import mlflow",
        "from mlflow",
        "MlflowClient",
        "mlflow.artifacts",
        "train_pipeline.",
        "from ..train_pipeline",
        "inference_pipeline._",
        "from ..inference_pipeline._",
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


def test_train_pipeline_train_stage_uses_train_api_boundary() -> None:
    text = (SRC / "train_pipeline" / "experiment_stages.py").read_text(encoding="utf-8")
    assert "from ..train.api import train_model" in text
    assert "train_model(" in text
    assert "from ..train._" not in text
    assert "real_train" not in text
