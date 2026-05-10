from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_RUN_ID = "a7838f91528a47e1931b685c2ea06686"


def export_mlflow_run_to_triton(
    *,
    run_id: str,
    repository: Path,
    triton_model_name: str,
    model_name: str,
    input_bands: int,
    tile_size: int,
    max_batch_size: int,
    backend: str = "auto",
) -> Path:
    import mlflow

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    checkpoint = _find_checkpoint_for_run(client, run_id=run_id, model_name=model_name)
    if checkpoint is None:
        artifacts = [item.path for item in client.list_artifacts(run_id)]
        raise RuntimeError(f"No checkpoint artifact found for MLflow run {run_id}. Top-level artifacts: {artifacts}")
    if backend in {"auto", "onnx"}:
        try:
            try:
                from mlsystem.src.inference.triton_export import export_segmentation_checkpoint_to_onnx
            except ImportError:
                from src.inference.triton_export import export_segmentation_checkpoint_to_onnx

            return export_segmentation_checkpoint_to_onnx(
                checkpoint_path=checkpoint,
                model_name=model_name,
                output_repository=repository,
                triton_model_name=triton_model_name,
                input_bands=input_bands,
                tile_size=tile_size,
                max_batch_size=max_batch_size,
            )
        except Exception:
            if backend == "onnx":
                raise
    return export_python_backend(
        checkpoint_path=checkpoint,
        model_name=model_name,
        output_repository=repository,
        triton_model_name=triton_model_name,
        input_bands=input_bands,
        tile_size=tile_size,
        max_batch_size=max_batch_size,
    )


def export_python_backend(
    *,
    checkpoint_path: Path,
    model_name: str,
    output_repository: Path,
    triton_model_name: str,
    input_bands: int,
    tile_size: int,
    max_batch_size: int,
) -> Path:
    import shutil

    model_root = output_repository / triton_model_name
    version_dir = model_root / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_target = version_dir / "checkpoint.pt"
    shutil.copy2(checkpoint_path, checkpoint_target)
    (version_dir / "model.py").write_text(
        f'''import numpy as np
import torch
import triton_python_backend_utils as pb_utils

try:
    from mlsystem.src.real_train import _build_model
except ImportError:
    from src.real_train import _build_model


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _build_model("{model_name}", {int(input_bands)}, 1, 8).to(self.device).eval()
        payload = torch.load("/models/{triton_model_name}/1/checkpoint.pt", map_location=self.device)
        state = payload.get("model_state_dict") if isinstance(payload, dict) else payload
        self.model.load_state_dict(state)

    def execute(self, requests):
        responses = []
        for request in requests:
            tensor = pb_utils.get_input_tensor_by_name(request, "INPUT__0")
            arr = tensor.as_numpy().astype(np.float32, copy=False)
            sample = torch.from_numpy(arr).to(self.device)
            with torch.no_grad():
                logits = self.model(sample)
                if tuple(logits.shape[-2:]) != tuple(sample.shape[-2:]):
                    logits = torch.nn.functional.interpolate(logits, size=sample.shape[-2:], mode="bilinear", align_corners=False)
            output = pb_utils.Tensor("OUTPUT__0", logits.detach().cpu().numpy().astype(np.float32, copy=False))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses
''',
        encoding="utf-8",
    )
    (model_root / "config.pbtxt").write_text(
        f'''name: "{triton_model_name}"
backend: "python"
max_batch_size: {int(max_batch_size)}
input [
  {{ name: "INPUT__0" data_type: TYPE_FP32 dims: [ {int(input_bands)}, {int(tile_size)}, {int(tile_size)} ] }}
]
output [
  {{ name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 1, -1, -1 ] }}
]
instance_group [
  {{ kind: KIND_GPU count: 1 }}
]
dynamic_batching {{
  preferred_batch_size: [ 4, {int(max_batch_size)} ]
  max_queue_delay_microseconds: 50000
}}
''',
        encoding="utf-8",
    )
    return version_dir / "model.py"


def _find_checkpoint(root: Path) -> Path | None:
    candidates = []
    for suffix in ("*.pt", "*.pth", "*.ckpt"):
        candidates.extend(root.rglob(suffix))
    if not candidates:
        return None
    preferred = [path for path in candidates if "best" in path.name.lower() or "checkpoint" in path.name.lower()]
    return sorted(preferred or candidates, key=lambda item: (len(item.parts), str(item).lower()))[0]


def _find_checkpoint_for_run(client: Any, *, run_id: str, model_name: str) -> Path | None:
    import mlflow

    checked_dirs: list[Path] = []
    try:
        checked_dirs.append(Path(mlflow.artifacts.download_artifacts(run_id=run_id)))
    except Exception:
        pass
    for directory in checked_dirs:
        checkpoint = _find_checkpoint(directory)
        if checkpoint is not None:
            return checkpoint
    run = client.get_run(run_id)
    for value in _checkpoint_hints(run.data.params, run.data.tags):
        checkpoint = _checkpoint_from_hint(value)
        if checkpoint is not None:
            return checkpoint
    try:
        children = client.search_runs(
            [run.info.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{run_id}'",
            max_results=200,
            order_by=["attributes.start_time DESC"],
        )
    except Exception:
        children = []
    for child in children:
        try:
            checkpoint = _find_checkpoint(Path(mlflow.artifacts.download_artifacts(run_id=child.info.run_id)))
            if checkpoint is not None:
                return checkpoint
        except Exception:
            continue
        for value in _checkpoint_hints(child.data.params, child.data.tags):
            checkpoint = _checkpoint_from_hint(value)
            if checkpoint is not None:
                return checkpoint
    return _find_checkpoint_in_filesystem(run_id=run_id, model_name=model_name)


def _checkpoint_hints(*mappings: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    for mapping in mappings:
        for key, value in (mapping or {}).items():
            text = str(value)
            key_text = str(key).lower()
            if any(token in key_text for token in ("checkpoint", "model_path", "weights", "ckpt")):
                hints.append(text)
    return hints


def _checkpoint_from_hint(value: str) -> Path | None:
    if not value:
        return None
    path = Path(value.replace("file://", ""))
    if path.exists() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
        return path
    return None


def _find_checkpoint_in_filesystem(*, run_id: str, model_name: str) -> Path | None:
    roots = [Path("/data/mlsystem/models"), Path("/data/mlsystem/artifacts"), Path("/data/mlsystem/mlflow"), Path("/data/mlsystem/minio")]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        command = f"find {root} -type f \\( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \\) 2>/dev/null | head -2000"
        try:
            output = subprocess.check_output(command, shell=True, text=True, timeout=120)
        except Exception:
            continue
        candidates.extend(Path(line.strip()) for line in output.splitlines() if line.strip())
    if not candidates:
        return None
    run_id_short = run_id[:12].lower()
    model_tokens = {token for token in model_name.lower().replace("-", "_").split("_") if token}

    def score(path: Path) -> tuple[int, float, str]:
        text = str(path).lower()
        points = 0
        if run_id.lower() in text or run_id_short in text:
            points += 100
        if any(token in text for token in model_tokens):
            points += 30
        if "best" in path.name.lower():
            points += 20
        if "checkpoint" in path.name.lower() or "ckpt" in path.name.lower():
            points += 10
        if "inference-engine" in text:
            points -= 50
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (points, mtime, str(path))

    ranked = sorted((path for path in candidates if path.exists()), key=score, reverse=True)
    return ranked[0] if ranked else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an MLflow segmentation run to Triton ONNX model_repository")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--repository", default="/data/mlsystem/triton/model_repository")
    parser.add_argument("--triton-model-name", default="segformer_b2")
    parser.add_argument("--model-name", default="segformer_b2")
    parser.add_argument("--input-bands", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--backend", choices=["auto", "onnx", "python"], default="auto")
    args = parser.parse_args()
    path = export_mlflow_run_to_triton(
        run_id=args.run_id,
        repository=Path(args.repository),
        triton_model_name=args.triton_model_name,
        model_name=args.model_name,
        input_bands=args.input_bands,
        tile_size=args.tile_size,
        max_batch_size=args.max_batch_size,
        backend=args.backend,
    )
    print(path)


if __name__ == "__main__":
    main()
