from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    local_dir = Path(mlflow.artifacts.download_artifacts(run_id=run_id))
    checkpoint = _find_checkpoint(local_dir)
    if checkpoint is None:
        artifacts = [item.path for item in client.list_artifacts(run_id)]
        raise RuntimeError(f"No checkpoint artifact found for MLflow run {run_id}. Top-level artifacts: {artifacts}")
    if backend in {"auto", "onnx"}:
        try:
            from mlsystem.src.inference.triton_export import export_segmentation_checkpoint_to_onnx

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

from mlsystem.src.real_train import _build_model


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
