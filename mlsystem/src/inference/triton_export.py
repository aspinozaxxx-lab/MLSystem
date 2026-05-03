from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_segmentation_checkpoint_to_onnx(
    *,
    checkpoint_path: Path,
    model_name: str,
    output_repository: Path,
    triton_model_name: str,
    input_bands: int,
    tile_size: int,
    max_batch_size: int,
) -> Path:
    from ..real_train import _build_model

    version_dir = output_repository / triton_model_name / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    model = _build_model(model_name, input_bands, 1, 8).eval().cpu()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint payload: {checkpoint_path}")
    model.load_state_dict(state)
    dummy = torch.randn(1, input_bands, tile_size, tile_size, dtype=torch.float32)
    onnx_path = version_dir / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["INPUT__0"],
            output_names=["OUTPUT__0"],
            dynamic_axes={"INPUT__0": {0: "batch"}, "OUTPUT__0": {0: "batch", 2: "height", 3: "width"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    config_path = output_repository / triton_model_name / "config.pbtxt"
    config_path.write_text(
        f'''name: "{triton_model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {int(max_batch_size)}
input [
  {{ name: "INPUT__0" data_type: TYPE_FP32 dims: [ {input_bands}, {tile_size}, {tile_size} ] }}
]
output [
  {{ name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 1, -1, -1 ] }}
]
instance_group [ {{ kind: KIND_GPU count: 1 }} ]
''',
        encoding="utf-8",
    )
    return onnx_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLSystem segmentation checkpoints for Triton ONNXRuntime")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--triton-model-name", required=True)
    parser.add_argument("--repository", default="/data/mlsystem/triton/model_repository")
    parser.add_argument("--input-bands", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=16)
    args = parser.parse_args()
    onnx_path = export_segmentation_checkpoint_to_onnx(
        checkpoint_path=Path(args.checkpoint),
        model_name=args.model_name,
        output_repository=Path(args.repository),
        triton_model_name=args.triton_model_name,
        input_bands=args.input_bands,
        tile_size=args.tile_size,
        max_batch_size=args.max_batch_size,
    )
    print(onnx_path)


if __name__ == "__main__":
    main()
