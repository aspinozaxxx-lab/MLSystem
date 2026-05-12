from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from export_mlflow_run_to_triton import DEFAULT_RUN_ID, export_mlflow_run_to_triton


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an MLflow segmentation run to Triton TensorRT-oriented repositories")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--repository", default="/data/mlsystem/triton/model_repository")
    parser.add_argument("--triton-model-name", default="segformer_b2_trt")
    parser.add_argument("--model-name", default="segformer_b2")
    parser.add_argument("--input-bands", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--backend", choices=["onnxruntime-tensorrt", "tensorrt-plan"], default="onnxruntime-tensorrt")
    parser.add_argument("--precision", choices=["FP16", "FP32"], default="FP16")
    parser.add_argument("--workspace-bytes", default=str(8 * 1024 * 1024 * 1024))
    args = parser.parse_args()

    repository = Path(args.repository)
    onnx_model = export_mlflow_run_to_triton(
        run_id=args.run_id,
        repository=repository,
        triton_model_name=args.triton_model_name,
        model_name=args.model_name,
        input_bands=args.input_bands,
        tile_size=args.tile_size,
        max_batch_size=args.max_batch_size,
        instance_count=args.instance_count,
        backend="onnx",
    )
    if args.backend == "onnxruntime-tensorrt":
        write_onnxruntime_tensorrt_config(
            repository=repository,
            triton_model_name=args.triton_model_name,
            input_bands=args.input_bands,
            tile_size=args.tile_size,
            max_batch_size=args.max_batch_size,
            instance_count=args.instance_count,
            precision=args.precision,
            workspace_bytes=args.workspace_bytes,
        )
        print(onnx_model)
        return
    plan_path = build_tensorrt_plan(
        repository=repository,
        triton_model_name=args.triton_model_name,
        onnx_model=onnx_model,
        input_bands=args.input_bands,
        tile_size=args.tile_size,
        max_batch_size=args.max_batch_size,
        precision=args.precision,
        workspace_bytes=args.workspace_bytes,
    )
    write_tensorrt_plan_config(
        repository=repository,
        triton_model_name=args.triton_model_name,
        input_bands=args.input_bands,
        tile_size=args.tile_size,
        max_batch_size=args.max_batch_size,
        instance_count=args.instance_count,
    )
    print(plan_path)


def write_onnxruntime_tensorrt_config(
    *,
    repository: Path,
    triton_model_name: str,
    input_bands: int,
    tile_size: int,
    max_batch_size: int,
    instance_count: int,
    precision: str,
    workspace_bytes: str,
) -> None:
    (repository / triton_model_name / "config.pbtxt").write_text(
        f'''name: "{triton_model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {int(max_batch_size)}
input [
  {{ name: "INPUT__0" data_type: TYPE_FP32 dims: [ {int(input_bands)}, {int(tile_size)}, {int(tile_size)} ] }}
]
output [
  {{ name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 1, -1, -1 ] }}
]
instance_group [ {{ kind: KIND_GPU count: {max(1, int(instance_count))} }} ]
dynamic_batching {{
  preferred_batch_size: [ 4, 8, {int(max_batch_size)} ]
  max_queue_delay_microseconds: 1000
}}
optimization {{
  execution_accelerators {{
    gpu_execution_accelerator : [
      {{
        name : "tensorrt"
        parameters {{ key: "precision_mode" value: "{precision}" }}
        parameters {{ key: "max_workspace_size_bytes" value: "{workspace_bytes}" }}
      }}
    ]
  }}
}}
''',
        encoding="utf-8",
    )


def build_tensorrt_plan(
    *,
    repository: Path,
    triton_model_name: str,
    onnx_model: Path,
    input_bands: int,
    tile_size: int,
    max_batch_size: int,
    precision: str,
    workspace_bytes: str,
) -> Path:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise RuntimeError("trtexec is required for --backend tensorrt-plan")
    version_dir = repository / triton_model_name / "1"
    plan_path = version_dir / "model.plan"
    command = [
        trtexec,
        f"--onnx={onnx_model}",
        f"--saveEngine={plan_path}",
        f"--minShapes=INPUT__0:1x{input_bands}x{tile_size}x{tile_size}",
        f"--optShapes=INPUT__0:{max(1, max_batch_size // 2)}x{input_bands}x{tile_size}x{tile_size}",
        f"--maxShapes=INPUT__0:{max_batch_size}x{input_bands}x{tile_size}x{tile_size}",
        f"--memPoolSize=workspace:{max(1, int(workspace_bytes) // (1024 * 1024))}",
    ]
    if precision == "FP16":
        command.append("--fp16")
    subprocess.run(command, check=True)
    return plan_path


def write_tensorrt_plan_config(
    *,
    repository: Path,
    triton_model_name: str,
    input_bands: int,
    tile_size: int,
    max_batch_size: int,
    instance_count: int,
) -> None:
    (repository / triton_model_name / "config.pbtxt").write_text(
        f'''name: "{triton_model_name}"
platform: "tensorrt_plan"
max_batch_size: {int(max_batch_size)}
input [
  {{ name: "INPUT__0" data_type: TYPE_FP32 dims: [ {int(input_bands)}, {int(tile_size)}, {int(tile_size)} ] }}
]
output [
  {{ name: "OUTPUT__0" data_type: TYPE_FP32 dims: [ 1, -1, -1 ] }}
]
instance_group [ {{ kind: KIND_GPU count: {max(1, int(instance_count))} }} ]
dynamic_batching {{
  preferred_batch_size: [ 4, 8, {int(max_batch_size)} ]
  max_queue_delay_microseconds: 1000
}}
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
