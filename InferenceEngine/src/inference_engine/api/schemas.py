from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    mlflow_run_id: str | None = "a7838f91528a47e1931b685c2ea06686"
    model_name: str | None = None
    architecture: str | None = "segformer_b2"
    triton_model_name: str | None = "segformer_b2"
    triton_model_version: str | None = None


class PreprocessConfig(BaseModel):
    tile_size: int | None = None
    patch_size: int = 1024
    stride: int = 768
    input_bands: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    crop_mode: str = "full"
    center_size: int | None = None
    context_bounds: int | None = None
    stitch_mode: str = "weighted_overlap"

    @property
    def effective_tile_size(self) -> int:
        return int(self.tile_size or self.patch_size)


class PseudolabelConfig(BaseModel):
    threshold: float = 0.5
    core_size_px: int = 4096
    halo_px: int = 512
    workers: int = 4
    local_min_area: float = 0.0
    final_min_area: float = 0.0
    merge_epsilon: float = 1.0
    simplify_tolerance: float = 0.0
    max_objects: int | None = None


class ResourceConfig(BaseModel):
    triton_batch_size: int = 8
    batches_ahead: int = 512
    max_preprocess_queue: int = 4096
    max_spool_bytes: int = 128 * 1024 * 1024 * 1024
    max_scenes_inflight: int = 32
    max_blocks_inflight: int = 32
    triton_instance_count: int = 1
    max_wait_ms: int = 50


class SceneInput(BaseModel):
    scene_id: str | None = None
    name: str | None = None
    uri: str | None = None
    image_uri: str | None = None
    bucket: str | None = None
    key: str | None = None
    object_key: str | None = None
    s3_key: str | None = None
    path: str | None = None
    width: int | None = None
    height: int | None = None
    crs: str | None = "EPSG:3857"
    transform: list[float] | None = None
    probability_rects: list[list[float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class JobRequest(BaseModel):
    run_id: str | None = None
    experiment_id: str
    scenes: list[SceneInput] | None = None
    inference_manifest: str | None = None
    images_uri: str | None = None
    layout_uri: str | None = None
    storage: dict[str, Any] = Field(default_factory=dict)
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    pseudolabel: PseudolabelConfig = Field(default_factory=PseudolabelConfig)
    vectorization: PseudolabelConfig | None = None
    resource: ResourceConfig = Field(default_factory=ResourceConfig)
    run_dir: str | None = None
    source: str = "inference_engine"
    max_scenes: int | None = None

    def effective_vectorization(self) -> PseudolabelConfig:
        return self.vectorization or self.pseudolabel


class JobCreated(BaseModel):
    job_id: str
    status: str
    events_url: str
    artifacts_url: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "success", "failed", "cancelled"]
    created_at: str
    updated_at: str
    experiment_id: str
    run_id: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    counters: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
