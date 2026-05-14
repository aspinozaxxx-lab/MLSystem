from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..api.security import mask_secrets


class ArtifactBase(BaseModel):
    schema_version: int = 1

    def safe_dump(self) -> dict[str, Any]:
        return mask_secrets(self.model_dump())


class InventoryScenesArtifact(ArtifactBase):
    stage: str = "inventory_scenes"
    experiment_id: str | None = None
    images_uri: str | None = None
    layout_uri: str | None = None
    annotation_uri: str | None = None
    scenes_uri: str | None = None
    scene_count: int = 0
    matched_count: int = 0
    missing_count: int = 0
    ambiguous_count: int = 0
    matched: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    ambiguous: list[dict[str, Any]] = Field(default_factory=list)


class DatasetManifestArtifact(ArtifactBase):
    experiment_id: str
    created_by: str = "prepare_dataset"
    source: str = "pipeline_runner"
    split_strategy: str
    object_count_mode: str
    selected_scene_count: int
    train_scene_count: int
    val_scene_count: int
    train_scenes: list[dict[str, Any]] = Field(default_factory=list)
    val_scenes: list[dict[str, Any]] = Field(default_factory=list)
    split_summary: dict[str, Any] = Field(default_factory=dict)


class InferenceManifestArtifact(ArtifactBase):
    stage: str = "inference_engine_pipeline"
    experiment_id: str
    run_on: str
    bad_scene_policy: str = "skip"
    scene_count: int = 0
    missing_count: int = 0
    scenes: list[dict[str, Any]] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
