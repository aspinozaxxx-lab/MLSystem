from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PredictionTileInfo:
    tile_id: str
    scene_id: str
    scene_name: str
    npz_path: str
    meta_path: str
    bounds: tuple[float, float, float, float]
    transform: tuple[float, float, float, float, float, float]
    crs: str | None
    width: int
    height: int
    probability_band: str = "prob_uint8"
    nodata: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictionTileInfo":
        return cls(
            tile_id=str(payload["tile_id"]),
            scene_id=str(payload.get("scene_id") or payload["tile_id"]),
            scene_name=str(payload.get("scene_name") or payload.get("scene_id") or payload["tile_id"]),
            npz_path=str(payload["npz_path"]),
            meta_path=str(payload["meta_path"]),
            bounds=tuple(float(v) for v in payload["bounds"]),  # type: ignore[arg-type]
            transform=tuple(float(v) for v in payload["transform"]),  # type: ignore[arg-type]
            crs=payload.get("crs"),
            width=int(payload["width"]),
            height=int(payload["height"]),
            probability_band=str(payload.get("probability_band") or "prob_uint8"),
            nodata=payload.get("nodata"),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class ProcessingBlock:
    block_id: str
    scene_id: str
    core_window: tuple[int, int, int, int]
    expanded_window: tuple[int, int, int, int]
    core_bbox: tuple[float, float, float, float]
    expanded_bbox: tuple[float, float, float, float]
    crs: str | None
    intersecting_tile_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProcessingBlock":
        return cls(
            block_id=str(payload["block_id"]),
            scene_id=str(payload["scene_id"]),
            core_window=tuple(int(v) for v in payload["core_window"]),  # type: ignore[arg-type]
            expanded_window=tuple(int(v) for v in payload["expanded_window"]),  # type: ignore[arg-type]
            core_bbox=tuple(float(v) for v in payload["core_bbox"]),  # type: ignore[arg-type]
            expanded_bbox=tuple(float(v) for v in payload["expanded_bbox"]),  # type: ignore[arg-type]
            crs=payload.get("crs"),
            intersecting_tile_ids=[str(v) for v in payload.get("intersecting_tile_ids") or []],
        )


@dataclass
class BlockVectorizationJob:
    run_id: str
    block: ProcessingBlock
    tiles: list[PredictionTileInfo]
    threshold: float
    local_min_area: float
    output_dir: str
    vector_format: str = "geojson"
    class_name: str = "deforest"
    worker_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["block"] = self.block.to_dict()
        payload["tiles"] = [tile.to_dict() for tile in self.tiles]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockVectorizationJob":
        return cls(
            run_id=str(payload["run_id"]),
            block=ProcessingBlock.from_dict(dict(payload["block"])),
            tiles=[PredictionTileInfo.from_dict(dict(item)) for item in payload.get("tiles") or []],
            threshold=float(payload["threshold"]),
            local_min_area=float(payload.get("local_min_area") or 0.0),
            output_dir=str(payload["output_dir"]),
            vector_format=str(payload.get("vector_format") or "geojson"),
            class_name=str(payload.get("class_name") or "deforest"),
            worker_id=payload.get("worker_id"),
        )


@dataclass
class BlockVectorizationResult:
    block_id: str
    status: str
    output_vector_path: str | None
    summary_path: str | None
    polygons_before_clip: int = 0
    polygons_after_clip: int = 0
    polygons_after_local_filter: int = 0
    boundary_candidates_count: int = 0
    duration_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockVectorizationResult":
        return cls(
            block_id=str(payload["block_id"]),
            status=str(payload["status"]),
            output_vector_path=payload.get("output_vector_path"),
            summary_path=payload.get("summary_path"),
            polygons_before_clip=int(payload.get("polygons_before_clip") or 0),
            polygons_after_clip=int(payload.get("polygons_after_clip") or 0),
            polygons_after_local_filter=int(payload.get("polygons_after_local_filter") or 0),
            boundary_candidates_count=int(payload.get("boundary_candidates_count") or 0),
            duration_sec=float(payload.get("duration_sec") or 0.0),
            warnings=[str(item) for item in payload.get("warnings") or []],
            errors=[str(item) for item in payload.get("errors") or []],
        )


@dataclass
class VectorizationPlan:
    tiles_total: int
    blocks_total: int
    crs: str | None
    bounds: tuple[float, float, float, float] | None
    core_size_px: int
    halo_px: int
    workers_requested: int
    workers_effective: int
    memory_guard: dict[str, Any]
    blocks: list[ProcessingBlock]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocks"] = [block.to_dict() for block in self.blocks]
        return payload


def ensure_json_serializable_path(path: str | Path) -> str:
    return str(path)
