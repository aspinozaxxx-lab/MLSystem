from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from affine import Affine

from ..api.schemas import JobRequest, SceneInput
from ..tiling.windows import tile_insert_slices, window_grid
from ..vectorization.tile_index import window_bounds


@dataclass(frozen=True)
class TileDescriptor:
    job_id: str
    scene_id: str
    tile_id: str
    x: int
    y: int
    width: int
    height: int
    patch_size: int
    stride: int
    insert: dict[str, int]
    artifact_path: str
    meta_path: str
    checksum_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TileDescriptor":
        return cls(
            job_id=str(payload["job_id"]),
            scene_id=str(payload["scene_id"]),
            tile_id=str(payload["tile_id"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            patch_size=int(payload["patch_size"]),
            stride=int(payload["stride"]),
            insert={str(key): int(value) for key, value in dict(payload.get("insert") or {}).items()},
            artifact_path=str(payload["artifact_path"]),
            meta_path=str(payload["meta_path"]),
            checksum_path=str(payload["checksum_path"]),
        )


@dataclass(frozen=True)
class BlockDescriptor:
    job_id: str
    scene_id: str
    block_id: str
    core_window: tuple[int, int, int, int]
    expanded_window: tuple[int, int, int, int]
    core_bbox: tuple[float, float, float, float]
    expanded_bbox: tuple[float, float, float, float]
    crs: str | None
    dependency_tile_ids: list[str]
    artifact_path: str
    vector_path: str
    summary_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BlockDescriptor":
        return cls(
            job_id=str(payload["job_id"]),
            scene_id=str(payload["scene_id"]),
            block_id=str(payload["block_id"]),
            core_window=tuple(int(v) for v in payload["core_window"]),  # type: ignore[arg-type]
            expanded_window=tuple(int(v) for v in payload["expanded_window"]),  # type: ignore[arg-type]
            core_bbox=tuple(float(v) for v in payload["core_bbox"]),  # type: ignore[arg-type]
            expanded_bbox=tuple(float(v) for v in payload["expanded_bbox"]),  # type: ignore[arg-type]
            crs=payload.get("crs"),
            dependency_tile_ids=[str(v) for v in payload.get("dependency_tile_ids") or []],
            artifact_path=str(payload["artifact_path"]),
            vector_path=str(payload["vector_path"]),
            summary_path=str(payload["summary_path"]),
        )


@dataclass
class ScenePlan:
    job_id: str
    scene_id: str
    scene_name: str
    width: int
    height: int
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    tiles: list[TileDescriptor] = field(default_factory=list)
    blocks: list[BlockDescriptor] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scene_id": self.scene_id,
            "scene_name": self.scene_name,
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": list(self.transform),
            "tiles": [tile.to_dict() for tile in self.tiles],
            "blocks": [block.to_dict() for block in self.blocks],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScenePlan":
        return cls(
            job_id=str(payload["job_id"]),
            scene_id=str(payload["scene_id"]),
            scene_name=str(payload.get("scene_name") or payload["scene_id"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            crs=payload.get("crs"),
            transform=tuple(float(v) for v in payload["transform"][:6]),  # type: ignore[arg-type]
            tiles=[TileDescriptor.from_dict(dict(item)) for item in payload.get("tiles") or []],
            blocks=[BlockDescriptor.from_dict(dict(item)) for item in payload.get("blocks") or []],
            source=dict(payload.get("source") or {}),
        )


def build_job_plan(job_id: str, request: JobRequest, job_dir: Path) -> list[ScenePlan]:
    scenes = resolve_scene_inputs(request)
    return [build_scene_plan(job_id, idx, scene, request, job_dir) for idx, scene in enumerate(scenes)]


def resolve_scene_inputs(request: JobRequest) -> list[SceneInput]:
    scenes = list(request.scenes or [])
    if request.max_scenes is not None:
        scenes = scenes[: max(0, int(request.max_scenes))]
    if not scenes and request.inference_manifest:
        scenes = _read_manifest_scenes(Path(request.inference_manifest), request)
        if request.max_scenes is not None:
            scenes = scenes[: max(0, int(request.max_scenes))]
    if not scenes:
        scenes = [SceneInput(scene_id="synthetic", name="synthetic", width=256, height=256, probability_rects=[[32, 32, 220, 220, 1.0]])]
    return scenes


def build_scene_plan(job_id: str, scene_index: int, scene: SceneInput, request: JobRequest, job_dir: Path) -> ScenePlan:
    preprocess = request.preprocess
    vector_cfg = request.effective_vectorization()
    source_payload = scene.model_dump()
    if not (source_payload.get("uri") or source_payload.get("image_uri") or source_payload.get("path")):
        source_payload["image_uri"] = _scene_input_uri(scene, request)
    size_scene = scene.model_copy(update={"image_uri": source_payload.get("image_uri")})
    inferred_metadata = _infer_scene_metadata(size_scene)
    width = int(scene.width or (inferred_metadata.get("width") if inferred_metadata else 256))
    height = int(scene.height or (inferred_metadata.get("height") if inferred_metadata else 256))
    scene_id = str(scene.scene_id or scene.name or f"scene_{scene_index:04d}")
    scene_name = str(scene.name or scene_id)
    transform_values = scene.transform or inferred_metadata.get("transform") or [1, 0, 0, 0, -1, float(height)]
    transform = tuple(float(v) for v in transform_values[:6])  # type: ignore[assignment]
    crs = scene.crs or inferred_metadata.get("crs")
    affine = Affine(*transform)
    scene_dir = job_dir / "scenes" / scene_id
    tiles_dir = scene_dir / "tiles"
    blocks_dir = scene_dir / "blocks"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir.mkdir(parents=True, exist_ok=True)

    tile_descriptors: list[TileDescriptor] = []
    patch_size = preprocess.effective_tile_size
    for window in window_grid(width, height, patch_size, preprocess.stride, scene_id=scene_id):
        tile_id = f"{scene_id}_tile_{int(window.index or 0):06d}"
        insert = tile_insert_slices(
            window.x,
            window.y,
            window.width,
            window.height,
            width,
            height,
            patch_size=patch_size,
            crop_mode=preprocess.crop_mode,
            center_size=preprocess.center_size,
            context_bounds=preprocess.context_bounds,
        )
        tile_descriptors.append(
            TileDescriptor(
                job_id=job_id,
                scene_id=scene_id,
                tile_id=tile_id,
                x=window.x,
                y=window.y,
                width=window.width,
                height=window.height,
                patch_size=patch_size,
                stride=preprocess.stride,
                insert=insert,
                artifact_path=str(tiles_dir / f"{tile_id}.npz"),
                meta_path=str(tiles_dir / f"{tile_id}.json"),
                checksum_path=str(tiles_dir / f"{tile_id}.sha256"),
            )
        )

    block_descriptors: list[BlockDescriptor] = []
    rows = max(1, math.ceil(height / vector_cfg.core_size_px))
    cols = max(1, math.ceil(width / vector_cfg.core_size_px))
    for row in range(rows):
        for col in range(cols):
            col_off = col * vector_cfg.core_size_px
            row_off = row * vector_cfg.core_size_px
            core_w = min(vector_cfg.core_size_px, width - col_off)
            core_h = min(vector_cfg.core_size_px, height - row_off)
            exp_col = max(0, col_off - vector_cfg.halo_px)
            exp_row = max(0, row_off - vector_cfg.halo_px)
            exp_end_col = min(width, col_off + core_w + vector_cfg.halo_px)
            exp_end_row = min(height, row_off + core_h + vector_cfg.halo_px)
            expanded = (int(exp_col), int(exp_row), int(exp_end_col - exp_col), int(exp_end_row - exp_row))
            dependencies = [
                tile.tile_id
                for tile in tile_descriptors
                if _windows_intersect((tile.x, tile.y, tile.width, tile.height), expanded)
            ]
            block_id = f"{scene_id}_block_{row:04d}_{col:04d}"
            block_descriptors.append(
                BlockDescriptor(
                    job_id=job_id,
                    scene_id=scene_id,
                    block_id=block_id,
                    core_window=(int(col_off), int(row_off), int(core_w), int(core_h)),
                    expanded_window=expanded,
                    core_bbox=window_bounds(affine, col_off, row_off, core_w, core_h),
                    expanded_bbox=window_bounds(affine, exp_col, exp_row, exp_end_col - exp_col, exp_end_row - exp_row),
                    crs=crs,
                    dependency_tile_ids=dependencies,
                    artifact_path=str(blocks_dir / f"{block_id}.npz"),
                    vector_path=str(blocks_dir / f"{block_id}.geojson"),
                    summary_path=str(blocks_dir / f"{block_id}.summary.json"),
                )
            )
    return ScenePlan(
        job_id=job_id,
        scene_id=scene_id,
        scene_name=scene_name,
        width=width,
        height=height,
        crs=crs,
        transform=transform,
        tiles=tile_descriptors,
        blocks=block_descriptors,
        source=source_payload,
    )


def write_plan(job_dir: Path, plans: list[ScenePlan]) -> Path:
    path = job_dir / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "scenes": [plan.to_dict() for plan in plans]}, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_scene_plan(job_dir: Path, plan: ScenePlan) -> Path:
    path = job_dir / "scenes" / plan.scene_id / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_scene_plan(job_dir: Path, scene_id: str) -> ScenePlan:
    path = job_dir / "scenes" / scene_id / "plan.json"
    return ScenePlan.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


def _windows_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _read_manifest_scenes(path: Path, request: JobRequest) -> list[SceneInput]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    scenes: list[SceneInput] = []
    for idx, row in enumerate(payload.get("scenes") or payload.get("scene_results") or []):
        uri = _manifest_row_uri(row, request)
        scenes.append(
            SceneInput(
                scene_id=str(row.get("scene_id") or row.get("entry") or row.get("name") or f"scene_{idx:04d}"),
                name=str(row.get("scene_name") or row.get("name") or row.get("entry") or f"scene_{idx:04d}"),
                uri=uri,
                key=row.get("key"),
                width=row.get("width"),
                height=row.get("height"),
                crs=row.get("crs"),
                transform=row.get("transform"),
                metadata=dict(row),
            )
        )
    return scenes


def _manifest_row_uri(row: dict[str, Any], request: JobRequest) -> str | None:
    explicit = row.get("uri") or row.get("image_uri") or row.get("path") or row.get("url")
    if explicit:
        return str(explicit)
    key = row.get("key") or row.get("object_key") or row.get("s3_key")
    if not key:
        return None
    key_text = str(key).lstrip("/")
    if key_text.startswith(("s3://", "file://", "/")):
        return str(key)
    bucket = row.get("bucket") or row.get("s3_bucket")
    if bucket:
        return f"s3://{str(bucket).strip('/')}/{key_text}"
    return _image_uri_for_key(request.images_uri, key_text)


def _scene_input_uri(scene: SceneInput, request: JobRequest) -> str | None:
    key = scene.key or scene.object_key or scene.s3_key or scene.metadata.get("key") or scene.metadata.get("object_key") or scene.metadata.get("s3_key")
    if not key:
        return None
    key_text = str(key).lstrip("/")
    if key_text.startswith(("s3://", "file://", "/")):
        return str(key)
    bucket = scene.bucket or scene.metadata.get("bucket") or scene.metadata.get("s3_bucket")
    if bucket:
        return f"s3://{str(bucket).strip('/')}/{key_text}"
    return _image_uri_for_key(request.images_uri, key_text)


def _infer_scene_metadata(scene: SceneInput) -> dict[str, Any]:
    path = scene.path or scene.uri or scene.image_uri
    if not path:
        return {}
    try:
        import rasterio

        path_text = str(path)
        if path_text.startswith("file://"):
            path_text = path_text[7:]
        elif path_text.startswith("s3://"):
            bucket_key = path_text[5:]
            bucket, _, key = bucket_key.partition("/")
            path_text = f"/vsis3/{bucket}/{key}"
        with rasterio.open(path_text) as ds:
            metadata: dict[str, Any] = {
                "width": int(ds.width),
                "height": int(ds.height),
                "transform": [float(v) for v in ds.transform[:6]],
            }
            if ds.crs:
                metadata["crs"] = str(ds.crs)
            return metadata
    except Exception:
        return {}


def _image_uri_for_key(images_uri: str | None, key: Any) -> str | None:
    if not key:
        return None
    key_text = str(key).lstrip("/")
    if not images_uri:
        return key_text
    base = images_uri.rstrip("/")
    if base.startswith("s3://"):
        bucket_key = base[5:]
        bucket, _, prefix = bucket_key.partition("/")
        prefix = prefix.strip("/")
        if prefix and (key_text == prefix or key_text.startswith(prefix + "/")):
            return f"s3://{bucket}/{key_text}"
    return base + "/" + key_text
