from __future__ import annotations

# Compatibility-safe relocation: root mlsystem.src.object_metrics remains the
# stable implementation for existing imports; new code should import this path.
from ..object_metrics import (  # noqa: F401
    ObjectMatch,
    clean_geometries,
    compute_object_f1,
    compute_object_metrics_by_scene,
    compute_pairwise_iou,
    match_objects_by_iou,
    prepare_geometries,
    reproject_geometries,
)

__all__ = [
    "ObjectMatch",
    "clean_geometries",
    "compute_object_f1",
    "compute_object_metrics_by_scene",
    "compute_pairwise_iou",
    "match_objects_by_iou",
    "prepare_geometries",
    "reproject_geometries",
]
