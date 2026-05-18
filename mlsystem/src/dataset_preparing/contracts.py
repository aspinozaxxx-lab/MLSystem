from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetIdentity:
    fingerprint: str
    version: str
    version_source: str
    objects_count: int
    scenes_count: int
    selected_scenes: list[str] = field(default_factory=list)
    train_scenes: list[str] = field(default_factory=list)
    val_scenes: list[str] = field(default_factory=list)
    git_commit: str | None = None
    git_commit_date: str | None = None
    class_name: str | None = None
    class_slug: str | None = None
    split_strategy: str | None = None
    annotation_uri: str | None = None
    scenes_uri: str | None = None
    images_uri: str | None = None
    layout_uri: str | None = None


class DatasetPreparingError(Exception):
    pass
