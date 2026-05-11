from __future__ import annotations

from collections.abc import Callable

from .context import StageContext
from .report import StageReport

StageEntrypoint = Callable[[StageContext], StageReport]

_ENTRYPOINTS: dict[str, StageEntrypoint] = {}
_BUILTINS_REGISTERED = False


def register_stage(name: str, entrypoint: StageEntrypoint) -> None:
    _ENTRYPOINTS[name] = entrypoint


def get_stage_entrypoint(name: str) -> StageEntrypoint:
    if not _BUILTINS_REGISTERED:
        register_builtin_stages()
    try:
        return _ENTRYPOINTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ENTRYPOINTS))
        raise KeyError(f"Unknown MLSystem stage entrypoint: {name}. Known registry stages: {known}") from exc


def known_stages() -> list[str]:
    if not _BUILTINS_REGISTERED:
        register_builtin_stages()
    return sorted(_ENTRYPOINTS)


def register_builtin_stages() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .inference_engine_pipeline import run as inference_engine_pipeline
    from .inventory_scenes import run as inventory_scenes
    from .prepare_dataset import run as prepare_dataset

    register_stage("inventory_scenes", inventory_scenes)
    register_stage("prepare_dataset", prepare_dataset)
    register_stage("inference_engine_pipeline", inference_engine_pipeline)
    _BUILTINS_REGISTERED = True
