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
    from .export_pseudolabel import run as export_pseudolabel
    from .inventory_scenes import run as inventory_scenes
    from .postprocess_pseudolabel import run as postprocess_pseudolabel
    from .prepare_dataset import run as prepare_dataset
    from .prepare_inference_scenes import run as prepare_inference_scenes
    from .probability_maps import run as validate_probability_maps
    from .pseudolabel_inference import run as run_pseudolabel_inference
    from .vectorize_pseudolabel import run as vectorize_pseudolabel

    register_stage("inventory_scenes", inventory_scenes)
    register_stage("prepare_dataset", prepare_dataset)
    register_stage("prepare_inference_scenes", prepare_inference_scenes)
    register_stage("run_pseudolabel_inference", run_pseudolabel_inference)
    register_stage("validate_probability_maps", validate_probability_maps)
    register_stage("vectorize_pseudolabel", vectorize_pseudolabel)
    register_stage("postprocess_pseudolabel", postprocess_pseudolabel)
    register_stage("export_pseudolabel_artifacts", export_pseudolabel)
    _BUILTINS_REGISTERED = True
