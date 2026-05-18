from __future__ import annotations

import importlib
from collections.abc import Callable

from .context import StageContext
from .report import StageReport

StageEntrypoint = Callable[[StageContext], StageReport]

_ENTRYPOINTS: dict[str, StageEntrypoint] = {}
_BUILTIN_MODULES: dict[str, str] = {
    "inventory_scenes": ".inventory_scenes",
    "prepare_dataset": ".prepare_dataset",
}
_BUILTINS_REGISTERED = False


def register_stage(name: str, entrypoint: StageEntrypoint) -> None:
    _ENTRYPOINTS[name] = entrypoint


def get_stage_entrypoint(name: str) -> StageEntrypoint:
    if not _BUILTINS_REGISTERED:
        register_builtin_stages()
    if name not in _ENTRYPOINTS and name in _BUILTIN_MODULES:
        module = importlib.import_module(_BUILTIN_MODULES[name], package=__package__)
        register_stage(name, module.run)
    try:
        return _ENTRYPOINTS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_ENTRYPOINTS))
        raise KeyError(f"Unknown MLSystem stage entrypoint: {name}. Known registry stages: {known}") from exc


def known_stages() -> list[str]:
    if not _BUILTINS_REGISTERED:
        register_builtin_stages()
    return sorted(set(_ENTRYPOINTS) | set(_BUILTIN_MODULES))


def register_builtin_stages() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True
