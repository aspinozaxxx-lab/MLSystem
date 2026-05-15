from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any


def filter_geometries_linear(geometries: list[Any], window_bounds: Any) -> list[Any]:
    return [geom for geom in geometries if _usable_geometry(geom) and geom.intersects(window_bounds)]


@dataclass
class GeometryWindowIndex:
    geometries: list[Any]
    _tree: Any = None
    _tree_items: list[Any] | None = None
    _id_to_index: dict[int, int] | None = None

    @classmethod
    def build(cls, geometries: list[Any]) -> "GeometryWindowIndex":
        usable = [(index, geom) for index, geom in enumerate(geometries) if _usable_geometry(geom)]
        index = cls(geometries=list(geometries))
        if not usable:
            index._tree_items = []
            index._id_to_index = {}
            return index
        try:
            from shapely.strtree import STRtree

            tree_items = [geom for _original_index, geom in usable]
            index._tree = STRtree(tree_items)
            index._tree_items = tree_items
            index._id_to_index = {id(geom): original_index for original_index, geom in usable}
        except Exception:  # noqa: BLE001
            index._tree = None
            index._tree_items = None
            index._id_to_index = None
        return index

    def query(self, window_bounds: Any) -> list[Any]:
        if self._tree is None or self._tree_items is None:
            return filter_geometries_linear(self.geometries, window_bounds)
        try:
            candidates = self._tree.query(window_bounds)
        except Exception:  # noqa: BLE001
            return filter_geometries_linear(self.geometries, window_bounds)

        selected: list[tuple[int, Any]] = []
        for item in candidates:
            original_index: int | None = None
            geom: Any
            if isinstance(item, Integral) or hasattr(item, "__index__"):
                original_index = int(item)
                geom = self._tree_items[original_index]
                original_index = self._original_index_for_geometry(geom, fallback=original_index)
            else:
                geom = item
                original_index = self._original_index_for_geometry(geom)
            if original_index is None or not _usable_geometry(geom):
                continue
            try:
                if geom.intersects(window_bounds):
                    selected.append((original_index, geom))
            except Exception:  # noqa: BLE001
                continue
        selected.sort(key=lambda pair: pair[0])
        return [geom for _index, geom in selected]

    def _original_index_for_geometry(self, geom: Any, *, fallback: int | None = None) -> int | None:
        if self._id_to_index is None:
            return fallback
        return self._id_to_index.get(id(geom), fallback)


def _usable_geometry(geom: Any) -> bool:
    try:
        return bool(geom is not None and geom.is_valid and not geom.is_empty)
    except Exception:  # noqa: BLE001
        return False
