from __future__ import annotations

from dataclasses import dataclass, field

from ..planning.planner import BlockDescriptor, ScenePlan


@dataclass
class DependencyTracker:
    remaining_by_block: dict[str, set[str]] = field(default_factory=dict)
    blocks_by_tile: dict[str, list[str]] = field(default_factory=dict)
    ready_blocks: set[str] = field(default_factory=set)
    done_tiles: set[str] = field(default_factory=set)

    @classmethod
    def from_scene_plan(cls, plan: ScenePlan) -> "DependencyTracker":
        tracker = cls()
        for block in plan.blocks:
            deps = set(block.dependency_tile_ids)
            tracker.remaining_by_block[block.block_id] = deps
            if not deps:
                tracker.ready_blocks.add(block.block_id)
            for tile_id in deps:
                tracker.blocks_by_tile.setdefault(tile_id, []).append(block.block_id)
        return tracker

    def mark_tile_done(self, tile_id: str) -> list[str]:
        if tile_id in self.done_tiles:
            return []
        self.done_tiles.add(tile_id)
        newly_ready: list[str] = []
        for block_id in self.blocks_by_tile.get(tile_id, []):
            remaining = self.remaining_by_block.setdefault(block_id, set())
            remaining.discard(tile_id)
            if not remaining and block_id not in self.ready_blocks:
                self.ready_blocks.add(block_id)
                newly_ready.append(block_id)
        return sorted(newly_ready)

    def block(self, block_id: str, plan: ScenePlan) -> BlockDescriptor:
        for block in plan.blocks:
            if block.block_id == block_id:
                return block
        raise KeyError(block_id)
