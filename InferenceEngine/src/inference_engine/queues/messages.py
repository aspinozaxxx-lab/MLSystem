from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


QUEUE_NAMES = [
    "ie.jobs.submit",
    "ie.scene.plan",
    "ie.tile.preprocess",
    "ie.tile.infer",
    "ie.tile.done",
    "ie.block.ready",
    "ie.block.vectorize",
    "ie.block.done",
    "ie.scene.merge",
    "ie.job.finalize",
    "ie.events",
    "ie.dead_letter",
]


@dataclass(frozen=True)
class QueueMessage:
    schema_version: int
    message_id: str
    job_id: str
    stage: str
    payload: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QueueMessage":
        return cls(
            schema_version=int(payload.get("schema_version") or 1),
            message_id=str(payload["message_id"]),
            job_id=str(payload["job_id"]),
            stage=str(payload["stage"]),
            payload=dict(payload.get("payload") or {}),
            attempt=int(payload.get("attempt") or 0),
        )

    @classmethod
    def from_json(cls, text: str | bytes) -> "QueueMessage":
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return cls.from_dict(json.loads(text))


def deterministic_message_id(job_id: str, stage: str, scene_id: str | None = None, tile_id: str | None = None, block_id: str | None = None) -> str:
    parts = [job_id, scene_id, tile_id, block_id, stage]
    return "/".join(str(part) for part in parts if part)


def make_message(
    *,
    job_id: str,
    stage: str,
    payload: dict[str, Any] | None = None,
    scene_id: str | None = None,
    tile_id: str | None = None,
    block_id: str | None = None,
    attempt: int = 0,
) -> QueueMessage:
    return QueueMessage(
        schema_version=1,
        message_id=deterministic_message_id(job_id, stage, scene_id=scene_id, tile_id=tile_id, block_id=block_id),
        job_id=job_id,
        stage=stage,
        payload=payload or {},
        attempt=attempt,
    )


def validate_queue_contracts() -> dict[str, Any]:
    required = set(QUEUE_NAMES)
    missing = required - set(QUEUE_NAMES)
    sample = make_message(job_id="job", stage="tile.infer", scene_id="scene", tile_id="tile")
    decoded = QueueMessage.from_json(sample.to_json())
    if decoded.message_id != "job/scene/tile/tile.infer":
        raise AssertionError("deterministic message id contract changed")
    return {"queues": sorted(required), "missing": sorted(missing), "sample_message_id": decoded.message_id}
