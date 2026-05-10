from __future__ import annotations

from dataclasses import dataclass

from ..api.schemas import ResourceConfig


@dataclass
class BackpressureState:
    paused: bool = False
    preprocess_pauses_total: int = 0
    preprocess_resumes_total: int = 0


class AdaptiveProducer:
    def __init__(self, resource: ResourceConfig) -> None:
        self.resource = resource
        self.state = BackpressureState()
        target = resource.triton_batch_size * max(2, resource.triton_instance_count * resource.batches_ahead)
        self.target_ready_tiles = int(target)
        self.high_watermark = max(1, int(resource.max_preprocess_queue))
        self.low_watermark = max(1, self.high_watermark // 2)

    def should_pause(self, *, infer_ready: int, infer_unacked: int, spool_bytes: int) -> bool:
        depth = int(infer_ready) + int(infer_unacked)
        should = depth > self.high_watermark or int(spool_bytes) > int(self.resource.max_spool_bytes)
        if should and not self.state.paused:
            self.state.paused = True
            self.state.preprocess_pauses_total += 1
        return self.state.paused

    def should_resume(self, *, infer_ready: int, infer_unacked: int, spool_bytes: int) -> bool:
        depth = int(infer_ready) + int(infer_unacked)
        should = depth <= self.low_watermark and int(spool_bytes) <= int(self.resource.max_spool_bytes)
        if should and self.state.paused:
            self.state.paused = False
            self.state.preprocess_resumes_total += 1
        return not self.state.paused
