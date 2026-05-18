from __future__ import annotations

from dataclasses import dataclass

from .contracts import TrainProgressEvent, TrainProgressSink


@dataclass
class EarlyStopping:
    patience: int
    best: float = float("-inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0

    def update(self, value: float, epoch: int) -> bool:
        if value > self.best:
            self.best = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return False
        self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def emit_progress(sink: TrainProgressSink | None, event: TrainProgressEvent) -> None:
    if sink is not None:
        sink.emit(event)
