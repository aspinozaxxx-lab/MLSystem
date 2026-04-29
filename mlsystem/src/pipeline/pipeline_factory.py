from __future__ import annotations

from .prediction_pipeline import PredictionPipeline
from .training_pipeline import TrainingPipeline


class PipelineFactory:
    def training(self) -> TrainingPipeline:
        return TrainingPipeline()

    def prediction(self) -> PredictionPipeline:
        return PredictionPipeline()
