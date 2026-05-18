from __future__ import annotations

from .contracts import MetricsError


def compute_pixel_metrics_from_logits(*args, **kwargs) -> dict[str, float | int]:
    from .segmentation import binary_segmentation_metrics_from_logits

    return binary_segmentation_metrics_from_logits(*args, **kwargs)


class PixelMetricAccumulator:
    def __new__(cls, *args, **kwargs):
        from .segmentation import PixelMetricAccumulator as _PixelMetricAccumulator

        return _PixelMetricAccumulator(*args, **kwargs)


__all__ = [
    "MetricsError",
    "PixelMetricAccumulator",
    "compute_pixel_metrics_from_logits",
]
