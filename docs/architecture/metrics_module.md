# metrics

## Purpose

`metrics` owns reusable model-quality metric calculations. It exposes only the metric surface needed by production modules and keeps geometry/debug helpers internal.

## Public API

```python
from mlsystem.src.metrics.api import (
    MetricsError,
    PixelMetricAccumulator,
    compute_pixel_metrics_from_logits,
)
```

### `compute_pixel_metrics_from_logits(logits, target, *, threshold=0.5, class_index=1, ignore_index=None)`

- `logits`: model output tensor.
- `target`: target mask tensor.
- `threshold`: probability threshold for binary mask.
- `class_index`: positive class index for multi-channel logits.
- `ignore_index`: optional target value ignored in counters.

### `PixelMetricAccumulator(threshold=0.5, ignore_index=None)`

- Accumulates pixel metrics over validation batches.

### `MetricsError`

- Public metrics module exception.

## Forbidden Overlaps

- No training loop or loss accumulation ownership: that belongs to `train`.
- No MLflow logging.
- No vectorization/postprocess/pseudolabel export.
- Object geometry helpers and metric artifact writers are internal unless a production module needs a new public boundary.
