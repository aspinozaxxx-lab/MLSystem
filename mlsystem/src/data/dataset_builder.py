from __future__ import annotations

from typing import Any

import numpy as np


def summarize_samples(samples: list[tuple[np.ndarray, np.ndarray]], *, prefix: str) -> dict[str, Any]:
    positive = sum(int(mask.sum() > 0) for _image, mask in samples)
    negative = sum(int(mask.sum() == 0) for _image, mask in samples)
    return {
        f"{prefix}_tile_count": len(samples),
        f"{prefix}_positive_tiles": positive,
        f"{prefix}_negative_tiles": negative,
    }
