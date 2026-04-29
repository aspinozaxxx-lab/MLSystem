from __future__ import annotations

import numpy as np


def probability_nonzero_bbox(prob_map: np.ndarray) -> list[int] | None:
    nz = np.argwhere(prob_map > 1e-6)
    if not nz.size:
        return None
    y0, x0 = nz.min(axis=0)
    y1, x1 = nz.max(axis=0)
    return [int(x0), int(y0), int(x1) + 1, int(y1) + 1]
