from __future__ import annotations

import numpy as np


def threshold_probability_map(prob_map: np.ndarray, threshold: float) -> np.ndarray:
    return (prob_map >= float(threshold)).astype("uint8")
