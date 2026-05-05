from __future__ import annotations

import numpy as np


def weighted_average(probability_arrays: list[np.ndarray], weight_arrays: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Blend overlapping probability arrays with explicit weights.

    The current block_parallel MVP passes one scene-level probability map per
    block. This helper keeps the contract ready for tile-level Rabbit workers.
    """
    if not probability_arrays:
        raise ValueError("No probability arrays to blend")
    if weight_arrays is None:
        weight_arrays = [np.ones_like(item, dtype=np.float32) for item in probability_arrays]
    prob_sum = np.zeros_like(probability_arrays[0], dtype=np.float32)
    weight_sum = np.zeros_like(probability_arrays[0], dtype=np.float32)
    for prob, weight in zip(probability_arrays, weight_arrays, strict=True):
        prob_sum += prob.astype(np.float32) * weight.astype(np.float32)
        weight_sum += weight.astype(np.float32)
    blended = np.zeros_like(prob_sum, dtype=np.float32)
    np.divide(prob_sum, weight_sum, out=blended, where=weight_sum > 0)
    return blended, weight_sum
