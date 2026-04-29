from __future__ import annotations

import numpy as np

from ..tiling.debug_tiling import probability_nonzero_bbox


def seam_jump_stats(prob_map: np.ndarray, stride: int) -> dict[str, float]:
    if prob_map.size == 0 or stride <= 0:
        return {"mean_seam_jump": 0.0, "max_seam_jump": 0.0}
    jumps: list[float] = []
    for x in range(stride, prob_map.shape[1], stride):
        jumps.append(float(np.mean(np.abs(prob_map[:, x] - prob_map[:, x - 1]))))
    for y in range(stride, prob_map.shape[0], stride):
        jumps.append(float(np.mean(np.abs(prob_map[y, :] - prob_map[y - 1, :]))))
    return {"mean_seam_jump": float(np.mean(jumps)) if jumps else 0.0, "max_seam_jump": float(np.max(jumps)) if jumps else 0.0}


__all__ = ["probability_nonzero_bbox", "seam_jump_stats"]
