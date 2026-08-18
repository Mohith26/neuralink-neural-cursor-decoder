"""Integrate decoded velocity into a cursor position trajectory."""

from __future__ import annotations

import numpy as np


def integrate_velocity(
    velocity: np.ndarray, bin_s: float, init_position: np.ndarray
) -> np.ndarray:
    """Euler-integrate (B, 2) velocity (mm/s) into (B, 2) position (mm).

    position[i] = init_position + cumulative sum of velocity * bin_s up to i.
    """
    if velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError(f"velocity must be (B, 2), got {velocity.shape}")
    steps = np.cumsum(velocity * bin_s, axis=0)
    return init_position[None, :] + steps
