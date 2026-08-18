"""Feature construction: neural history taps for the Wiener/ridge decoder.

Motor-cortex activity leads movement by ~100-150 ms, so a linear (Wiener)
velocity decoder benefits from a short window of current + past neural bins.
"""

from __future__ import annotations

import numpy as np

from .config import WIENER_TAPS


def make_tap_features(rates: np.ndarray, taps: int = WIENER_TAPS) -> np.ndarray:
    """Stack current + (taps-1) past bins of firing rates.

    rates: (B, U). Returns (B, U*taps). Rows before enough history exist are
    zero-padded so the output length matches the input (no bins dropped, no
    look-ahead into the future).
    """
    if taps < 1:
        raise ValueError("taps must be >= 1")
    b, u = rates.shape
    out = np.zeros((b, u * taps), dtype=np.float64)
    for k in range(taps):
        # Column block k holds rates shifted k bins into the past (causal).
        if k == 0:
            out[:, :u] = rates
            continue
        out[k:, k * u : (k + 1) * u] = rates[:-k]
    return out
