"""Shuffle / chance control: destroy the neural<->kinematic correspondence.

Circularly shifting the firing-rate matrix by half its length relative to the
kinematics preserves each unit's marginal firing statistics and temporal
autocorrelation but removes any real relationship to movement. A decoder fit on
this shifted data measures the chance/floor performance the real decoder must
beat decisively.
"""

from __future__ import annotations

import numpy as np

from .binning import BinnedData


def circular_shift_rates(binned: BinnedData, shift: int | None = None) -> BinnedData:
    """Return a copy of `binned` with its firing rates circularly time-shifted.

    Kinematics stay put; only the rates move, breaking the causal link. A new
    BinnedData is returned (no mutation of the input).
    """
    n = binned.n_bins
    if shift is None:
        shift = n // 2
    shift %= n
    shifted = np.roll(binned.rates, shift, axis=0)
    return BinnedData(
        bin_times=binned.bin_times,
        rates=shifted,
        position=binned.position,
        velocity=binned.velocity,
        target=binned.target,
        bin_s=binned.bin_s,
        kind=binned.kind,
    )
