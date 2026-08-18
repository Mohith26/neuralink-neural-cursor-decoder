"""Leakage-safe temporal train/test split (earlier time trains, later tests)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import BinnedData
from .config import TRAIN_FRACTION


@dataclass(frozen=True)
class TemporalSplit:
    """A contiguous train/test split with no shuffling across time."""

    train: BinnedData
    test: BinnedData
    split_bin: int  # first test-bin index in the original series


def _slice(b: BinnedData, lo: int, hi: int) -> BinnedData:
    return BinnedData(
        bin_times=b.bin_times[lo:hi],
        rates=b.rates[lo:hi],
        position=b.position[lo:hi],
        velocity=b.velocity[lo:hi],
        target=b.target[lo:hi],
        bin_s=b.bin_s,
        kind=b.kind,
    )


def temporal_split(binned: BinnedData, train_fraction: float = TRAIN_FRACTION) -> TemporalSplit:
    """Split into earlier-train / later-test with a hard time boundary.

    No bin appears in both sets and every test bin is strictly later than every
    train bin, so nothing from the future leaks into training.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    n = binned.n_bins
    split_bin = int(round(n * train_fraction))
    if split_bin < 2 or split_bin > n - 2:
        raise ValueError("split leaves too few bins on one side")
    train = _slice(binned, 0, split_bin)
    test = _slice(binned, split_bin, n)
    # Guarantee no temporal overlap.
    assert train.bin_times[-1] < test.bin_times[0]
    return TemporalSplit(train=train, test=test, split_bin=split_bin)
