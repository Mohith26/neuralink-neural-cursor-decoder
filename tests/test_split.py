"""Temporal-split no-leakage tests."""

import numpy as np
import pytest

from neurocursor.split import temporal_split


def test_split_has_no_temporal_overlap(binned):
    s = temporal_split(binned, train_fraction=0.75)
    # Every test bin is strictly later than every train bin.
    assert s.train.bin_times.max() < s.test.bin_times.min()


def test_split_partitions_all_bins(binned):
    s = temporal_split(binned, train_fraction=0.75)
    assert s.train.n_bins + s.test.n_bins == binned.n_bins


def test_split_is_contiguous_in_time(binned):
    s = temporal_split(binned, train_fraction=0.75)
    combined = np.concatenate([s.train.bin_times, s.test.bin_times])
    assert np.array_equal(combined, binned.bin_times)  # order preserved, no shuffle


def test_split_respects_fraction(binned):
    s = temporal_split(binned, train_fraction=0.8)
    assert abs(s.train.n_bins / binned.n_bins - 0.8) < 0.01


def test_invalid_fraction_rejected(binned):
    with pytest.raises(ValueError):
        temporal_split(binned, train_fraction=1.5)
