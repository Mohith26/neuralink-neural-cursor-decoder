"""Binning + feature-shape tests."""

import numpy as np

from neurocursor.binning import bin_session
from neurocursor.features import make_tap_features


def test_bin_shapes_align(session):
    b = bin_session(session, bin_ms=64.0)
    assert b.rates.shape[0] == b.n_bins
    assert b.rates.shape[1] == session.n_units
    assert b.position.shape == (b.n_bins, 2)
    assert b.velocity.shape == (b.n_bins, 2)
    assert b.target.shape == (b.n_bins, 2)
    assert b.bin_times.shape == (b.n_bins,)


def test_bin_duration_matches_request(session):
    b = bin_session(session, bin_ms=64.0)
    assert abs(b.bin_s - 0.064) < 1e-12


def test_spike_counts_are_nonnegative_integers(session):
    b = bin_session(session)
    assert (b.rates >= 0).all()
    assert np.allclose(b.rates, np.round(b.rates))


def test_total_binned_counts_do_not_exceed_spikes(session):
    b = bin_session(session)
    total_binned = b.rates.sum()
    total_spikes = sum(len(s) for s in session.spike_times)
    # Every binned spike is a real spike (bins tile the recording window).
    assert 0 < total_binned <= total_spikes


def test_tap_features_shape_and_causality():
    rates = np.arange(20, dtype=float).reshape(10, 2)
    feats = make_tap_features(rates, taps=3)
    assert feats.shape == (10, 6)
    # First block is the current bin.
    assert np.array_equal(feats[:, :2], rates)
    # Row 0 has no history -> lagged blocks are zero (no look-ahead).
    assert np.array_equal(feats[0, 2:], np.zeros(4))
    # Row 5's first-lag block equals row 4's rates (causal shift).
    assert np.array_equal(feats[5, 2:4], rates[4])
