"""Evaluation-math unit tests on toy inputs with known answers."""

import numpy as np

from neurocursor.metrics import (
    correlation,
    per_axis_metrics,
    r2_score,
    rmse,
    shannon_bitrate,
)


def test_correlation_perfect_and_anti():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(correlation(x, x) - 1.0) < 1e-12
    assert abs(correlation(x, -x) + 1.0) < 1e-12


def test_correlation_constant_is_zero():
    x = np.array([1.0, 2.0, 3.0])
    assert correlation(x, np.zeros(3)) == 0.0


def test_r2_perfect_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(r2_score(y, y) - 1.0) < 1e-12
    # Predicting the mean everywhere gives R^2 = 0.
    assert abs(r2_score(y, np.full_like(y, y.mean()))) < 1e-12


def test_r2_matches_sklearn():
    from sklearn.metrics import r2_score as sk_r2

    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    p = y + rng.normal(scale=0.3, size=50)
    assert abs(r2_score(y, p) - sk_r2(y, p)) < 1e-9


def test_rmse_known_value():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([3.0, 4.0, 0.0])
    assert abs(rmse(y, p) - np.sqrt((9 + 16) / 3)) < 1e-12


def test_per_axis_bundle_keys():
    y = np.random.default_rng(1).normal(size=(30, 2))
    p = y + 0.1
    m = per_axis_metrics(y, p)
    for k in ("cc_x", "cc_y", "r2_x", "r2_y", "rmse_x", "rmse_y",
              "cc_mean", "r2_mean", "rmse_mean"):
        assert k in m


def test_bitrate_known_value():
    # 2 successes, distance = radius -> ID = log2(2) = 1 bit each, 4 s total.
    dist = np.array([10.0, 10.0])
    br = shannon_bitrate(n_success=2, distances=dist, radius=10.0, total_time_s=4.0)
    assert abs(br - (1.0 * 2) / 4.0) < 1e-12


def test_bitrate_zero_when_no_success():
    assert shannon_bitrate(0, np.array([]), 10.0, 5.0) == 0.0
