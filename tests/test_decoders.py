"""Decoder shape, determinism, chance-beating, and integration tests."""

import numpy as np

from neurocursor.integrate import integrate_velocity
from neurocursor.kalman import KalmanVelocityDecoder
from neurocursor.metrics import per_axis_metrics
from neurocursor.ridge import RidgeVelocityDecoder
from neurocursor.shuffle import circular_shift_rates


def test_kalman_output_shape(split):
    dec = KalmanVelocityDecoder().fit(split.train)
    pred = dec.decode(split.test)
    assert pred.shape == (split.test.n_bins, 2)


def test_ridge_output_shape(split):
    dec = RidgeVelocityDecoder().fit(split.train)
    pred = dec.decode(split.test)
    assert pred.shape == (split.test.n_bins, 2)


def test_kalman_deterministic(split):
    p1 = KalmanVelocityDecoder().fit(split.train).decode(split.test)
    p2 = KalmanVelocityDecoder().fit(split.train).decode(split.test)
    assert np.array_equal(p1, p2)


def test_ridge_deterministic(split):
    p1 = RidgeVelocityDecoder().fit(split.train).decode(split.test)
    p2 = RidgeVelocityDecoder().fit(split.train).decode(split.test)
    assert np.array_equal(p1, p2)


def test_kalman_beats_chance_decisively(split):
    dec = KalmanVelocityDecoder().fit(split.train)
    real = per_axis_metrics(split.test.velocity, dec.decode(split.test))
    chance = per_axis_metrics(
        split.test.velocity, dec.decode(circular_shift_rates(split.test))
    )
    # Real decode must clear a strong margin over the shuffle control.
    assert real["cc_mean"] > 0.5
    assert real["cc_mean"] > chance["cc_mean"] + 0.4
    assert abs(chance["cc_mean"]) < 0.2


def test_ridge_beats_chance_decisively(split):
    dec = RidgeVelocityDecoder().fit(split.train)
    real = per_axis_metrics(split.test.velocity, dec.decode(split.test))
    chance = per_axis_metrics(
        split.test.velocity, dec.decode(circular_shift_rates(split.test))
    )
    assert real["cc_mean"] > 0.5
    assert real["cc_mean"] > chance["cc_mean"] + 0.4


def test_integration_recovers_position():
    vel = np.tile(np.array([1.0, -2.0]), (5, 1))
    pos = integrate_velocity(vel, bin_s=0.5, init_position=np.array([10.0, 20.0]))
    # After one 0.5 s step: 10 + 1*0.5 = 10.5 ; 20 + (-2)*0.5 = 19.0
    assert np.allclose(pos[0], [10.5, 19.0])
    assert np.allclose(pos[-1], [10.0 + 1.0 * 0.5 * 5, 20.0 - 2.0 * 0.5 * 5])
