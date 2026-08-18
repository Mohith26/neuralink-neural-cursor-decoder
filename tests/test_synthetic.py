"""Synthetic generator: determinism + honesty-tag tests."""

import numpy as np

from neurocursor.synthetic import generate_session
from neurocursor.task import acceptance_radius, segment_reaches, task_success


def test_synthetic_is_seeded_deterministic():
    a = generate_session(duration_s=30.0, n_units=16, seed=7)
    b = generate_session(duration_s=30.0, n_units=16, seed=7)
    assert np.array_equal(a.cursor_pos, b.cursor_pos)
    assert len(a.spike_times) == len(b.spike_times)
    assert np.array_equal(a.spike_times[0], b.spike_times[0])


def test_synthetic_different_seed_differs():
    a = generate_session(duration_s=30.0, n_units=16, seed=7)
    b = generate_session(duration_s=30.0, n_units=16, seed=8)
    assert not np.array_equal(a.cursor_pos, b.cursor_pos)


def test_synthetic_tagged_synthetic():
    s = generate_session(duration_s=20.0, n_units=8, seed=1)
    assert s.kind == "SYNTHETIC"
    assert s.facts()["kind"] == "SYNTHETIC"


def test_true_velocity_reaches_targets(binned):
    # Integrating the TRUE velocity should acquire essentially every reach —
    # a sanity check on the task-success proxy machinery.
    reaches = segment_reaches(binned)
    radius = acceptance_radius(reaches)
    res = task_success(binned, binned.velocity, radius)
    assert res["success_rate"] > 0.95
