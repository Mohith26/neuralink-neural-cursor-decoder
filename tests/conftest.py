"""Shared fixtures. Tests run on the fast seeded synthetic session so they never
require the (gitignored) real dataset download."""

import pytest

from neurocursor.binning import bin_session
from neurocursor.split import temporal_split
from neurocursor.synthetic import generate_session


@pytest.fixture(scope="session")
def session():
    return generate_session(duration_s=120.0, n_units=48, seed=42)


@pytest.fixture(scope="session")
def binned(session):
    return bin_session(session)


@pytest.fixture(scope="session")
def split(binned):
    return temporal_split(binned)
