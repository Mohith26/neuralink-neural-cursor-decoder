"""Real O'Doherty loader test. Skipped when the (gitignored) file is absent."""

import pytest

from neurocursor.config import DATA_DIR, DEFAULT_SESSION_FILE

_REAL_PATH = DATA_DIR / DEFAULT_SESSION_FILE
_pytestmark_reason = f"{_REAL_PATH} not downloaded (run scripts/download_data.sh)"
pytestmark = pytest.mark.skipif(not _REAL_PATH.exists(), reason=_pytestmark_reason)


def test_real_session_loads_and_is_tagged_real():
    from neurocursor.loader_odoherty import load_session

    s = load_session(_REAL_PATH)
    assert s.kind == "REAL"
    assert s.n_channels == 96
    assert s.n_units > 0
    assert s.cursor_pos.shape[1] == 2
    assert s.duration_s > 60.0
    # Spikes fall within the kinematic recording window.
    assert s.spike_times[0].min() >= s.t[0] - 1e-6
    assert s.spike_times[0].max() <= s.t[-1] + 1e-6
