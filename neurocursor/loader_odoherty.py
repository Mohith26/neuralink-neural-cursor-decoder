"""Loader for the O'Doherty et al. 2017 NHP reaching dataset (.mat, HDF5 v7.3).

Zenodo DOI 10.5281/zenodo.583331. Each per-session .mat holds:
  t          (1, T)   timestamps (s), ~250 Hz
  cursor_pos (2, T)   2D cursor position (mm)
  target_pos (2, T)   2D reach target (mm)
  finger_pos (3, T)   finger position (unused here)
  spikes     (5, 96)  cell array [unit, channel]; each cell = spike times (s)
                      unit row 0 = unsorted threshold-crossing "hash";
                      unit rows 1-4 = sorted single units.
  chan_names (1, 96)  channel labels.

Read with h5py because MATLAB v7.3 files are HDF5. All returned arrays are
time-major (T, ...) — h5py exposes MATLAB arrays transposed, so we transpose back.
"""

from __future__ import annotations

import numpy as np

from .session import SessionData

# Unit-row 0 is the unsorted threshold-crossing hash; rows 1-4 are sorted units.
_HASH_UNIT_ROW = 0


def _deref_spike_times(f, ref) -> np.ndarray:
    """Dereference one spikes-cell HDF5 reference into a 1D spike-time array.

    Empty cells are stored as non-float placeholder arrays; treat as no spikes.
    """
    obj = np.array(f[ref])
    if obj.dtype.kind != "f" or obj.size == 0:
        return np.empty(0, dtype=np.float64)
    return obj.ravel().astype(np.float64)


def load_session(path, include_hash: bool = False) -> SessionData:
    """Load one O'Doherty session file into an immutable SessionData.

    include_hash=False (default) keeps only sorted single units (rows 1-4),
    the field-standard choice for this dataset. Set True to also include the
    per-channel unsorted threshold-crossing hash (row 0).
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        t = np.array(f["t"]).ravel().astype(np.float64)
        cursor_pos = np.array(f["cursor_pos"]).T.astype(np.float64)  # (T, 2)
        target_pos = np.array(f["target_pos"]).T.astype(np.float64)  # (T, 2)

        spikes = f["spikes"]  # (n_unit_rows, n_channels) object refs
        n_unit_rows, n_channels = spikes.shape

        spike_times: list = []
        unit_ids: list = []
        for chan in range(n_channels):
            for unit in range(n_unit_rows):
                if unit == _HASH_UNIT_ROW and not include_hash:
                    continue
                st = _deref_spike_times(f, spikes[unit, chan])
                if st.size == 0:
                    continue
                # Keep only spikes within the kinematic recording window.
                st = st[(st >= t[0]) & (st <= t[-1])]
                if st.size == 0:
                    continue
                spike_times.append(np.sort(st))
                unit_ids.append((int(chan), int(unit)))

    meta = {
        "include_hash": include_hash,
        "unit_selection": "sorted_units_1-4" if not include_hash else "hash+sorted_0-4",
    }
    return SessionData(
        t=t,
        cursor_pos=cursor_pos,
        target_pos=target_pos,
        spike_times=spike_times,
        unit_ids=unit_ids,
        n_channels=int(n_channels),
        source=str(getattr(path, "name", path)),
        kind="REAL",
        meta=meta,
    )
