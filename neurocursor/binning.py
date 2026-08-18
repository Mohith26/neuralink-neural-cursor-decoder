"""Bin spikes into a firing-rate feature matrix aligned to cursor kinematics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BIN_MS
from .session import SessionData


@dataclass(frozen=True)
class BinnedData:
    """Binned, time-aligned features and kinematic targets (all time-major).

    bin_times:  (B,) bin-center timestamps (s).
    rates:      (B, U) spike counts per bin per unit (firing-rate features).
    position:   (B, 2) cursor position sampled at bin centers (mm).
    velocity:   (B, 2) cursor velocity per bin (mm/s).
    target:     (B, 2) reach target at bin centers (mm).
    bin_s:      bin duration (s).
    kind:       "REAL" or "SYNTHETIC" (carried through from the session).
    """

    bin_times: np.ndarray
    rates: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    target: np.ndarray
    bin_s: float
    kind: str

    @property
    def n_bins(self) -> int:
        return self.rates.shape[0]

    @property
    def n_units(self) -> int:
        return self.rates.shape[1]


def _sample_at(t: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Linearly interpolate (T, D) values sampled at t onto query times, (Q, D)."""
    out = np.empty((query.shape[0], values.shape[1]), dtype=np.float64)
    for d in range(values.shape[1]):
        out[:, d] = np.interp(query, t, values[:, d])
    return out


def bin_session(session: SessionData, bin_ms: float = BIN_MS) -> BinnedData:
    """Bin a session into fixed windows.

    Spikes are counted in half-open bins [edge, edge+bin_s). Position/target are
    sampled at bin centers; velocity is the finite difference of binned position.
    """
    bin_s = bin_ms / 1000.0
    t = session.t
    edges = np.arange(t[0], t[-1], bin_s)
    if edges.size < 3:
        raise ValueError("session too short for the requested bin size")
    n_bins = edges.size - 1  # last edge closes the final bin
    centers = edges[:-1] + bin_s / 2.0

    # Firing-rate features: histogram spike times per unit into the bin edges.
    rates = np.zeros((n_bins, session.n_units), dtype=np.float64)
    for u, st in enumerate(session.spike_times):
        counts, _ = np.histogram(st, bins=edges)
        rates[:, u] = counts

    position = _sample_at(t, session.cursor_pos, centers)
    target = _sample_at(t, session.target_pos, centers)
    velocity = np.gradient(position, bin_s, axis=0)

    return BinnedData(
        bin_times=centers,
        rates=rates,
        position=position,
        velocity=velocity,
        target=target,
        bin_s=bin_s,
        kind=session.kind,
    )
