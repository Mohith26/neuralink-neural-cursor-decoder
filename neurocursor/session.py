"""Immutable container for one recording session (real or synthetic)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SessionData:
    """One motor-cortex reaching session.

    Attributes are immutable (frozen dataclass) so downstream stages create new
    objects instead of mutating this one.

    t:            (T,) sample timestamps in seconds (monotonic).
    cursor_pos:   (T, 2) 2D cursor/hand position (mm).
    target_pos:   (T, 2) 2D reach-target position (mm); target changes = new reach.
    spike_times:  list of length n_units; each an (n_i,) array of spike times (s).
    unit_ids:     list of (channel, unit) tuples, aligned with spike_times.
    n_channels:   number of electrode channels in the array.
    source:       provenance string, e.g. "indy_20161005_06.mat".
    kind:         data honesty tag, exactly "REAL" or "SYNTHETIC".
    """

    t: np.ndarray
    cursor_pos: np.ndarray
    target_pos: np.ndarray
    spike_times: list
    unit_ids: list
    n_channels: int
    source: str
    kind: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("REAL", "SYNTHETIC"):
            raise ValueError(f"kind must be REAL or SYNTHETIC, got {self.kind!r}")
        if self.cursor_pos.ndim != 2 or self.cursor_pos.shape[1] != 2:
            raise ValueError(f"cursor_pos must be (T, 2), got {self.cursor_pos.shape}")
        if self.t.shape[0] != self.cursor_pos.shape[0]:
            raise ValueError("t and cursor_pos must share length T")
        if len(self.spike_times) != len(self.unit_ids):
            raise ValueError("spike_times and unit_ids must be the same length")

    @property
    def n_units(self) -> int:
        return len(self.spike_times)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def sample_dt(self) -> float:
        return float(np.median(np.diff(self.t)))

    def facts(self) -> dict:
        """Human-readable dataset facts for RESULTS/results JSON."""
        total_spikes = int(sum(len(s) for s in self.spike_times))
        return {
            "kind": self.kind,
            "source": self.source,
            "n_channels": int(self.n_channels),
            "n_units": int(self.n_units),
            "n_samples": int(self.t.shape[0]),
            "duration_s": round(self.duration_s, 2),
            "sample_rate_hz": round(1.0 / self.sample_dt, 1),
            "total_spikes": total_spikes,
            **self.meta,
        }
