"""Seeded synthetic neural-population generator (documented fallback).

Ground-truth kinematics (random-target reaches with minimum-jerk velocity
profiles) drive cosine-tuned motor-cortex units (Georgopoulos tuning); spikes
are drawn from an inhomogeneous Poisson process. This is a KNOWN-ground-truth
sanity generator used only when the real dataset cannot be obtained. Every
result derived from it MUST be tagged SYNTHETIC.
"""

from __future__ import annotations

import numpy as np

from .config import SEED
from .session import SessionData

_SAMPLE_DT = 0.004  # 250 Hz, matching the O'Doherty sampling rate
_WORKSPACE_MM = 80.0
_MIN_REACH_S = 0.5
_MAX_REACH_S = 1.2


def _min_jerk(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    """Minimum-jerk position profile from p0 to p1 over n samples, (n, 2)."""
    tau = np.linspace(0.0, 1.0, n)[:, None]
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return p0[None, :] + (p1 - p0)[None, :] * s


def _make_trajectory(rng, duration_s: float):
    """Build a random-target reaching trajectory. Returns (t, pos, target)."""
    n_total = int(duration_s / _SAMPLE_DT)
    pos_chunks, target_chunks = [], []
    cur = np.zeros(2)
    n_done = 0
    while n_done < n_total:
        nxt = rng.uniform(-_WORKSPACE_MM / 2, _WORKSPACE_MM / 2, size=2)
        reach_s = rng.uniform(_MIN_REACH_S, _MAX_REACH_S)
        n = max(2, int(reach_s / _SAMPLE_DT))
        n = min(n, n_total - n_done)
        seg = _min_jerk(cur, nxt, n)
        pos_chunks.append(seg)
        target_chunks.append(np.tile(nxt, (n, 1)))
        cur = nxt
        n_done += n
    pos = np.vstack(pos_chunks)[:n_total]
    target = np.vstack(target_chunks)[:n_total]
    t = np.arange(n_total) * _SAMPLE_DT
    return t, pos, target


def _tuning_rates(vel: np.ndarray, rng, n_units: int) -> np.ndarray:
    """Cosine-tuned instantaneous firing rate (Hz) per unit, (T, n_units)."""
    speed = np.linalg.norm(vel, axis=1)  # (T,)
    direction = np.arctan2(vel[:, 1], vel[:, 0])  # (T,)
    pref_dir = rng.uniform(-np.pi, np.pi, size=n_units)
    baseline = rng.uniform(2.0, 8.0, size=n_units)  # Hz
    gain = rng.uniform(0.05, 0.25, size=n_units)  # Hz per (mm/s)
    cos_term = np.cos(direction[:, None] - pref_dir[None, :])  # (T, n_units)
    rate = baseline[None, :] + gain[None, :] * speed[:, None] * cos_term
    return np.clip(rate, 0.0, None)


def generate_session(
    duration_s: float = 360.0,
    n_units: int = 96,
    n_channels: int = 96,
    seed: int = SEED,
) -> SessionData:
    """Generate a reproducible synthetic session with known ground-truth kinematics."""
    rng = np.random.default_rng(seed)
    t, pos, target = _make_trajectory(rng, duration_s)
    vel = np.gradient(pos, _SAMPLE_DT, axis=0)  # mm/s

    rates = _tuning_rates(vel, rng, n_units)  # (T, n_units) Hz
    counts = rng.poisson(rates * _SAMPLE_DT)  # (T, n_units) spikes per sample

    spike_times: list = []
    unit_ids: list = []
    for u in range(n_units):
        idx = np.nonzero(counts[:, u])[0]
        if idx.size == 0:
            continue
        # Place each spike at a jittered time within its sample interval.
        reps = counts[idx, u]
        base = np.repeat(t[idx], reps)
        jitter = rng.uniform(0.0, _SAMPLE_DT, size=base.size)
        spike_times.append(np.sort(base + jitter))
        unit_ids.append((int(u % n_channels), 1))

    meta = {
        "generator": "cosine-tuned Poisson (Georgopoulos)",
        "seed": seed,
        "unit_selection": "synthetic_units",
    }
    return SessionData(
        t=t,
        cursor_pos=pos,
        target_pos=target,
        spike_times=spike_times,
        unit_ids=unit_ids,
        n_channels=int(n_channels),
        source=f"synthetic(seed={seed})",
        kind="SYNTHETIC",
        meta=meta,
    )
