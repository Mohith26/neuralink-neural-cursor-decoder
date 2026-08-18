"""Closed-loop-style task-success proxy (offline, open-loop integration).

Reaches are segments between target changes. For each held-out reach we place a
virtual cursor at the true reach-start position and integrate the DECODED
velocity forward over the reach window; the reach "succeeds" if the virtual
cursor's closest approach to the target falls within an acceptance radius. This
is an OFFLINE open-loop proxy, NOT a live closed loop with a subject.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import BinnedData
from .config import ACCEPT_RADIUS_FRACTION, MIN_REACH_DISTANCE_MM
from .integrate import integrate_velocity


@dataclass(frozen=True)
class Reach:
    start: int  # bin index of target onset (inclusive)
    stop: int  # bin index of reach end (exclusive)
    target: np.ndarray  # (2,)
    start_pos: np.ndarray  # (2,) true cursor position at onset
    distance: float  # start-to-target distance (mm)


def segment_reaches(binned: BinnedData, min_distance: float = MIN_REACH_DISTANCE_MM) -> list:
    """Split a binned block into reaches at target changes.

    Micro-reaches whose start-to-target distance is below min_distance are
    dropped (they carry negligible task information).
    """
    target = binned.target
    changed = np.any(np.abs(np.diff(target, axis=0)) > 1e-6, axis=1)
    boundaries = [0, *(np.nonzero(changed)[0] + 1).tolist(), binned.n_bins]
    reaches: list = []
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if hi - lo < 2:
            continue
        tgt = target[lo]
        start_pos = binned.position[lo]
        dist = float(np.linalg.norm(tgt - start_pos))
        if dist < min_distance:
            continue
        reaches.append(Reach(start=lo, stop=hi, target=tgt, start_pos=start_pos, distance=dist))
    return reaches


def acceptance_radius(reaches: list, fraction: float = ACCEPT_RADIUS_FRACTION) -> float:
    """Acceptance radius = fraction of the median reach distance (mm)."""
    if not reaches:
        return 0.0
    med = float(np.median([r.distance for r in reaches]))
    return fraction * med


def task_success(binned: BinnedData, decoded_velocity: np.ndarray, radius: float) -> dict:
    """Fraction of held-out reaches acquired within radius by the decoded cursor.

    Returns success rate, counts, per-success distances, and total time — the
    inputs the Shannon-bitrate estimate needs.
    """
    reaches = segment_reaches(binned)
    if not reaches:
        return {"n_reaches": 0, "n_success": 0, "success_rate": 0.0,
                "distances": np.empty(0), "total_time_s": 0.0, "radius_mm": radius}
    n_success = 0
    success_distances = []
    for r in reaches:
        vel = decoded_velocity[r.start : r.stop]
        traj = integrate_velocity(vel, binned.bin_s, r.start_pos)
        closest = float(np.min(np.linalg.norm(traj - r.target[None, :], axis=1)))
        if closest <= radius:
            n_success += 1
            success_distances.append(r.distance)
    total_time_s = float(binned.n_bins * binned.bin_s)
    return {
        "n_reaches": len(reaches),
        "n_success": n_success,
        "success_rate": n_success / len(reaches),
        "distances": np.array(success_distances),
        "total_time_s": total_time_s,
        "radius_mm": radius,
    }
