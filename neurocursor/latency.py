"""Per-bin decode-latency benchmark. Must be < bin duration to be real-time."""

from __future__ import annotations

import time

import numpy as np

from .binning import BinnedData
from .kalman import KalmanVelocityDecoder


def measure_step_latency(decoder: KalmanVelocityDecoder, test: BinnedData) -> dict:
    """Time each Kalman predict/update step (the core real-time decode work).

    Reports p50/p95/p99/max in milliseconds and the real-time margin vs the bin
    duration. This times the decode compute only (the hot path a C++ port would
    replace), excluding any HTTP framing.
    """
    decoder.reset(test.position[0])
    times_ms = np.empty(test.n_bins, dtype=np.float64)
    for i in range(test.n_bins):
        z = test.rates[i]
        t0 = time.perf_counter()
        decoder.step(z)
        times_ms[i] = (time.perf_counter() - t0) * 1e3
    bin_ms = test.bin_s * 1e3
    p50, p95, p99 = np.percentile(times_ms, [50, 95, 99])
    return {
        "n_bins": int(test.n_bins),
        "n_units": int(test.n_units),
        "bin_ms": float(bin_ms),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(times_ms.max()),
        "mean_ms": float(times_ms.mean()),
        "realtime_p95": bool(p95 < bin_ms),
        "realtime_margin_x": float(bin_ms / p95) if p95 > 0 else float("inf"),
        "throughput_units_x_hz": float(test.n_units / test.bin_s),
    }
