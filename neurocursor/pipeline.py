"""End-to-end decode pipeline: load -> bin -> split -> fit -> evaluate.

Deterministic and seeded. Produces a plain-dict result bundle that the bench
runner serializes to results/*.json. No numbers are invented here; every value
is computed from an actual decode run.
"""

from __future__ import annotations

import numpy as np

from .binning import bin_session
from .config import BIN_MS, DATA_DIR, DEFAULT_SESSION_FILE, TRAIN_FRACTION
from .kalman import KalmanVelocityDecoder
from .metrics import per_axis_metrics, shannon_bitrate
from .mlp import MLPVelocityDecoder
from .ridge import RidgeVelocityDecoder
from .shuffle import circular_shift_rates
from .split import temporal_split
from .task import acceptance_radius, segment_reaches, task_success


def load_binned_split(
    session=None,
    session_path=None,
    bin_ms: float = BIN_MS,
    train_fraction: float = TRAIN_FRACTION,
    include_hash: bool = False,
):
    """Load a session (real file, given SessionData, or synthetic) and split it."""
    if session is None:
        if session_path is None:
            session_path = DATA_DIR / DEFAULT_SESSION_FILE
        if not session_path.exists():
            raise FileNotFoundError(
                f"{session_path} missing. Run scripts/download_data.sh for the real "
                f"O'Doherty session, or pass a synthetic session."
            )
        from .loader_odoherty import load_session

        session = load_session(session_path, include_hash=include_hash)
    binned = bin_session(session, bin_ms=bin_ms)
    split = temporal_split(binned, train_fraction=train_fraction)
    return session, binned, split


def _evaluate(name, decoder, split):
    """Fit on train, decode test, and compute chance (shuffle) control."""
    decoder.fit(split.train)
    pred = decoder.decode(split.test)
    real = per_axis_metrics(split.test.velocity, pred)

    shuffled_test = circular_shift_rates(split.test)
    pred_shuf = decoder.decode(shuffled_test)
    chance = per_axis_metrics(split.test.velocity, pred_shuf)
    return {"name": name, "pred": pred, "metrics": real, "chance": chance}


def run_pipeline(
    session=None,
    session_path=None,
    bin_ms: float = BIN_MS,
    train_fraction: float = TRAIN_FRACTION,
    include_hash: bool = False,
    with_mlp: bool = True,
) -> dict:
    """Run the full decode + evaluation pipeline. Returns a result dict."""
    session, binned, split = load_binned_split(
        session, session_path, bin_ms, train_fraction, include_hash
    )

    kalman = _evaluate("kalman", KalmanVelocityDecoder(), split)
    ridge = _evaluate("ridge_wiener", RidgeVelocityDecoder(), split)
    decoders = [kalman, ridge]
    if with_mlp:
        decoders.append(_evaluate("mlp", MLPVelocityDecoder(), split))

    # Task success + bitrate on the held-out test block (best linear decoder used
    # for the headline; also reported for the true-velocity upper bound).
    radius = acceptance_radius(segment_reaches(split.test))
    task_kalman = task_success(split.test, kalman["pred"], radius)
    task_true = task_success(split.test, split.test.velocity, radius)
    bitrate_kalman = shannon_bitrate(
        task_kalman["n_success"], task_kalman["distances"], radius, task_kalman["total_time_s"]
    )

    return {
        "dataset": session.facts(),
        "bin_ms": bin_ms,
        "bin_s": binned.bin_s,
        "train_fraction": train_fraction,
        "n_bins_total": binned.n_bins,
        "n_bins_train": split.train.n_bins,
        "n_bins_test": split.test.n_bins,
        "decoders": {d["name"]: {"metrics": d["metrics"], "chance": d["chance"]} for d in decoders},
        "task_success": {
            "radius_mm": radius,
            "kalman": {k: v for k, v in task_kalman.items() if k != "distances"},
            "true_velocity_upper_bound": {
                k: v for k, v in task_true.items() if k != "distances"
            },
            "bitrate_bits_per_s_kalman": bitrate_kalman,
        },
        "_pred": {d["name"]: d["pred"] for d in decoders},  # kept for plots, not serialized
        "_split": split,
    }
