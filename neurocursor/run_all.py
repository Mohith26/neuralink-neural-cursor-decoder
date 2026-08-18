"""Run the full decode benchmark and write results/*.json (+ figures).

Usage:
    python -m neurocursor.run_all              # REAL O'Doherty session (default)
    python -m neurocursor.run_all --synthetic  # seeded synthetic fallback

Every number written here comes from an actual decode run on this machine.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import date

import numpy as np

from .binning import bin_session
from .config import BIN_MS, DATA_DIR, DEFAULT_SESSION_FILE, RESULTS_DIR
from .kalman import KalmanVelocityDecoder
from .latency import measure_step_latency
from .pipeline import run_pipeline
from .replay import replay
from .split import temporal_split


def _round(obj):
    """Recursively round floats for stable, human-readable JSON."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return _round(obj.item())
    return obj


def _write(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(_round(payload), indent=2) + "\n")
    print(f"  wrote {path.relative_to(RESULTS_DIR.parent)}")


def _load_session(use_synthetic: bool):
    if use_synthetic:
        from .synthetic import generate_session

        return generate_session()
    from .loader_odoherty import load_session

    path = DATA_DIR / DEFAULT_SESSION_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run scripts/download_data.sh, or pass --synthetic."
        )
    return load_session(path)


def _latency_sweep(session, bin_sizes=(32.0, 50.0, 64.0, 100.0)) -> list:
    """Measure per-bin Kalman decode latency across bin sizes on held-out data."""
    rows = []
    for bin_ms in bin_sizes:
        binned = bin_session(session, bin_ms=bin_ms)
        split = temporal_split(binned)
        dec = KalmanVelocityDecoder().fit(split.train)
        rows.append(measure_step_latency(dec, split.test))
    return rows


def _plot(res: dict, session, kind: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"  (skipping figures: {exc})")
        return

    split = res["_split"]
    kal = res["_pred"]["kalman"]
    true = split.test.velocity
    t = split.test.bin_times - split.test.bin_times[0]
    win = slice(0, min(400, split.test.n_bins))

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for i, name in enumerate(("vx", "vy")):
        ax[i].plot(t[win], true[win, i], color="k", lw=1.3, label="true")
        ax[i].plot(t[win], kal[win, i], color="C3", lw=1.1, alpha=0.85, label="Kalman decoded")
        ax[i].set_ylabel(f"{name} (mm/s)")
        ax[i].legend(loc="upper right", fontsize=8)
    ax[1].set_xlabel("time (s)")
    ax[0].set_title(f"Decoded vs true cursor velocity — {kind} ({session.source})")
    fig.tight_layout()
    out_path = RESULTS_DIR / "velocity_traces.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(RESULTS_DIR.parent)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="use the seeded synthetic fallback")
    args = parser.parse_args()

    session = _load_session(args.synthetic)
    kind = session.kind
    print(f"[NeuroCursor] {kind} data: {session.source} — {session.n_units} units, "
          f"{session.duration_s:.0f} s")

    res = run_pipeline(session=session, with_mlp=True)

    # Verify the streaming service reproduces the batch Kalman decode.
    split = res["_split"]
    kal = KalmanVelocityDecoder().fit(split.train)
    streamed = replay(kal, split.test)
    replay_max_diff = float(np.max(np.abs(streamed - res["_pred"]["kalman"])))

    latency_rows = _latency_sweep(session)
    default_lat = next(r for r in latency_rows if abs(r["bin_ms"] - BIN_MS) < 1e-6)

    meta = {
        "measured_at": str(date.today()),
        "data_kind": kind,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }

    _write("dataset.json", {**meta, **res["dataset"], "bin_ms": res["bin_ms"],
                            "n_bins_total": res["n_bins_total"],
                            "n_bins_train": res["n_bins_train"],
                            "n_bins_test": res["n_bins_test"]})
    _write("decode_metrics.json", {**meta, "bin_ms": res["bin_ms"],
                                   "decoders": res["decoders"]})
    _write("task_success.json", {**meta, **res["task_success"]})
    _write("latency.json", {**meta, "bin_ms_default": BIN_MS, "sweep": latency_rows,
                            "replay_matches_batch": replay_max_diff < 1e-9,
                            "replay_max_abs_diff": replay_max_diff})

    kalman_m = res["decoders"]["kalman"]["metrics"]
    ridge_m = res["decoders"]["ridge_wiener"]["metrics"]
    summary = {
        **meta,
        "dataset": res["dataset"],
        "bin_ms": res["bin_ms"],
        "kalman_cc_mean": kalman_m["cc_mean"],
        "kalman_r2_mean": kalman_m["r2_mean"],
        "ridge_cc_mean": ridge_m["cc_mean"],
        "chance_cc_mean_kalman": res["decoders"]["kalman"]["chance"]["cc_mean"],
        "task_success_rate_kalman": res["task_success"]["kalman"]["success_rate"],
        "bitrate_bits_per_s": res["task_success"]["bitrate_bits_per_s_kalman"],
        "latency_p50_ms": default_lat["p50_ms"],
        "latency_p95_ms": default_lat["p95_ms"],
        "realtime_p95": default_lat["realtime_p95"],
        "throughput_units_x_hz": default_lat["throughput_units_x_hz"],
        "replay_matches_batch": replay_max_diff < 1e-9,
    }
    _write("summary.json", summary)
    _plot(res, session, kind)

    print(f"\n[NeuroCursor] {kind}: Kalman CC {kalman_m['cc_mean']:.3f} "
          f"(chance {summary['chance_cc_mean_kalman']:.3f}), "
          f"latency p95 {default_lat['p95_ms']:.3f} ms < {BIN_MS:.0f} ms bin.")


if __name__ == "__main__":
    main()
