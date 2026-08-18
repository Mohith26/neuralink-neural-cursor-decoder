"""Field-standard decode-accuracy metrics: CC, R2, RMSE, and Shannon bitrate."""

from __future__ import annotations

import numpy as np


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient (CC) between two 1D signals."""
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot (1D)."""
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error (1D)."""
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def per_axis_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """CC, R2, RMSE for a (B, 2) velocity signal, per axis and mean."""
    axes = ("x", "y")
    out: dict = {}
    for i, ax in enumerate(axes):
        out[f"cc_{ax}"] = correlation(y_true[:, i], y_pred[:, i])
        out[f"r2_{ax}"] = r2_score(y_true[:, i], y_pred[:, i])
        out[f"rmse_{ax}"] = rmse(y_true[:, i], y_pred[:, i])
    out["cc_mean"] = 0.5 * (out["cc_x"] + out["cc_y"])
    out["r2_mean"] = 0.5 * (out["r2_x"] + out["r2_y"])
    out["rmse_mean"] = 0.5 * (out["rmse_x"] + out["rmse_y"])
    return out


def shannon_bitrate(
    n_success: int, distances: np.ndarray, radius: float, total_time_s: float
) -> float:
    """Fitts/Shannon information throughput (bits/s) from successful reaches.

    Index of difficulty per reach = log2(distance / radius + 1). Throughput is
    the summed ID over successful reaches divided by the total task time. This is
    an OFFLINE estimate derived from the open-loop task-success proxy, not a live
    closed-loop bitrate.
    """
    if total_time_s <= 0 or n_success <= 0:
        return 0.0
    dist = np.asarray(distances, dtype=np.float64)
    ids = np.log2(dist / radius + 1.0)
    # Attribute the mean index of difficulty to each success (order-independent).
    bits = float(ids.mean()) * n_success
    return bits / total_time_s
