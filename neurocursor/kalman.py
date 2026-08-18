"""Kalman-filter velocity decoder (Wu et al. 2006), the standard BMI decoder.

Latent state x_t = [px, py, vx, vy] evolves linearly; binned firing rates are
linear-Gaussian observations of the state. Parameters (A, W, C, b, Q) are fit by
closed-form maximum likelihood on the training bins; decoding is the standard
predict/update recursion. Velocity is read off the filtered state (x[2:4]).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binning import BinnedData

_STATE_DIM = 4  # [px, py, vx, vy]


def _states(binned: BinnedData) -> np.ndarray:
    """(B, 4) latent-state observations [px, py, vx, vy] from kinematics."""
    return np.hstack([binned.position, binned.velocity])


@dataclass(frozen=True)
class KalmanParams:
    A: np.ndarray  # (4, 4) state transition
    W: np.ndarray  # (4, 4) state (process) noise cov
    C: np.ndarray  # (U, 4) observation (tuning) matrix
    b: np.ndarray  # (U,) observation baseline (mean firing)
    Q: np.ndarray  # (U, U) observation noise cov


# With many units (~O(100)), the observation-noise covariance Q can be
# ill-conditioned (near-silent or collinear units give near-zero residual
# variance). A small diagonal loading keeps the innovation covariance S
# invertible without materially changing well-estimated units.
_Q_DIAG_LOADING = 1e-3


class KalmanVelocityDecoder:
    """Deterministic Kalman velocity decoder. No RNG — fully reproducible."""

    def __init__(self, q_diag_loading: float = _Q_DIAG_LOADING) -> None:
        self.q_diag_loading = q_diag_loading
        self.params: KalmanParams | None = None
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None

    def fit(self, train: BinnedData) -> "KalmanVelocityDecoder":
        x = _states(train)  # (B, 4)
        z = train.rates  # (B, U)

        x1, x2 = x[:-1], x[1:]  # consecutive states
        # A: x2 ~ x1 @ A_rows  ->  transition A s.t. x_t = A @ x_{t-1}.
        A_rows = np.linalg.lstsq(x1, x2, rcond=None)[0]  # (4, 4), maps x1->x2 row-wise
        A = A_rows.T
        resid_state = x2 - x1 @ A_rows
        W = (resid_state.T @ resid_state) / (x1.shape[0])

        # Observation model with intercept: z ~ x @ H + b.
        ones = np.ones((x.shape[0], 1))
        xb = np.hstack([x, ones])
        Hb = np.linalg.lstsq(xb, z, rcond=None)[0]  # (5, U)
        H = Hb[:_STATE_DIM]  # (4, U)
        b = Hb[_STATE_DIM]  # (U,)
        C = H.T  # (U, 4)
        resid_obs = z - xb @ Hb
        Q = (resid_obs.T @ resid_obs) / z.shape[0]
        # Diagonal loading for numerical stability with many units.
        mean_var = float(np.mean(np.diag(Q)))
        Q = Q + self.q_diag_loading * mean_var * np.eye(Q.shape[0])

        self.params = KalmanParams(A=A, W=W, C=C, b=b, Q=Q)
        return self

    def reset(self, init_position: np.ndarray) -> None:
        """Start a fresh filter run at a known cursor position, zero velocity."""
        if self.params is None:
            raise RuntimeError("fit() before reset()")
        self._x = np.array([init_position[0], init_position[1], 0.0, 0.0], dtype=np.float64)
        self._P = np.eye(_STATE_DIM)

    def step(self, z: np.ndarray) -> np.ndarray:
        """One predict/update step on a bin of firing counts. Returns velocity (2,)."""
        if self.params is None or self._x is None:
            raise RuntimeError("call reset() before step()")
        p = self.params
        # Predict
        x_pred = p.A @ self._x
        P_pred = p.A @ self._P @ p.A.T + p.W
        # Update
        innovation = z - p.b - p.C @ x_pred
        S = p.C @ P_pred @ p.C.T + p.Q
        K = P_pred @ p.C.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        self._x = x_pred + K @ innovation
        self._P = (np.eye(_STATE_DIM) - K @ p.C) @ P_pred
        return self._x[2:4].copy()

    def decode(self, test: BinnedData) -> np.ndarray:
        """Decode a full test block; returns (B, 2) decoded velocity."""
        self.reset(test.position[0])
        out = np.empty((test.n_bins, 2), dtype=np.float64)
        for i in range(test.n_bins):
            out[i] = self.step(test.rates[i])
        return out
