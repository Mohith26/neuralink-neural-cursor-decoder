"""Wiener / ridge-regression velocity decoder (the linear baseline).

A ridge-regularized linear map from a short window of neural history taps to 2D
velocity. This is the classic Wiener filter (linear cascade without the output
nonlinearity), the standard baseline the Kalman filter is compared against.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .binning import BinnedData
from .config import RIDGE_ALPHAS, WIENER_TAPS
from .features import make_tap_features


class RidgeVelocityDecoder:
    """Deterministic ridge/Wiener decoder with time-series alpha selection."""

    def __init__(self, taps: int = WIENER_TAPS, alphas=RIDGE_ALPHAS) -> None:
        self.taps = taps
        self.alphas = tuple(alphas)
        self.scaler: StandardScaler | None = None
        self.model: Ridge | None = None
        self.alpha_: float | None = None

    def _features(self, binned: BinnedData) -> np.ndarray:
        return make_tap_features(binned.rates, self.taps)

    def _select_alpha(self, x: np.ndarray, y: np.ndarray) -> float:
        """Pick alpha by forward-chaining time-series CV (no shuffling)."""
        if len(self.alphas) == 1:
            return self.alphas[0]
        splitter = TimeSeriesSplit(n_splits=4)
        best_alpha, best_score = self.alphas[0], -np.inf
        for alpha in self.alphas:
            scores = []
            for tr, va in splitter.split(x):
                sc = StandardScaler().fit(x[tr])
                m = Ridge(alpha=alpha).fit(sc.transform(x[tr]), y[tr])
                scores.append(m.score(sc.transform(x[va]), y[va]))
            mean = float(np.mean(scores))
            if mean > best_score:
                best_score, best_alpha = mean, alpha
        return best_alpha

    def fit(self, train: BinnedData) -> "RidgeVelocityDecoder":
        x = self._features(train)
        y = train.velocity
        self.alpha_ = self._select_alpha(x, y)
        self.scaler = StandardScaler().fit(x)
        self.model = Ridge(alpha=self.alpha_).fit(self.scaler.transform(x), y)
        return self

    def decode(self, test: BinnedData) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit() before decode()")
        x = self.scaler.transform(self._features(test))
        return self.model.predict(x)

    def step_features(self, feat_row: np.ndarray) -> np.ndarray:
        """Decode one already-built (U*taps,) feature row -> velocity (2,)."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit() before step_features()")
        x = self.scaler.transform(feat_row[None, :])
        return self.model.predict(x)[0]
