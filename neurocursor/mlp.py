"""Optional small MLP velocity decoder (nonlinear baseline).

A compact scikit-learn MLPRegressor on the same neural history taps. Seeded for
determinism. Kept intentionally small (CPU, no GPU) per the locked stack.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .binning import BinnedData
from .config import SEED, WIENER_TAPS
from .features import make_tap_features


class MLPVelocityDecoder:
    """Deterministic (seeded) small MLP decoder."""

    def __init__(self, taps: int = WIENER_TAPS, hidden=(64,), seed: int = SEED) -> None:
        self.taps = taps
        self.hidden = tuple(hidden)
        self.seed = seed
        self.scaler: StandardScaler | None = None
        self.model: MLPRegressor | None = None

    def fit(self, train: BinnedData) -> "MLPVelocityDecoder":
        x = make_tap_features(train.rates, self.taps)
        self.scaler = StandardScaler().fit(x)
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=300,
            random_state=self.seed,
        )
        self.model.fit(self.scaler.transform(x), train.velocity)
        return self

    def decode(self, test: BinnedData) -> np.ndarray:
        if self.model is None or self.scaler is None:
            raise RuntimeError("fit() before decode()")
        x = self.scaler.transform(make_tap_features(test.rates, self.taps))
        return self.model.predict(x)
