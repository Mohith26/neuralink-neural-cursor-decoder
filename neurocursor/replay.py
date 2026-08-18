"""Replay harness: stream held-out bins through the decode service in order.

Uses FastAPI's in-process TestClient so it runs without a network socket. This
verifies the streaming /decode path reproduces the batch decoder and lets the
latency bench time real per-bin request work.
"""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from .app import DecodeState, create_app
from .binning import BinnedData
from .kalman import KalmanVelocityDecoder


def build_client(decoder: KalmanVelocityDecoder, test: BinnedData) -> tuple:
    """Wire a fitted Kalman decoder + test block into an in-process client."""
    state = DecodeState()
    state.init(decoder, test.n_units, test.bin_s, test.kind, test.position[0])
    client = TestClient(create_app(state))
    return client, state


def replay(decoder: KalmanVelocityDecoder, test: BinnedData) -> np.ndarray:
    """Replay every test bin through /decode; return (B, 2) decoded velocity."""
    client, _ = build_client(decoder, test)
    client.post("/reset", json=test.position[0].tolist())
    out = np.empty((test.n_bins, 2), dtype=np.float64)
    for i in range(test.n_bins):
        resp = client.post("/decode", json={"counts": test.rates[i].tolist()})
        resp.raise_for_status()
        out[i] = resp.json()["velocity"]
    return out
