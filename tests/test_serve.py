"""Serve + replay + latency tests."""

import numpy as np

from neurocursor.kalman import KalmanVelocityDecoder
from neurocursor.latency import measure_step_latency
from neurocursor.replay import build_client, replay


def test_health_and_decode_roundtrip(split):
    dec = KalmanVelocityDecoder().fit(split.train)
    client, _ = build_client(dec, split.test)
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["n_units"] == split.test.n_units
    r = client.post("/decode", json={"counts": split.test.rates[0].tolist()})
    assert r.status_code == 200
    assert len(r.json()["velocity"]) == 2
    assert r.json()["data_kind"] in ("REAL", "SYNTHETIC")


def test_decode_wrong_unit_count_rejected(split):
    dec = KalmanVelocityDecoder().fit(split.train)
    client, _ = build_client(dec, split.test)
    r = client.post("/decode", json={"counts": [0.0, 1.0]})
    assert r.status_code == 422


def test_replay_matches_batch_decode(split):
    dec = KalmanVelocityDecoder().fit(split.train)
    batch = dec.decode(split.test)
    streamed = replay(KalmanVelocityDecoder().fit(split.train), split.test)
    # Streaming the same bins through the service reproduces the batch decode.
    assert np.allclose(batch, streamed, atol=1e-9)


def test_latency_under_bin_duration(split):
    dec = KalmanVelocityDecoder().fit(split.train)
    lat = measure_step_latency(dec, split.test)
    assert lat["p95_ms"] < lat["bin_ms"]  # real-time
    assert lat["realtime_p95"] is True
