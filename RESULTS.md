# NeuroCursor: measured results

Measured 2026-08-17 with Python 3.12.13, numpy 2.5.2, on macOS arm64 (Apple Silicon, CPU). All numbers below are from actual decode runs on a real recorded non-human-primate session (details next), and everything is committed under `results/*.json`. This is offline decode on a recorded dataset, not a live closed loop with a subject.

## Dataset (`results/dataset.json`)

O'Doherty, Cardoso, Makin & Sabes (2017), "Nonhuman Primate Reaching with Multichannel Sensorimotor Cortex Electrophysiology." Zenodo record 583331, DOI 10.5281/zenodo.583331. Self-paced continuous random-target reaching; 96-channel Utah array in sensorimotor cortex; 2D cursor + target positions at 250 Hz.

| Fact | Value |
|---|---|
| Session file | `indy_20161005_06.mat` (md5 `5ea300952642e0fc54245144499db9bb`, 84 MB) |
| Electrode channels | 96 |
| Sorted units used (rows 1-4) | 164 (unsorted per-channel "hash" excluded by default) |
| Session length | 374.0 s |
| Kinematic sample rate | 250 Hz (93,501 samples) |
| Total spikes (selected units) | 188,306 |
| Bin size | 64 ms -> 5,843 bins (train 4,382 / test 1,461) |
| Held-out reaches (test block) | 71 |
| Train/test split | first 75% of time trains, last 25% tests (no shuffle) |

## Decode accuracy (`results/decode_metrics.json`)

Per-axis correlation (CC) and R² between decoded and true cursor velocity on the held-out later 25% of the session (64 ms bins). The chance control re-decodes the same held-out bins after circularly shifting the firing-rate matrix by half its length, which destroys the neural-to-movement correspondence while preserving each unit's firing statistics.

| Decoder | CC x | CC y | CC mean | R² x | R² y | R² mean | RMSE mean (mm/s) |
|---|---|---|---|---|---|---|---|
| Kalman filter | 0.592 | 0.688 | 0.640 | 0.345 | 0.463 | 0.404 | 58.31 |
| Wiener / ridge (4 taps) | 0.607 | 0.778 | 0.692 | 0.364 | 0.601 | 0.483 | 54.02 |
| Small MLP (64-unit) | 0.615 | 0.795 | 0.705 | 0.263 | 0.591 | 0.427 | 56.63 |
| Chance (shuffle control) | -0.05 | -0.04 | -0.047 | -0.51 | -0.56 | -0.53 | 93.63 |

Every decoder beats the shuffle control decisively: CC around 0.64 to 0.71 vs chance around -0.05 (a 0.69 to 0.75 absolute CC gap), and the shuffle R² is negative (worse than predicting the mean), exactly as a proper chance floor should be.

On Kalman vs the baseline: on this open-loop session the linear Wiener/ridge baseline modestly outperforms the single-bin Kalman (CC 0.692 vs 0.640). That's expected and reported faithfully. The Wiener filter sees a 4-bin (~256 ms) neural-history window, while the standard velocity Kalman conditions on the current bin plus its own state dynamics; the Kalman's advantage in the BMI literature is chiefly in closed-loop control (smoother, correctable trajectories), which this offline benchmark does not exercise. I did not tune the numbers to make the Kalman win. A physiologically-motivated 64 ms neural lag nudges the Kalman to CC 0.656 but does not change the ranking, so no lag knob is applied by default.

## Task-level metrics (`results/task_success.json`)

This is an open-loop task-success proxy, offline, not a live closed loop: each held-out reach starts a virtual cursor at the true reach-start position, the decoded velocity is integrated forward over the reach window, and the reach succeeds if the virtual cursor's closest approach to the target is within the acceptance radius.

| Metric | Value |
|---|---|
| Acceptance radius | 16.09 mm (= 25% of the median reach distance) |
| Held-out reaches | 71 |
| Kalman task-success | 24 / 71 = 33.8% |
| True-velocity upper bound (sanity) | 71 / 71 = 100% |
| Shannon/Fitts bitrate (Kalman) | 0.483 bits/s over the 93.5 s held-out block |

The true-velocity upper bound of 100% validates the proxy machinery: perfect velocity acquires every reach. Bitrate = sum of log2(distance/radius + 1) over successful reaches divided by held-out time, an offline Fitts-style estimate, not a live-BCI bitrate.

## Real-time latency (`results/latency.json`)

Per-bin predict+update wall-clock time on the held-out block (the hot path a C++ port would replace), across bin sizes. Real-time here means p95 < bin duration.

| Bin size | p50 (ms) | p95 (ms) | p99 (ms) | Real-time? | Margin | Throughput (units x Hz) |
|---|---|---|---|---|---|---|
| 32 ms | 0.179 | 0.265 | 0.305 | yes | 121x | 5,125 |
| 50 ms | 0.171 | 0.185 | 0.197 | yes | 270x | 3,280 |
| 64 ms (default) | 0.176 | 0.186 | 0.199 | yes | 344x | 2,562.5 |
| 100 ms | 0.182 | 0.193 | 0.207 | yes | 519x | 1,640 |

At the 64 ms default the decoder runs about 340x faster than real-time (p95 around 0.19 ms against a 64 ms bin), decoding 164 units every bin. Wall-clock latency varies by about 0.1 ms run-to-run (single-thread Python/numpy); the robust conclusion, sub-millisecond with over 100x margin at every bin size, does not. The FastAPI streaming `/decode` replay reproduces the batch decode exactly (`replay_max_abs_diff = 0.0`).

## How to reproduce

```bash
# 0. environment (Python 3.12, CPU, no API keys)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# 1. download ONE real O'Doherty-2017 session (~84 MB, md5-verified) into data/
bash scripts/download_data.sh
#    -> data/indy_20161005_06.mat  (NOT committed to git)

# 2. run the test suite (34 tests: binning, no-leak split, decoder shapes,
#    determinism, eval math, chance-beating, serve/replay, latency, real loader)
pytest -q                       # -> 34 passed

# 3. run the full benchmark on the real session -> results/*.json + figure
python -m neurocursor.run_all   # real O'Doherty session (default)

# (fallback, tagged SYNTHETIC, only if the download is unavailable)
python -m neurocursor.run_all --synthetic
```

Serve the streaming decoder and replay held-out bins through it:

```bash
python -m neurocursor.run_all          # also verifies replay == batch decode
# programmatic: neurocursor.replay.replay(decoder, test_block)
```

## Notes and limitations

- Real recorded data, offline decode. No live closed loop, no subject, no implant/hardware, no clinical claim.
- Single session, single subject (monkey "indy", 2016-10-05). Numbers are for this session; other sessions/subjects from the dataset will differ. The loader handles any session file.
- Sorted units (rows 1-4) by default, 164 units. Including the unsorted hash (`include_hash=True`) changes unit count and results.
- Wiener/ridge beats the Kalman here (open-loop); reported plainly, not hidden or tuned away.
- Task success and bitrate are offline proxies (open-loop integration plus a Fitts-style estimate), not live closed-loop control metrics; the acceptance radius is a documented fraction of median reach distance.
- Latency is single-thread Python/numpy on Apple Silicon and times decode compute only, excluding HTTP. A C++ hot path is not built.
- Q diagonal-loading (1e-3 x mean variance) keeps the 164-unit innovation covariance invertible; a raw covariance is singular (near-silent units).
