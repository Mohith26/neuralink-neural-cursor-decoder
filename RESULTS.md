# NeuroCursor — Measured Results

**Data tag: `REAL`** — public non-human-primate (NHP) motor-cortex recording.
**Date measured:** 2026-08-17 · Python 3.12.13 · numpy 2.5.2 · macOS arm64 (Apple Silicon, CPU).

> This is **offline decode on a recorded dataset** — not a live closed loop with a
> subject. Every number below comes from an actual decode run on this machine and is
> committed under `results/*.json`. Anything not measured is written as the literal `___`.

---

## Dataset (REAL) — `results/dataset.json`

**O'Doherty, Cardoso, Makin & Sabes (2017), "Nonhuman Primate Reaching with
Multichannel Sensorimotor Cortex Electrophysiology."** Zenodo record **583331**,
DOI **10.5281/zenodo.583331**. Self-paced continuous **random-target reaching**;
96-channel Utah array in sensorimotor cortex; 2D cursor + target positions at 250 Hz.

| Fact | Value |
|---|---|
| Session file | **`indy_20161005_06.mat`** (md5 `5ea300952642e0fc54245144499db9bb`, 84 MB) |
| Electrode channels | **96** |
| Sorted units used (rows 1–4) | **164** (unsorted per-channel "hash" excluded by default) |
| Session length | **374.0 s** |
| Kinematic sample rate | **250 Hz** (93,501 samples) |
| Total spikes (selected units) | **188,306** |
| Bin size | **64 ms** → **5,843 bins** (train 4,382 / test 1,461) |
| Held-out reaches (test block) | **71** |
| Train/test split | first **75 %** of time trains, last **25 %** tests (no shuffle) |

---

## 1. Decode accuracy — `results/decode_metrics.json`

Per-axis correlation (**CC**) and **R²** between **decoded and true cursor velocity**
on the **held-out later 25 %** of the session (64 ms bins). The **chance control**
re-decodes the same held-out bins after **circularly shifting the firing-rate matrix
by half its length** (destroys the neural↔movement correspondence, preserves each
unit's firing statistics).

| Decoder | CC x | CC y | **CC mean** | R² x | R² y | **R² mean** | RMSE mean (mm/s) |
|---|---|---|---|---|---|---|---|
| **Kalman filter** | 0.592 | 0.688 | **0.640** | 0.345 | 0.463 | **0.404** | 58.31 |
| Wiener / ridge (4 taps) | 0.607 | 0.778 | **0.692** | 0.364 | 0.601 | **0.483** | 54.02 |
| Small MLP (64-unit) | 0.615 | 0.795 | **0.705** | 0.263 | 0.591 | 0.427 | 56.63 |
| **Chance (shuffle control)** | −0.05 | −0.04 | **−0.047** | −0.51 | −0.56 | −0.53 | 93.63 |

- **Every decoder beats the shuffle control decisively** — CC ≈ 0.64–0.71 vs chance
  **≈ −0.05** (a ~0.69–0.75 absolute CC gap); the shuffle R² is negative (worse than
  predicting the mean), exactly as a proper chance floor should be.
- **Honest note on Kalman vs baseline:** on this **open-loop** session the linear
  **Wiener/ridge baseline modestly outperforms the single-bin Kalman** (CC 0.692 vs
  0.640). This is expected and reported faithfully: the Wiener filter sees a 4-bin
  (~256 ms) neural-history window, while the standard velocity Kalman conditions on
  the current bin plus its own state dynamics. The Kalman's advantage in the BMI
  literature is chiefly in **closed-loop** control (smoother, correctable trajectories),
  which this offline benchmark does not exercise. The numbers were **not tuned** to
  make the Kalman win. A physiologically-motivated 64 ms neural lag nudges the Kalman
  to CC 0.656 but does not change the ranking, so no lag knob is applied by default.

## 2. Task-level metrics — `results/task_success.json`

**Open-loop task-success proxy** (offline, NOT a live closed loop): each held-out
reach starts a virtual cursor at the true reach-start position; the **decoded**
velocity is integrated forward over the reach window; the reach **succeeds** if the
virtual cursor's closest approach to the target is within the acceptance radius.

| Metric | Value |
|---|---|
| Acceptance radius | **16.09 mm** (= 25 % of the median reach distance) |
| Held-out reaches | **71** |
| **Kalman task-success** | **24 / 71 = 33.8 %** |
| True-velocity upper bound (sanity) | **71 / 71 = 100 %** |
| **Shannon/Fitts bitrate (Kalman)** | **0.483 bits/s** over the 93.5 s held-out block |

The **true-velocity upper bound of 100 %** validates the proxy machinery: perfect
velocity acquires every reach. Bitrate = Σ log₂(distance/radius + 1) over successful
reaches ÷ held-out time — an **offline Fitts-style estimate**, not a live-BCI bitrate.

## 3. Real-time latency (Kalman decode compute) — `results/latency.json`

Per-bin **predict+update** wall-clock time on the held-out block (the hot path a C++
port would replace), across bin sizes. Real-time = **p95 < bin duration**.

| Bin size | p50 (ms) | **p95 (ms)** | p99 (ms) | Real-time? | Margin | Throughput (units×Hz) |
|---|---|---|---|---|---|---|
| 32 ms | 0.179 | **0.265** | 0.305 | ✅ | 121× | 5,125 |
| 50 ms | 0.171 | **0.185** | 0.197 | ✅ | 270× | 3,280 |
| **64 ms (default)** | **0.176** | **0.186** | 0.199 | ✅ | **344×** | **2,562.5** |
| 100 ms | 0.182 | **0.193** | 0.207 | ✅ | 519× | 1,640 |

At the 64 ms default the decoder runs **~340× faster than real-time** (p95 ≈ 0.19 ms ≪
64 ms), decoding **164 units** every bin. Wall-clock latency varies ~0.1 ms run-to-run
(single-thread Python/numpy); the robust conclusion — **sub-millisecond, >100× margin
at every bin size** — does not. The FastAPI streaming `/decode` replay reproduces the
batch decode **exactly** (`replay_max_abs_diff = 0.0`).

---

## How to reproduce (exact commands)

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

# 3. run the full benchmark on the REAL session -> results/*.json + figure
python -m neurocursor.run_all   # REAL O'Doherty session (default)

# (documented fallback, tagged SYNTHETIC — only if the download is unavailable)
python -m neurocursor.run_all --synthetic
```

Serve the streaming decoder and replay held-out bins through it:

```bash
python -m neurocursor.run_all          # also verifies replay == batch decode
# programmatic: neurocursor.replay.replay(decoder, test_block)
```

---

## Honest limitations / notes

- **REAL data, OFFLINE decode.** Recorded NHP session; not a live closed loop, no
  subject, no implant/hardware, no clinical claim.
- **Single session, single subject** (monkey "indy", 2016-10-05). Numbers are for this
  session; other O'Doherty sessions/subjects will differ. Loader handles any session file.
- **Sorted units (rows 1–4) by default** (164 units). Including the unsorted hash
  (`include_hash=True`) changes unit count and results.
- **Wiener/ridge > Kalman here** (open-loop) — reported plainly, not hidden or tuned away.
- **Task success + bitrate are OFFLINE proxies** (open-loop integration + Fitts-style
  estimate), not live closed-loop control metrics; acceptance radius is a documented
  fraction of median reach distance.
- **Latency is single-thread Python/numpy** on Apple Silicon; the C++ hot path is
  Should-have (v2), not built. Latency times decode compute only, excluding HTTP.
- **Q diagonal-loading** (1e-3 × mean variance) keeps the 164-unit innovation
  covariance invertible; a raw covariance is singular (near-silent units).
