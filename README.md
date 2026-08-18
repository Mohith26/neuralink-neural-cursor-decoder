# NeuroCursor — Real-Time Motor-Cortex Neural Cursor Decoder

Turn multi-channel **motor-cortex spike activity** into a continuously decoded
**2D cursor velocity/position** on a **public non-human-primate (NHP) BMI reaching
dataset** — a **Kalman-filter** decoder (vs **Wiener/ridge** and a small **MLP**
baseline), benchmarked with the field-standard metrics: per-axis decoded-vs-true
velocity **correlation / R²**, a closed-loop-style **task-success** proxy, a
**Fitts/Shannon bitrate**, and per-bin **decode latency**. This mirrors Neuralink's
flagship "control a computer with your mind" **decode path**.

> ### Data & honesty (read me)
> - **Data tag: `REAL`.** Public NHP recording — **O'Doherty et al. 2017**
>   (Zenodo `583331`, DOI 10.5281/zenodo.583331), session `indy_20161005_06.mat`.
> - This is **OFFLINE decode on a recorded dataset** — **not** a live closed loop
>   with a subject, no implant/hardware, no clinical claim.
> - The dataset is **not committed** (see `.gitignore`); `scripts/download_data.sh`
>   fetches + md5-verifies one session on demand.
> - A **seeded synthetic** generator (`neurocursor/synthetic.py`) is the documented
>   fallback; results from it are tagged **`SYNTHETIC`**. The committed `results/*`
>   are from the **REAL** session.
> - Every number: **[RESULTS.md](RESULTS.md)** · raw JSON in `results/` · bullets in
>   **[BULLETS.md](BULLETS.md)**.

## Headline (REAL, measured 2026-08-17)

| Metric | Value |
|---|---|
| Data | **REAL** — O'Doherty-2017 `indy_20161005_06.mat`, **96 ch / 164 sorted units / 374 s** |
| Kalman decode CC (mean x,y) | **0.640** (x 0.592, y 0.688) vs true velocity |
| Kalman R² (mean) | **0.404** · RMSE 58.3 mm/s |
| Wiener/ridge baseline CC | **0.692** (edges out the single-bin Kalman, open-loop) |
| **Chance (shuffle) control CC** | **−0.047** — every decoder beats it decisively |
| Task-success proxy (Kalman) | **33.8 %** (24/71 reaches; true-velocity UB = 100 %) |
| Bitrate (Kalman, offline Fitts) | **0.483 bits/s** |
| **Decode latency p95** | **~0.19 ms** ≪ 64 ms bin → **real-time (~340× margin)** |
| Throughput | **164 units × 15.6 Hz = 2,562 units·Hz** |
| Tests | **34 passed** (incl. the real O'Doherty loader) |

## Pipeline

```
 96-ch motor cortex          bin (64 ms)            decode 2D velocity          evaluate
 spike times + cursor  ─▶  firing-rate features ─▶  Kalman filter        ─▶  CC / R² / RMSE
 (O'Doherty 2017 .mat)      + velocity targets      Wiener/ridge, MLP        task success
        │                    temporal split         integrate → position     bitrate
        │                    (no time leakage)              │                 shuffle control
        └─ synthetic fallback (tagged SYNTHETIC)            └─▶ /decode stream + replay + latency
```

## Layout

```
neurocursor/
  config.py           constants (bin size, seed, split, dataset URL/md5) — no magic numbers
  session.py          immutable SessionData (frozen dataclass) + REAL/SYNTHETIC tag
  loader_odoherty.py  load O'Doherty-2017 .mat (HDF5 v7.3) via h5py; sorted-unit selection
  synthetic.py        seeded cosine-tuned Poisson generator (documented SYNTHETIC fallback)
  binning.py          spikes -> firing-rate feature matrix aligned to velocity/position
  split.py            leakage-safe temporal train/test split (earlier trains, later tests)
  features.py         causal neural history taps for the Wiener filter
  kalman.py           Kalman velocity decoder (Wu 2006) — the streaming real-time decoder
  ridge.py            Wiener / ridge baseline (TimeSeriesSplit alpha selection)
  mlp.py              optional small MLP baseline (seeded)
  integrate.py        velocity -> position integration
  metrics.py          CC, R², RMSE, Shannon/Fitts bitrate (unit-tested on toy inputs)
  task.py             reach segmentation + open-loop task-success proxy
  shuffle.py          circular-shift chance/shuffle control
  pipeline.py         load -> bin -> split -> fit -> evaluate (deterministic)
  app.py              FastAPI streaming /decode (stateful Kalman) + /reset + /health
  replay.py           replay held-out bins through the service (verifies == batch decode)
  latency.py          per-bin decode-latency benchmark (p50/p95 vs bin duration)
  run_all.py          run everything -> results/*.json + velocity_traces.png
scripts/download_data.sh   fetch + md5-verify one O'Doherty session into data/
tests/                     34 tests
results/*.json             committed measured numbers (REAL, 2026-08-17)
RESULTS.md / BULLETS.md
```

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

bash scripts/download_data.sh      # ~84 MB O'Doherty session -> data/ (md5-verified)
pytest -q                          # 34 passed
python -m neurocursor.run_all      # REAL run -> results/*.json + figure
```

No dataset? `python -m neurocursor.run_all --synthetic` runs the seeded fallback
(all outputs tagged **SYNTHETIC**).

## Decoders

- **Kalman filter (`kalman.py`)** — latent state `[px, py, vx, vy]`, linear-Gaussian
  observation of binned firing rates; parameters fit in closed form by ML on the
  training bins; standard predict/update recursion; velocity read off the filtered
  state. This is the **stateful decoder the `/decode` service streams** in real time.
- **Wiener / ridge (`ridge.py`)** — ridge-regularized linear map from a short window
  of neural history taps to velocity (the classic Wiener filter); alpha chosen by
  forward-chaining `TimeSeriesSplit` (no shuffling).
- **MLP (`mlp.py`)** — compact seeded `MLPRegressor` on the same taps (optional).

## Streaming decode API

| Endpoint | Purpose |
|---|---|
| `POST /decode` | `{counts:[…units]}` → `{velocity, position, data_kind}` (stateful) |
| `POST /reset` | reset the filter to a known cursor position |
| `GET /health` | decoder status, unit count, bin size, `data_kind` (REAL/SYNTHETIC) |

```bash
uvicorn neurocursor.app:app   # (after initializing STATE with a fitted decoder)
```

The replay harness (`neurocursor.replay`) streams every held-out bin through the
service in order and confirms it reproduces the batch decode bit-for-bit.

## Tech stack

Python 3.12 · numpy / scipy · scikit-learn (ridge, MLP) · h5py / pymatreader (.mat) ·
matplotlib · FastAPI + uvicorn · pytest. **Free / local / CPU, no API keys.**

## Not built (Should-have v2 / out of scope)

C++ (pybind11) real-time hot path; live matplotlib cursor animation; online
refit/adaptation; multi-session pooling; and — explicitly out of scope — any real
data collection, implant/hardware, live closed loop, GPU deep models, or clinical use.
