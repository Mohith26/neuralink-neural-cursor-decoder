# NeuroCursor

Decoding 2D cursor velocity from motor-cortex spikes, the core decode problem behind brain-computer interfaces that let someone control a computer cursor with neural activity.

I built this around a public non-human-primate (NHP) reaching dataset: O'Doherty et al. 2017 (Zenodo record 583331, DOI 10.5281/zenodo.583331), session `indy_20161005_06.mat`. Multi-channel spike activity gets binned into firing-rate features and decoded into continuous 2D cursor velocity/position by a Kalman filter, compared against Wiener/ridge and small-MLP baselines, and evaluated with the field-standard metrics: per-axis decoded-vs-true velocity correlation and R², an open-loop task-success proxy, a Fitts/Shannon bitrate, and per-bin decode latency. A FastAPI service streams the stateful Kalman decoder bin by bin, and a replay harness confirms the streamed decode matches the batch decode exactly.

To be clear about what this is: offline decode on a recorded dataset. No implant, no hardware, no live closed loop, no clinical claim.

## Results at a glance (measured 2026-08-17, real recorded session)

| Metric | Value |
|---|---|
| Data | O'Doherty-2017 `indy_20161005_06.mat`, 96 ch / 164 sorted units / 374 s |
| Kalman decode CC (mean x,y) | 0.640 (x 0.592, y 0.688) vs true velocity |
| Kalman R² (mean) | 0.404 · RMSE 58.3 mm/s |
| Wiener/ridge baseline CC | 0.692 (edges out the single-bin Kalman, open-loop) |
| Chance (shuffle) control CC | -0.047, so every decoder beats it decisively |
| Task-success proxy (Kalman) | 33.8% (24/71 reaches; true-velocity upper bound = 100%) |
| Bitrate (Kalman, offline Fitts) | 0.483 bits/s |
| Decode latency p95 | ~0.19 ms vs a 64 ms bin, real-time with ~340x margin |
| Throughput | 164 units x 15.6 Hz = 2,562 units·Hz |
| Tests | 34 passed (incl. the real data loader) |

Full numbers and methodology in [RESULTS.md](RESULTS.md); raw JSON in `results/`.

## Pipeline

```
 96-ch motor cortex          bin (64 ms)            decode 2D velocity          evaluate
 spike times + cursor  ─▶  firing-rate features ─▶  Kalman filter        ─▶  CC / R² / RMSE
 (O'Doherty 2017 .mat)      + velocity targets      Wiener/ridge, MLP        task success
        │                    temporal split         integrate → position     bitrate
        │                    (no time leakage)              │                 shuffle control
        └─ synthetic fallback (tagged SYNTHETIC)            └─▶ /decode stream + replay + latency
```

The dataset itself is not committed (see `.gitignore`); `scripts/download_data.sh` fetches and md5-verifies one session on demand. A seeded synthetic generator (`neurocursor/synthetic.py`) is the fallback when the download isn't available; anything it produces is tagged `SYNTHETIC` so it can't be confused with the real-session results. The committed `results/*` are from the real session.

## The decoders

- **Kalman filter (`kalman.py`)**: latent state `[px, py, vx, vy]`, linear-Gaussian observation of binned firing rates; parameters fit in closed form by ML on the training bins; standard predict/update recursion; velocity read off the filtered state. This is the stateful decoder the `/decode` service streams in real time.
- **Wiener / ridge (`ridge.py`)**: ridge-regularized linear map from a short window of neural history taps to velocity (the classic Wiener filter); alpha chosen by forward-chaining `TimeSeriesSplit`, no shuffling.
- **MLP (`mlp.py`)**: compact seeded `MLPRegressor` on the same taps, as an optional nonlinear baseline.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

bash scripts/download_data.sh      # ~84 MB O'Doherty session -> data/ (md5-verified)
pytest -q                          # 34 passed
python -m neurocursor.run_all      # real-session run -> results/*.json + figure
```

No dataset? `python -m neurocursor.run_all --synthetic` runs the seeded fallback (all outputs tagged SYNTHETIC).

## Streaming decode API

| Endpoint | Purpose |
|---|---|
| `POST /decode` | `{counts:[...units]}` -> `{velocity, position, data_kind}` (stateful) |
| `POST /reset` | reset the filter to a known cursor position |
| `GET /health` | decoder status, unit count, bin size, `data_kind` (REAL/SYNTHETIC) |

```bash
uvicorn neurocursor.app:app   # (after initializing STATE with a fitted decoder)
```

The replay harness (`neurocursor.replay`) streams every held-out bin through the service in order and confirms it reproduces the batch decode bit-for-bit.

## Layout

```
neurocursor/
  config.py           constants (bin size, seed, split, dataset URL/md5), no magic numbers
  session.py          immutable SessionData (frozen dataclass) + REAL/SYNTHETIC tag
  loader_odoherty.py  load O'Doherty-2017 .mat (HDF5 v7.3) via h5py; sorted-unit selection
  synthetic.py        seeded cosine-tuned Poisson generator (SYNTHETIC fallback)
  binning.py          spikes -> firing-rate feature matrix aligned to velocity/position
  split.py            leakage-safe temporal train/test split (earlier trains, later tests)
  features.py         causal neural history taps for the Wiener filter
  kalman.py           Kalman velocity decoder (Wu 2006), the streaming real-time decoder
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
results/*.json             committed measured numbers (real session, 2026-08-17)
```

Stack: Python 3.12, numpy/scipy, scikit-learn (ridge, MLP), h5py/pymatreader for .mat, matplotlib, FastAPI + uvicorn, pytest. Free, local, CPU-only, no API keys.

## Limitations

- Offline decode on a recorded session, not a live closed loop. Task success and bitrate are open-loop proxies (integrating decoded velocity over reach windows, plus a Fitts-style estimate), not live-BCI control metrics.
- Single session, single subject (monkey "indy", 2016-10-05). The loader handles any session file from the dataset, but the committed numbers are for this one.
- Sorted units (rows 1 to 4) by default, 164 units; including the unsorted hash changes the unit count and the results.
- On this open-loop benchmark, the Wiener/ridge baseline modestly outperforms the single-bin Kalman. That's reported as-is; the Kalman's advantage in the literature shows up mainly in closed-loop control, which this setup can't exercise.
- Latency is single-thread Python/numpy on Apple Silicon and times decode compute only, excluding HTTP. A C++ hot path, live cursor animation, online refitting, and multi-session pooling are all not built.
