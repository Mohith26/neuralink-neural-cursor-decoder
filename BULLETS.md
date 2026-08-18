# Résumé Bullets — NeuroCursor (filled strictly from measured results)

> **Data tag: `REAL`.** Measured 2026-08-17 on the public NHP dataset O'Doherty et al.
> 2017 (Zenodo 583331), session `indy_20161005_06.mat`, 64 ms bins, leakage-safe
> temporal split. Every number traces to `results/*.json`. Unmeasured values are the
> literal `___`. This is **offline decode on a recorded dataset**, not a live closed loop.

## Filled bullets

- Built a **real-time motor-cortex neural cursor decoder** (**Kalman filter, Python**)
  turning **96-channel / 164-sorted-unit** spike activity into **2D cursor velocity** on
  a **public NHP BMI reaching dataset** (O'Doherty et al. 2017), hitting **CC 0.640 /
  R² 0.404** decoded-vs-true velocity — **vs a shuffle-control CC of −0.047** — the
  Neuralink "control a cursor with your mind" **decode path**, run **offline** on
  recorded **REAL** data.
  <br>_(MEASURED, REAL: CC mean 0.640 [x 0.592, y 0.688], R² 0.404, RMSE 58.3 mm/s;
  shuffle-control CC −0.047 → beaten decisively.)_

- Benchmarked the Kalman against a **Wiener/ridge baseline** and a small **MLP** — the
  **ridge/Wiener baseline actually edged the single-bin Kalman (CC 0.692 vs 0.640)**,
  reported honestly, not tuned away — and reached a **33.8 % task-success proxy** /
  **0.483 bits/s** offline Fitts throughput, all evaluated **leakage-safe on held-out
  later trials** against a **shuffle/chance control** (true-velocity upper bound = 100 %).
  <br>_(MEASURED, REAL: Wiener CC 0.692, MLP CC 0.705, Kalman CC 0.640 — baseline > Kalman
  open-loop; task success 24/71 = 33.8 %; bitrate 0.483 bits/s over 93.5 s held-out.
  Honesty: task success + bitrate are OFFLINE open-loop proxies, not live-BCI metrics.)_

- Decoded each spike bin in **p95 ~0.19 ms** (**≪ the 64 ms bin → real-time, ~340×
  margin**) over **164 units × 15.6 Hz (2,562 units·Hz; 96 electrode channels)**,
  verified by **34 passing tests** (binning, no-leak temporal split, decoder shapes,
  determinism, eval math, chance-beating, serve/replay, latency, real O'Doherty loader),
  **REAL-data-tagged and reproducible** (md5-verified download + seeded pipeline).
  <br>_(MEASURED, REAL: latency p50 0.176 ms / p95 0.186 ms vs 64 ms bin, ~0.1 ms
  run-to-run variance; streaming /decode replay reproduces the batch decode exactly
  [max abs diff 0.0]; 34 tests pass.)_

## Measured-value ledger

| Placeholder | Value | Status |
|---|---|---|
| electrode channels | 96 | MEASURED (REAL) |
| sorted units decoded | 164 | MEASURED (REAL) |
| Kalman CC mean (x, y) | 0.640 (0.592, 0.688) | MEASURED (REAL) |
| Kalman R² mean | 0.404 | MEASURED (REAL) |
| Kalman RMSE mean | 58.3 mm/s | MEASURED (REAL) |
| shuffle-control CC | −0.047 | MEASURED (REAL) |
| Wiener/ridge CC mean | 0.692 (> Kalman) | MEASURED (REAL) |
| MLP CC mean | 0.705 | MEASURED (REAL) |
| task-success proxy (Kalman) | 33.8 % (24/71) | MEASURED (REAL, offline proxy) |
| true-velocity upper bound | 100 % (71/71) | MEASURED (REAL, sanity) |
| bitrate (Kalman) | 0.483 bits/s | MEASURED (REAL, offline Fitts) |
| decode latency p50 / p95 | 0.176 / 0.186 ms | MEASURED (REAL) |
| bin duration | 64 ms | CONFIG |
| throughput | 164 units × 15.6 Hz = 2,562 units·Hz | MEASURED (REAL) |
| tests passing | 34 | MEASURED |
| C++ hot-path latency | `___` | NOT BUILT (Should-have v2) |

## Honesty tags

- ✅ **REAL** public NHP data (O'Doherty 2017, `indy_20161005_06.mat`, md5-verified).
- ✅ **OFFLINE** decode of recorded data — not a live closed loop, no subject/implant/clinical claim.
- ⚠️ **Wiener/ridge baseline > Kalman** on this open-loop session (CC 0.692 vs 0.640) — stated plainly, not tuned.
- ⚠️ **Task success + bitrate are OFFLINE proxies** (open-loop integration + Fitts estimate), not live-BCI throughput.
- ⚠️ **Single session / single subject** (monkey "indy", 2016-10-05); sorted units only (hash excluded).
- ❌ **C++ real-time hot path NOT built** (Should-have v2); latency is single-thread Python/numpy.
