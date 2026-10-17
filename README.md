# Hawkes Market Maker

Hawkes Market Maker is a low-latency market-making research and engineering project that combines microstructure-driven feature engineering, Hawkes-process order flow modeling, and an event-driven C++ quoting engine into one end-to-end pipeline. The project starts from raw market data, builds replayable features, calibrates Hawkes intensities, runs a queue- and toxicity-aware market-making engine, and then evaluates the strategy against a baseline through PnL, markout, regime, and latency analysis. For what changed relative to the original version (several engine bugs fixed, labels corrected, fill model calibrated, multi-seed evaluation), see [`docs/changes.md`](docs/changes.md). The Hawkes calibration diagnostics are in [`docs/hawkes_calibration.md`](docs/hawkes_calibration.md). The original write-up, [`Report.md`](./Report.md), predates these fixes.

## Why this project

Most trading projects stop at either research code or systems code. This project was built to show both.

It combines:

- microstructure-aware feature engineering,
- Hawkes-based modeling of clustered order flow,
- a replayable C++ event-driven engine,
- explicit inventory, queue, fee, and toxicity logic,
- parameter sweeps and regime analysis,
- and benchmarked runtime performance.

## Pipeline

The project is organized as a four-stage pipeline.

### Step 1: Feature engineering

Raw market data is transformed into a structured feature table. This stage produces time-aligned features such as midprice, microprice, volatility proxies, and order flow imbalance so the strategy can reason about market state efficiently during replay.

**Outputs**

- `output/step1_features/feature_table_1s.csv`
- `output/step1_features/feature_table_1s.parquet`
- `output/step1_features/ohlcv_1s.csv`
- `output/step1_features/ohlcv_1s.parquet`

### Step 2: Hawkes calibration

This stage labels trades, estimates order-flow activity, and generates Hawkes-based intensity features. The point of this layer is to capture self-excitation in order arrivals so the quoting engine can respond not only to the current book state but also to recent clustering in flow.

**Outputs**

- `output/step2_hawkes/feature_table_step2_1s.csv`
- `output/step2_hawkes/feature_table_step2_1s.parquet`
- `output/step2_hawkes/hawkes_intensities_1s.parquet`
- `output/step2_hawkes/baseline_intensities_1s.parquet`
- `output/step2_hawkes/event_counts_1s.parquet`
- `output/step2_hawkes/trades_labeled.parquet`
- `output/step2_hawkes/step2_summary.json`

### Step 3: Event-driven C++ engine

The core market-making engine is implemented in C++ and compiled as `bin/step3_mm`. It replays the feature stream event by event, extracts signals, generates quotes, applies toxicity and queue logic, enforces inventory/risk constraints, simulates fills, measures latency, and writes detailed results.

**Core source**

- `engine/event_driven_mm_engine.cpp`
- `bin/step3_mm`

**Outputs**

- `output/step3_engine_cpp/engine_results.csv`
- `output/step3_engine_cpp/step3_summary.json`

### Step 4: Analysis and validation

This stage aggregates run outputs into comparative summaries and plots. It is where Hawkes versus baseline performance is evaluated through final PnL, markout, adverse selection, regime analysis, and latency profiling.

**Outputs**

- `output/step4_analysis/step4_hawkes_vs_baseline_summary.csv`
- `output/step4_analysis/step4_master_summary.csv`
- `output/step4_analysis/step4_regime_aggregate.csv`
- `output/step4_analysis/step4_regime_master.csv`
- `output/step4_analysis/step4_report.txt`
- `output/step4_analysis/step4_top10_runs.csv`
- `plots/`

## Quick start

Requires [uv](https://docs.astral.sh/uv/), `g++` (C++17) and `make`.

```bash
uv sync --frozen      # pinned Python env (Python 3.12)
make test             # lint + 25 tests (builds bin/step3_mm)
make all              # engine -> Hawkes diagnostics + fill calibration -> multi-seed sweep -> plots (~4 min)
```

The raw Databento files are licensed and not in the repo. Steps 1–2 run from the committed outputs unless the raw
data is present; see [`data/README.md`](data/README.md). Individual targets: `make features | hawkes | build | engine
| diagnostics | analysis | plots | bench`.

## Repository structure

```text
├── engine/event_driven_mm_engine.cpp   # step 3 C++ engine (make build -> bin/step3_mm)
├── scripts/
│   ├── step1_build_feature_table.py    # raw MBP-10 + trades -> 1 s feature table
│   ├── step2_trade_label_and_hawkes.py # trade signs, counts, EWMA + discrete Hawkes intensities
│   ├── hawkes_diagnostics.py           # calibration diagnostics -> docs/hawkes_calibration.md
│   ├── calibrate_fill_model.py         # estimates the fill model's A, k from trades
│   ├── step4_analysis.py               # multi-seed sweep, paired Hawkes vs baseline, ablation
│   ├── plot_graphs.py                  # review plots -> plots/
│   ├── benchmark_engine.py             # throughput / latency benchmark
│   └── download_script.py              # Databento download (costs credits)
├── output/                             # step1_features, step2_hawkes, step3_engine_cpp, step4_analysis,
│                                       # hawkes_diagnostics, fill_calibration
├── plots/  benchmark_results/
├── tests/                              # smoke, engine behaviour, no-look-ahead tests
├── docs/                               # changes.md, hawkes_calibration.md
├── data/README.md                      # how to obtain the (gitignored) raw data
├── Makefile  pyproject.toml  uv.lock  .github/workflows/ci.yml
└── plan.md  Report.md
```

## Strategy design

The strategy uses an Avellaneda-Stoikov style market-making core and augments it with microstructure-aware adjustments.

The engine incorporates:

- reservation-price and spread logic,
- order flow imbalance and microprice skew,
- Hawkes intensity signals,
- inventory limits,
- queue haircut logic,
- toxicity haircuts,
- maker fees,
- fill probability modeling,
- markout tracking,
- and latency profiling.

This combination is important because it makes the simulator far more realistic than a naive quoting model that ignores execution quality and inventory risk.

## Key results

All numbers: AAPL, 2024-06-03, one session, in-sample, simulated fills. Mean over 10 seeds; 95% t-intervals on
paired differences. 24 trading configurations × 10 seeds = 240 engine runs. Source:
`output/step4_analysis/step4_report.txt`.

### Hawkes vs baseline

| Configuration | Hawkes PnL | Baseline PnL | Hawkes − baseline (95% CI) |
|---|---:|---:|---:|
| Best grid config (A=0.12, k=4) | 11,001 | 10,091 | +910 [841, 980] |
| Default (A=0.08, k=6) | 4,931 | 4,553 | +377 [336, 418] |
| Calibrated fill model (A=0.066, k=58) | 260 | 213 | +46 [23, 70] |
| Default, **OFI signal unused in quoting** | | | +0.2 [−51, 51] |
| Calibrated, **OFI signal unused in quoting** | | | −19 [−40, 2] |

**Conclusion: inconclusive about Hawkes as a signal, negative about the current signal logic.** Hawkes beats the
baseline in every seed, but the ablation removes the whole gap. The OFI-driven skew, widening and pulls *lose* money
in this simulator in both modes. The Hawkes OFI is smoother and rarely crosses the ±0.2 toxicity threshold, so it
loses less. Whether Hawkes intensities carry predictive information needs an out-of-sample prediction test (plan.md
Phase 3), not this fill simulator.

### Fill model and Hawkes calibration

- Calibrated from trades, the fill decay is **k ≈ 58 per dollar**, against the 4–8 assumed in the sweep. The grid
  configurations quote ~16 ticks from mid and get thousands of times more fills there than the data support (≈7,000×, extrapolated past the 10-tick data range). Their ~$5–11k/day is
  an artefact of the fill model. At the calibrated point the strategy quotes ~1.7 ticks from mid and makes ~$200–260/day
  in simulation, before any queue or latency realism.
- Hawkes ([report](docs/hawkes_calibration.md)): clustering is overwhelming against a Poisson null, and
  cross-excitation earns its keep. But 31% of 1 s buckets hold ≥3 trades, sub-window branching ratios (0.27–0.64) sit
  well below the full-day 0.82 (intraday seasonality is absorbed as excitation), and the exact event-time fit is
  dominated by millisecond-scale excitation the 1 s model cannot see. KS rejects every kernel tried.

### Engine benchmark

| Metric | Value |
|---|---:|
| Mean throughput (rows/s, including CSV parse and write) | ~127,000 |
| Median per-tick p50 / p99 latency | 0.07 µs / 0.14 µs |
| Latency budget exceedances | 0 |

Measured on an AMD Ryzen 9 7940HS (`make bench`). Per-tick latency covers the quoting and fill arithmetic of one
1-second row. It says nothing about exchange round-trip latency.

## Plots

`plots/` (from `make analysis`): `01` PnL path and `02` inventory path (best configs, seed 1), `03` markout by
horizon, `04` PnL decomposition, `05_*` regime comparisons, `06` latency CDF. `output/hawkes_diagnostics/` has the
calibration figures (bursts vs intensity, Hawkes vs EWMA OFI, Q–Q, occupancy, parameter stability), and
`output/fill_calibration/touch_curve.png` has the fill-model fit.

## Limitations

- **One day, one symbol.** Every number is in-sample on AAPL 2024-06-03. The CIs cover fill-simulation noise only,
  not day-to-day variation. There is no holdout.
- **Fills are simulated** from a parametric model that also consumes the Hawkes intensity (partly circular). There is
  no queue position, latency or trade-through check. The calibrated `A, k` measure touches, which upper-bound fills.
- **1-second granularity.** Real market-making decisions happen at microsecond scale.
- **In-sample Hawkes parameters.** They are fitted on the whole day and then used on the same day.
- **Labels.** The committed step 2 outputs were regenerated without the raw data (see `docs/changes.md` §2).
- Hawkes is a useful **descriptive** model of order-flow clustering. Nothing here shows it is a profitable strategy.

## Next steps (from plan.md)

Walk-forward evaluation on more days; an out-of-sample short-horizon prediction test with and without Hawkes
features; a queue-aware replay simulator (needs MBO data); a time-varying Hawkes baseline; parameter uncertainty.
