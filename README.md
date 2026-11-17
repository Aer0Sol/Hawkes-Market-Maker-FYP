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

### Step 4b: Corrected engine comparison

Step 4 is kept as the original comparison, but three things stop it answering whether Hawkes helps the market maker:
the engine reads full-day, stale Hawkes intensities; one ±0.2 threshold serves two signals of different scale; and
in Hawkes runs the Hawkes intensity also sets the fill rate. `make fixed` removes all three:

- `scripts/step2b_causal_features.py` builds a causal feature table. Hawkes parameters are refitted every 30 minutes
  on past data only, rows carry the one-step-ahead intensity, and trade imbalance uses the corrected labels.
- The engine standardises whichever OFI it uses causally (`--ofi-standardize`), so thresholds mean the same number
  of standard deviations for both signals.
- Fills are **replayed from the next second's real trades** (`--fill-model replay`). A trade printed through our
  price fills us in full. Trades at our price first consume the displayed queue ahead of us (FIFO, cancellations
  ignored, so conservative). There is no RNG and no fill model that the signal can influence.

**Outputs:** `output/step2b_causal/`, `output/step4b_fixed/step4b_report.txt`, `step4b_paired.csv`

### Step 5: Alpha-signal ablation (does Hawkes predict anything?)

The step 3/4 result shows Hawkes-driven quoting beats baseline-driven quoting in the fill simulator, but that result
collapses once the OFI signal is removed from quoting (see Key results below) — so it doesn't say whether the Hawkes
signal is actually informative. This stage answers that directly, independent of the engine: a walk-forward ridge
regression predicts short-horizon price moves and next-second order flow from the feature table, with and without
the Hawkes features, and reports out-of-sample R², hit rate, and a markout-style economic value with confidence
intervals. A third model swaps the 1 s Hawkes features for **event-time features**: exponentially decayed counts of
buy and sell aggressor events at 0.1 s, 1 s and 10 s half-lives, computed from individual trade timestamps, so they
see the millisecond clustering that 1 s buckets average away. See [`docs/alpha_signal.md`](docs/alpha_signal.md) for
the method and the full result, including a literature check of whether an HJB + Hawkes optimal-control
reformulation (arXiv:2510.26438) is a tractable next step.

**Outputs**

- `output/step5_alpha_signal/step5_ablation_summary.csv`
- `output/step5_alpha_signal/step5_fold_metrics.csv`
- `output/step5_alpha_signal/step5_paired_significance.csv`, `step5_regime_breakdown.csv`
- `docs/alpha_signal.md`
- `plots/08_alpha_signal_oos_r2.png`

### Step 6: Multi-day validation (Binance BTCUSDT)

`make validate-binance` runs the step 5 code on seven BTCUSDT days (2024-06-03 to 06-09). The data is free and
public with no API key (`scripts/download_binance.py`). It does three things:

- **Per-day Hawkes fit:** are the parameters stable across days?
- **Within-day walk-forward:** step 5's test, run on each day separately.
- **Cross-day test:** fit on day *d−1* and predict day *d*, then a t-test across the six test days. Days are close
  to independent, so this is the strongest evidence in the project.

It also exports the models for the live monitor. Not part of `make all` (about 550 MB to download once). See
[`docs/binance_validation.md`](docs/binance_validation.md). BTCUSDT 2024-07-01 to 07-07 is a recorded holdout
(`config/holdout.json`) that the scripts refuse to touch without `--allow-holdout`.

**Outputs:** `output/step6_binance_validation/`, `docs/binance_validation.md`, `plots/09_binance_oos_r2.png`

### Live monitor

`make live` starts one process. It collects Binance's public BTCUSDT trade and best bid/ask streams (no key, no
orders) and rebuilds each second the exact features the step 6 models use. It predicts the next second with the
baseline, +Hawkes and +event-time models, scores each prediction when that second closes, and serves a dashboard at
<http://127.0.0.1:8765>. It also records the stream to `data/live/` in the historical format, with the best
bid/ask each second added. `make live-replay` runs the same dashboard on a downloaded day, for when the network
blocks Binance. It needs the optional `websockets` package: `uv sync --frozen --group live`.

**How much money?** The dashboard also runs a paper taker (option 1). Each model holds $10,000 long or short by the
sign of its forecast, trades only when the sign flips, and pays Binance's standard 10 bp taker fee. `make taker`
runs the same strategy on the 6 out-of-sample BTCUSDT days (`output/taker_backtest/`):

| Model | Gross, no fees | Traded | Net at 10 bp | Break-even fee |
|---|---:|---:|---:|---:|
| Baseline (EWMA) | +$20,000 | $4.1 bn | −$4.1M | 0.048 bp |
| +Hawkes | +$21,202 | $3.2 bn | −$3.2M | 0.067 bp |
| +Event-time | +$21,197 | $3.7 bn | −$3.7M | 0.057 bp |

The forecasts carry a real gross edge. That gross is an upper bound: it assumes zero latency, trading at the exact
last price as each second closes. The strategy flips about 30,000 times a day, so it only breaks even at fees about
150 times below the standard rate. Hawkes features add a little gross and cut turnover by 23%, which helps, but
nowhere near enough. Option 2, a paper market maker that earns the spread instead of paying it, is planned in
`plan.md` §8a.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), `g++` (C++17) and `make`.

```bash
uv sync --frozen --group live   # pinned Python env (Python 3.12) + websockets for the live monitor
make test             # lint + tests (builds bin/step3_mm)
make all              # engine, diagnostics, step 4 sweep, step 4b corrected comparison, step 5 (~4 min)
make validate-binance # step 6 on 7 BTCUSDT days (first run downloads ~550 MB)
make live             # live dashboard on http://127.0.0.1:8765 (Ctrl-C to stop)
```

The raw Databento files are licensed and not in the repo. Steps 1–2 run from the committed outputs unless the raw
data is present; see [`data/README.md`](data/README.md). Individual targets: `make features | hawkes | build | engine
| diagnostics | analysis | causal | fixed | alpha | validate-binance | live | live-replay | plots | bench`.

## Repository structure

```text
├── engine/event_driven_mm_engine.cpp   # step 3 C++ engine (make build -> bin/step3_mm)
├── scripts/
│   ├── step1_build_feature_table.py    # raw MBP-10 + trades -> 1 s feature table
│   ├── step2_trade_label_and_hawkes.py # trade signs, counts, EWMA + discrete Hawkes intensities
│   ├── hawkes_diagnostics.py           # calibration diagnostics -> docs/hawkes_calibration.md
│   ├── calibrate_fill_model.py         # estimates the fill model's A, k from trades
│   ├── step2b_causal_features.py       # walk-forward Hawkes, corrected labels, trade file for replay fills
│   ├── step4_analysis.py               # original multi-seed sweep, paired Hawkes vs baseline, ablation
│   ├── step4b_fixed_comparison.py      # corrected comparison: replay fills, standardised OFI, causal features
│   ├── step5_alpha_signal.py           # out-of-sample prediction test (AAPL)
│   ├── step6_binance_validation.py     # the same test on 7 BTCUSDT days, within- and cross-day
│   ├── download_binance.py             # public Binance aggTrades (no key); enforces the holdout
│   ├── live_monitor.py, live_dashboard.html  # live collector, scorer and dashboard
│   ├── plot_graphs.py                  # review plots -> plots/
│   ├── benchmark_engine.py             # throughput / latency benchmark
│   └── download_script.py              # Databento download (costs credits)
├── output/                             # one folder per step, plus hawkes_diagnostics, fill_calibration
├── plots/  benchmark_results/
├── tests/                              # smoke, engine behaviour, no-look-ahead, gap-fix and live-parity tests
├── docs/                               # changes.md, hawkes_calibration.md, alpha_signal.md,
│                                       # binance_validation.md, project_overview.html
├── config/holdout.json                 # recorded holdout window and frozen evaluation
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

Hawkes beats the baseline in every seed here, but the ablation removes the whole gap, and the setup has the three
confounds described under step 4b. **These numbers are superseded by the corrected comparison below.**

### Corrected engine comparison (step 4b)

Replay fills from real trades, causally standardised OFI, walk-forward Hawkes. Deterministic runs; 95% CIs are
Newey–West (lag 300 s) on per-second PnL increment differences. Source: `output/step4b_fixed/step4b_report.txt`.

| Configuration | Hawkes $ | Baseline $ | Hawkes − baseline (95% CI) | OFI rules on − off, Hawkes / baseline |
|---|---:|---:|---:|---:|
| Calibrated (k = 58) | −931 | −978 | +47 [−26, 120] | +115 [19, 210] / +68 [−28, 163] |
| At the touch (k = 200, 1-tick floor) | −1,485 | −1,489 | +4 [−78, 87] | +285 [179, 392] / +281 [176, 386] |
| Legacy grid default (k = 6) | −4 | −4 | 0 | −0.3 [−0.7, 0.2] |

**What changes:**

- **The strategy loses money once fills come from real trades.** It loses $900–1,800/day, with markouts of −0.6 to
  −0.8¢/share at 1 s. The legacy k = 6 quotes almost never fill (10 fills in a day). The $5–11k/day of the original
  step 4 was an artefact of a fill model that draws fills independently of later price moves, so it never pays for
  adverse selection.
- **Hawkes and baseline quoting are statistically indistinguishable.**
- **The OFI rules now *help*:** widening and pulling quotes ahead of one-sided flow avoids adverse fills. The
  earlier finding that they lose money was produced by the parametric fill model.
- **Parametric robustness check:** run with the same EWMA fill pressure in both modes, Hawkes is still ahead
  (+65 [47, 83] calibrated). That model cannot see adverse selection, so the replay result is the one to trust.

### Does the Hawkes signal predict anything? (out-of-sample, no fill simulator)

The step 3/4 result above says nothing about whether the Hawkes signal is informative — only that it triggers a
money-losing quoting rule less often. [`docs/alpha_signal.md`](docs/alpha_signal.md) tests this directly: a
walk-forward (5-fold, chronological) ridge regression predicts 1/5/10-second-ahead returns and next-second order
flow from the feature table, with and without the Hawkes features.

Hawkes parameters are re-estimated on each fold's training window, and the two models are compared with paired
tests on the per-row squared-error difference: a fold-level t-test and a Diebold–Mariano test with a Newey–West
variance, judged at a Bonferroni level of p < 0.0125 for four targets.

**AAPL (one day): null.** Adding Hawkes features improves no target. At 10 s the Hawkes model is, if anything,
slightly *worse* (DM p = 0.019, not Bonferroni-significant). Event-time features help next-second flow but hurt the
return forecasts: nine extra regressors overfit on one day.

**BTCUSDT (seven days, `docs/binance_validation.md`): small, consistent gains.**

| Test | +Hawkes | +Event-time |
|---|---|---|
| Within-day, days with a significant gain on the return targets (21 day×horizon tests) | 17 of 21 at p < 0.05, 4 at p < 0.0125 | 17 of 21 at p < 0.05, 3 at p < 0.0125 |
| Cross-day (fit d−1, test d), days better on return targets | 13 of 18, p across days 0.06–0.21 | 15 of 18, p 0.016–0.030 |
| Cross-day, next-second order flow | 5 of 6 days, p = 0.35 | **6 of 6 days, p = 0.0006** |
| Hit-rate gain on real moves (> 1.5 ticks), cross-day | +2.2 to +2.8 pp | +1.1 to +1.8 pp |

On BTCUSDT the Hawkes / event-time features carry information the EWMA baseline lacks, consistently across a week,
but the effect is small and only the event-time flow forecast clears a multiple-testing threshold across days. AAPL
says no, and running AAPL with the same flow-only baseline does not change that. Neither "Hawkes is proven to work"
nor "Hawkes is useless" is supported. The evidence points to the millisecond-scale event-time features, not the 1 s
discrete Hawkes model, as the more useful order-flow signal.

Method details that change the answer, all covered in the reports:

- Hawkes is refitted per training window, and the feature is the one-step-ahead intensity λ<sub>t+1|t</sub>.
- Significance uses HAC (Newey–West) tests.
- BTCUSDT hit rates ignore the one-tick moves that make up 88% of its one-second price changes. Those are the
  bid-ask bounce, not price discovery.

### Fill model and Hawkes calibration

- Calibrated from trades, the fill decay is **k ≈ 58 per dollar**, against the 4–8 assumed in the sweep. The grid
  configurations quote ~16 ticks from mid and get thousands of times more fills there than the data support (≈7,000×, extrapolated past the 10-tick data range). Their ~$5–11k/day is
  an artefact of the fill model. At the calibrated point the parametric model makes ~$200–260/day; with fills
  replayed from real trades (step 4b) the same strategy loses ~$930/day.
- Hawkes ([report](docs/hawkes_calibration.md)): clustering is overwhelming against a Poisson null, and
  cross-excitation earns its keep. But 34% of 1 s buy buckets hold ≥3 trades, sub-window branching ratios (0.24–0.75) sit
  well below the full-day 0.83, and the exact event-time fit is dominated by millisecond-scale excitation the 1 s model
  cannot see. KS rejects every kernel tried.
- **Time-of-day baseline:** giving μ one value per 30-minute block lowers the branching ratio from 0.83 to **0.46**
  and the half-life from 11.6 s to 3.7 s (LR = 4,150 for 24 extra parameters). About half of the apparent
  self-excitation was intraday seasonality.

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
horizon, `04` PnL decomposition, `05_*` regime comparisons, `06` latency CDF, `08` OOS R² ablation (`make alpha`),
`09` same ablation on BTCUSDT (`make validate-binance`). `output/hawkes_diagnostics/` has the calibration figures
(bursts vs intensity, Hawkes vs EWMA OFI, Q–Q, occupancy, parameter stability), and
`output/fill_calibration/touch_curve.png` has the fill-model fit.

## Limitations

- **AAPL is one day.** Every engine number (steps 3, 4, 4b) is on AAPL 2024-06-03 with no AAPL holdout; a new AAPL day
  needs a Databento purchase. BTCUSDT has seven days plus a recorded, untouched holdout week (`config/holdout.json`).
- **Replay fills are conservative but not exact.** Queue ahead is never reduced by cancellations, behind-the-touch
  orders fill only on trade-throughs, and quotes are assumed live for the whole next second (no latency). Exact queue
  position needs MBO data.
- **1-second decisions.** The engine quotes once per second; the event-time features look at milliseconds, the
  quoting does not.
- **Time-of-day baselines are fitted on the day they describe.** A multi-day profile needs more days per instrument.
- **Labels.** The committed AAPL step 2 outputs were regenerated without the raw data (see `docs/changes.md` §2).
- Hawkes is a useful **descriptive** model of order-flow clustering and, on BTCUSDT, a modestly useful feature. Nothing
  here shows a profitable strategy: with real-trade fills the AAPL market maker loses money.

## Next steps

- More days per instrument, then the single holdout evaluation (`config/holdout.json`).
- Event-time Hawkes features inside the quoting engine, and an event-driven (sub-second) engine to use them.
- A queue-aware simulator on MBO data, to replace the conservative replay.
- An HJB + Hawkes controller (plan.md Phase 6a) only after the above. The gating question, whether Hawkes features
  forecast out of sample, now has a qualified yes on BTCUSDT and a no on AAPL. A recent paper attempting exactly this
  (arXiv:2510.26438) found no closed form and its numerical solver did not converge (see `docs/alpha_signal.md`).
