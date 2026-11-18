# Hawkes Market Maker

Hawkes Market Maker is a low-latency market-making research and engineering project that combines microstructure-driven feature engineering, Hawkes-process order flow modeling, and an event-driven C++ quoting engine into one end-to-end pipeline. The project starts from raw market data, builds replayable features, calibrates Hawkes intensities, runs a queue- and toxicity-aware market-making engine, and then evaluates the strategy against a baseline through PnL, markout, regime, and latency analysis. Relative to the original version, several engine bugs were fixed, labels corrected, the fill model calibrated and the evaluation made multi-seed. The Hawkes calibration diagnostics are in `output/hawkes_diagnostics/`.

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
- `output/step4_analysis/step4_hawkes_vs_baseline_paired.csv`, `step4_ablation_signal_off.csv`
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

**Outputs:** `output/step2b_causal/`, `output/step4b_fixed/step4b_paired.csv`, `step4b_replay_runs.csv`

### Step 5: Alpha-signal ablation (does Hawkes predict anything?)

The step 3/4 result shows Hawkes-driven quoting beats baseline-driven quoting in the fill simulator, but that result
collapses once the OFI signal is removed from quoting (see Key results below) — so it doesn't say whether the Hawkes
signal is actually informative. This stage answers that directly, independent of the engine: a walk-forward ridge
regression predicts short-horizon price moves and next-second order flow from the feature table, with and without
the Hawkes features, and reports out-of-sample R², hit rate, and a markout-style economic value with confidence
intervals. A third model swaps the 1 s Hawkes features for **event-time features**: exponentially decayed counts of
buy and sell aggressor events at 0.1 s, 1 s and 10 s half-lives, computed from individual trade timestamps, so they
see the millisecond clustering that 1 s buckets average away. A literature check found that an HJB + Hawkes
optimal-control reformulation (arXiv:2510.26438) is not a tractable next step: no closed form, no converged solver.

**Outputs**

- `output/step5_alpha_signal/step5_ablation_summary.csv`
- `output/step5_alpha_signal/step5_fold_metrics.csv`
- `output/step5_alpha_signal/step5_paired_significance.csv`, `step5_regime_breakdown.csv`
- `plots/08_alpha_signal_oos_r2.png`

### Step 6: Multi-day validation (Binance BTCUSDT)

`make validate-binance` runs the step 5 code on seven BTCUSDT days (2024-06-03 to 06-09). The data is free and
public with no API key (`scripts/download_binance.py`). It does three things:

- **Per-day Hawkes fit:** are the parameters stable across days?
- **Within-day walk-forward:** step 5's test, run on each day separately.
- **Cross-day test:** fit on day *d−1* and predict day *d*, then a t-test across the six test days. Days are close
  to independent, so this is the strongest evidence in the project.

It also exports the models for the live monitor. Not part of `make all` (about 550 MB to download once). BTCUSDT 2024-07-01 to 07-07 is a recorded holdout
(`config/holdout.json`) that the scripts refuse to touch without `--allow-holdout`.

**Outputs:** `output/step6_binance_validation/`, `plots/09_binance_oos_r2.png`

### Step 7: Hawkes ensemble (the final model)

`make ensemble` builds one forecaster from every Hawkes signal in the project (20 features). It combines:

- **Baseline:** the 5 flow features.
- **1 s Hawkes:** the discrete intensity.
- **Exact event-time Hawkes:** a 3-kernel bivariate Hawkes fitted by maximum likelihood on event timestamps. The
  fitted half-lives are about 3 ms, 0.1 s and 5 s.
- **Event-time kernels:** the multi-scale decayed counts.

Its ridge penalty is chosen by chronological cross-validation inside the training data. It is tested against
every single-family model on the BTCUSDT cross-day test and on AAPL. The live monitor runs it.
Outputs: `output/step7_hawkes_ensemble/`.

### Step 8: Execution (making a forced trade cheaper)

`make execution`: a fund must trade $10,000 of BTCUSDT every minute (alternating buy and sell) within 10 seconds,
on the 6 out-of-sample days. Two strategies use each model's forecast:

- **Timing:** a taker that waits while the forecast says the price is moving its way.
- **Passive:** rests at the best price as a maker, and crosses when the forecast turns against it or at the
  deadline.

Benchmark: trade immediately. Outputs: `output/step8_execution/`.

| Result (8,628 orders per strategy) | Saving vs trading immediately |
|---|---|
| Timing, Hawkes ensemble | **+0.118 bp per order** [0.100, 0.136], better on 6/6 days |
| Timing, baseline signal | +0.093 bp, 6/6 days |
| Hawkes-family minus baseline, timing | +0.025 bp (p ≈ 0.04, 5/6 days): suggestive |
| Passive with a forecast minus blind passive | +0.19 bp (6/6 days, p = 0.003) |
| Passive at the standard tier (maker = taker fee) | −0.08 bp (maker fills are adversely selected) |

**Fees.** The execution rows report fees as a *difference* against the benchmark, not as fees paid.
Fees are charged as a percentage (10 bp) of each fill's dollar value, price × quantity. Each order's quantity is fixed at arrival, so a fill at a better price than the benchmark costs fewer dollars and pays a slightly smaller fee: the fee difference (strategy fees minus benchmark fees) is then negative. A fill at a worse price makes it positive. At the standard tier, where maker and taker fees are equal, it is about 1,000 times smaller than the price saving (a few dollars of price improvement on a ~$65,000 price moves a $10 fee by a fraction of a cent), and its sign follows the price saving. The live dashboard rounds it, so it can show −$0.01.

This is the project's honest money result. The forecasts don't pay for trading on their own, but they make
trades a fund has to do anyway measurably cheaper. $0.12 per $10k is about $1,200 a day for a desk executing
$100M a day. This assumes zero latency and approximates bid/ask from the last trade; the live dashboard runs the
same strategies on the real order book, with the ensemble's two rows in the leaderboard in red.

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
| Hawkes ensemble | +$22,697 | $3.3 bn | −$3.3M | 0.068 bp |

The forecasts carry a real gross edge. That gross is an upper bound: it assumes zero latency, trading at the exact
last price as each second closes. The strategy flips about 30,000 times a day, so it only breaks even at fees about
150 times below the standard rate. Hawkes features add a little gross and cut turnover by 23%, which helps, but
nowhere near enough. **Paper market maker (option 2).** The dashboard runs six paper market-maker books on the live order book:
- an at-touch control;
- Avellaneda–Stoikov quotes as in the C++ engine, with k calibrated on BTCUSDT (`make maker-calibration`);
- the same A–S quotes plus each model's forecast, as reservation skew and widen/pull protection.

Fills come from real trades with FIFO queues. Each book shows spread captured, 1 s and 10 s markouts, and P&L at zero
and standard maker fees, and every quote and fill is recorded. `make reconcile DATE=...` replays a recorded session
through the engine's per-second fill rule and compares it with the live fills.

First minutes: markouts are about −1 bp (fills come mostly when the price runs through the quote), so zero-fee P&L
is negative for every book and fees are far larger. The A–S books trade about 8× less than the at-touch control and
lose less. A full-day run is needed to compare the signals.

### More pairs, side by side (the hub)

`make train-pairs` downloads a week of eight more Binance pairs (ETH, SOL, BNB, XRP, DOGE, ADA, LINK, NEAR;
2026-11-10 to 16). `scripts/train_pair.py` then runs the step 6–7 cross-day test on each pair: every parameter is
fitted on day d−1 and tested on day d, for the baseline, +Hawkes, +event-time, exact Hawkes and the ensemble. It
also calibrates the pair's maker (tick from `exchangeInfo`, k from a tick-grid touch curve) and writes
`output/pairs/<SYMBOL>/` (`live_model.json`, `cross_day_summary.csv`, `validation.json`, ...).

`make hub` starts one `live_monitor.py` process per pair, each with its own stream, port, paper books and recording
under `data/live_pairs/<SYMBOL>/`, so they run in parallel and fail independently. It then serves
<http://127.0.0.1:8800>, which toggles between the pairs: an overview of who is leading on each pair and its offline
validation, plus each pair's full dashboard. `config/live_pairs.json` lists the pairs and ports (BTCUSDT
keeps :8765 and its 2024-trained step 6–7 models). A pair set to `managed: false` is shown but never started or
stopped. Each pair also runs two **event-driven** makers: `ev_none` re-prices the same A–S quotes on every trade and
quote change (speed alone), and `ev_hawkes` adds a per-trade pull driven by the exact Hawkes intensity. Each
decision reaches the market 50 ms later (`--ev-latency-ms`). `live_hub.py status` prints one line per pair, and `live_hub.py stop ETHUSDT` stops a single pair.

Offline validation (`output/pairs/*/validation.json`): the Hawkes ensemble beat the baseline on the 1 s return on
every test day for ETH, SOL, BNB, XRP, DOGE and ADA (p < 0.0125, the Bonferroni level). NEAR was better on 6/6 days
but not significant (p = 0.06). LINK was 5/6 and not significant. On the 1 s order-flow target the ensemble won
6/6 days on every pair except XRP.

**Every live strategy is also validated offline** (`make validate-strategies`, results in `output/validation/`):
each test day is replayed trade by trade through the
live monitor's own code, with models built from the day before. Results that set the live configuration:
- **Execution** is where the Hawkes signal earns money. Any signal beats passive-without-signal on 7/9 pairs; the
  best Hawkes model is pair-specific (ensemble on ETH, LINK, NEAR; 1 s Hawkes on XRP, DOGE). The choice per pair is
  written into each model file (`display.exec_model`, ★ on the dashboard).
- **Hawkes execution books are event-driven by default.** On every trade they cross when a Hawkes burst (log
  buy/sell intensity from kernels slower than the 50 ms latency, in the top/bottom 2% of its recent history) runs
  against the order. Against once-a-second twins on identical orders this is better on 9/9 pairs (passive) and 6/9
  (timing). Following the touch (re-pegging) lost on 8/9 pairs and is not used.
- **Market makers:** every A–S variant is within about 0.03 bp of the others at 0 fees; the forecast does not change
  maker outcomes. `make reconcile DATE=...` compares live fills with the C++ engine's per-second rule: per-fill
  markouts agree to about 0.13 bp.

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
data is present (`scripts/download_script.py` fetches it from Databento; it costs credits). Individual targets: `make features | hawkes | build | engine
| diagnostics | analysis | causal | fixed | alpha | validate-binance | live | live-replay | plots | bench`.

## Repository structure

```text
├── engine/event_driven_mm_engine.cpp   # step 3 C++ engine (make build -> bin/step3_mm)
├── scripts/
│   ├── step1_build_feature_table.py    # raw MBP-10 + trades -> 1 s feature table
│   ├── step2_trade_label_and_hawkes.py # trade signs, counts, EWMA + discrete Hawkes intensities
│   ├── hawkes_diagnostics.py           # calibration diagnostics -> output/hawkes_diagnostics/
│   ├── calibrate_fill_model.py         # estimates the fill model's A, k from trades
│   ├── step2b_causal_features.py       # walk-forward Hawkes, corrected labels, trade file for replay fills
│   ├── step4_analysis.py               # original multi-seed sweep, paired Hawkes vs baseline, ablation
│   ├── step4b_fixed_comparison.py      # corrected comparison: replay fills, standardised OFI, causal features
│   ├── step5_alpha_signal.py           # out-of-sample prediction test (AAPL)
│   ├── step6_binance_validation.py     # the same test on 7 BTCUSDT days, within- and cross-day
│   ├── step7_hawkes_ensemble.py        # exact Hawkes MLE + the Hawkes ensemble (BTCUSDT cross-day, AAPL)
│   ├── step8_execution.py              # BTCUSDT offline execution test (timing / passive)
│   ├── taker_backtest.py               # paper P&L of trading the 1 s forecasts
│   ├── calibrate_btc_maker.py          # A-S fill decay k for the BTCUSDT paper maker
│   ├── build_pair_model.py             # maker calibration + features for any Binance pair
│   ├── train_pair.py                   # cross-day training/validation of every model for one pair
│   ├── validate_live_strategies.py     # replays test days through the live code (every strategy)
│   ├── summarise_validation.py         # summary CSVs + per-pair execution model (display.exec_model)
│   ├── reconcile_live_maker.py         # live maker fills vs the engine's per-second fill rule
│   ├── download_binance.py             # public Binance aggTrades (no key); enforces the holdout
│   ├── live_monitor.py, live_dashboard.html  # one pair: collector, models, paper strategies, dashboard
│   ├── live_hub.py, live_hub.html      # runs every pair in config/live_pairs.json; hub page on :8800
│   ├── plot_graphs.py                  # review plots -> plots/
│   ├── benchmark_engine.py             # throughput / latency benchmark
│   └── download_script.py              # Databento download (costs credits)
├── output/                             # one folder per step; pairs/ (per-pair models), validation/
├── plots/  benchmark_results/
├── tests/                              # smoke, engine behaviour, no-look-ahead, gap-fix and live-parity tests
├── docs/                               # local notes and write-ups (git-ignored)
├── config/holdout.json, live_pairs.json  # recorded holdout; pairs, ports and models the hub runs
├── Makefile  pyproject.toml  uv.lock  .github/workflows/ci.yml
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
`output/step4_analysis/step4_hawkes_vs_baseline_paired.csv`.

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
Newey–West (lag 300 s) on per-second PnL increment differences. Source: `output/step4b_fixed/step4b_paired.csv`.

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
money-losing quoting rule less often. Step 5 tests this directly: a
walk-forward (5-fold, chronological) ridge regression predicts 1/5/10-second-ahead returns and next-second order
flow from the feature table, with and without the Hawkes features.

Hawkes parameters are re-estimated on each fold's training window, and the two models are compared with paired
tests on the per-row squared-error difference: a fold-level t-test and a Diebold–Mariano test with a Newey–West
variance, judged at a Bonferroni level of p < 0.0125 for four targets.

**AAPL (one day): null.** Adding Hawkes features improves no target. At 10 s the Hawkes model is, if anything,
slightly *worse* (DM p = 0.019, not Bonferroni-significant). Event-time features help next-second flow but hurt the
return forecasts: nine extra regressors overfit on one day.

**BTCUSDT (seven days): the Hawkes ensemble is the best model.** Cross-day (fit on d−1, test on d, six test days;
`output/step7_hawkes_ensemble/cross_day_summary.csv`):

| Model | Return tests better (of 18) | Flow, days better | p across days (1 s / 5 s / 10 s / flow) | Hit gain, 5 s return | R², 1 s |
|---|---:|---:|---|---:|---:|
| +Hawkes (1 s) | 13 | 5/6 | 0.09 / 0.06 / 0.21 / 0.35 | +2.8 pp | 0.007 |
| +Event-time kernels | 15 | 6/6 | 0.03 / 0.016 / 0.024 / 0.0006 | +1.8 pp | 0.010 |
| +Exact Hawkes | 15 | 6/6 | 0.03 / 0.020 / 0.031 / 0.003 | +0.6 pp | 0.006 |
| **Hawkes ensemble** | **18** | **6/6** | **0.012 / 0.005 / 0.004 / 0.0005** | **+3.3 pp** | **0.021** |

The ensemble is better than the baseline on all 24 day×target tests. It is the only model significant on every
target at the Bonferroni threshold (p < 0.0125), and it roughly doubles R². AAPL (one day) is still not
significant for any model. There the ensemble's tuned penalty stops the overfitting that made the event-time model
worse than the baseline. Still one crypto pair over one week; the holdout week is untouched.

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
- Hawkes (`output/hawkes_diagnostics/`): clustering is overwhelming against a Poisson null, and
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
- **Labels.** The committed AAPL step 2 outputs were regenerated without the raw data.
- Hawkes is a useful **descriptive** model of order-flow clustering and, on BTCUSDT, a modestly useful feature. Nothing
  here shows a profitable strategy: with real-trade fills the AAPL market maker loses money.

## Next steps

- More days per instrument, then the single holdout evaluation (`config/holdout.json`).
- Event-time Hawkes features inside the quoting engine, and an event-driven (sub-second) engine to use them.
- A queue-aware simulator on MBO data, to replace the conservative replay.
- An HJB + Hawkes controller only after the above. The gating question, whether Hawkes features
  forecast out of sample, now has a qualified yes on BTCUSDT and a no on AAPL. A recent paper attempting exactly this
  (arXiv:2510.26438) found no closed form and its numerical solver did not converge.
