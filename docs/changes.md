# What changed and why

This builds on the original repo (upstream commit `60c5656`) with the improvements suggested in `plan.md`.
Every behaviour change below was found by reading the code and confirmed numerically on the committed data
(AAPL, 2024-06-03). Before/after numbers are for the default engine config (`--use-hawkes`, γ=0.10, k=6, A=0.08,
seed 42) unless stated otherwise.

## 1. Engine fixes (`engine/event_driven_mm_engine.cpp`)

| # | Problem in the original | Fix |
|---|---|---|
| E1 | **Markout sign inverted.** `bid_markout = fill − future_mid` is negative when a buy is profitable (likewise for asks). The reported "avg markout −0.25" meant fills were on average *profitable* by about $0.25/share. | Markout = `future_mid − fill` (bid) and `fill − future_mid` (ask). Positive = profitable. |
| E2 | **"Adverse selection cost" measured gains.** It was `max(0, −markout)·qty`, which counted the favourable distance between fill price and mid (≈ the spread earned). That is why it equalled the realized spread (5,290 vs 5,274). | Adverse selection = mid move *against* the position from the mid at fill to the horizon mid: `max(0, mid − future_mid)·qty` for buys, mirrored for sells. |
| E3 | **Toxicity logic protected the wrong side.** With buy-flow imbalance (OFI > 0) the engine widened, pulled and hair-cut the **bid**, but informed buyers lift the **ask**. | Buy flow widens/pulls/haircuts the ask; sell flow the bid. |
| E4 | **Inventory-risk term ≈ 0.** σ = `rv_60` is a 1-second *log-return* std (~6e-5), used as if it were in dollars, with τ ∈ [0,1]. `q·γ·σ²·τ` was at most 5e-6 dollars, so inventory was controlled only by the hard limit (max \|q\| = 500, hit 2,946 times). | σ is converted to price units (`rv_60 · mid`), and τ = min(`--risk-horizon-s`, seconds left) with a default of 60 s. `--risk-horizon-s 0` restores the session-fraction τ. |
| E5 | Warm-up volatility fallback was 0.001 (≈16× typical `rv_60`). | `--sigma-fallback`, default 1e-4. |
| E6 | One-sided pull threshold hard-coded at 0.45. | `--toxic-pull-threshold` (default 0.45). |
| E7 | `seed` and `allow_one_sided_pull` were missing from the summary JSON; `risk_rejects` was counted but never written. Output dir created via `std::system("mkdir -p …")`. | Added to summary (`params.seed`, `params.allow_one_sided_pull`, `risk_reject_steps`, and the new params). Uses `std::filesystem::create_directories`. |

Default config, before → after: final PnL 5,463.57 → 4,666.81; max \|inventory\| 500 → 175; avg markout per bid fill
−0.164 → +0.140 (sign fix, same economics); total adverse selection 5,289.9 → 135.3 (now measures what its name
says). CLI flags, `MMParams` names and the `engine_results.csv` columns are unchanged (new flags only), so `step4`
and the plots still read the outputs.

## 2. Trade labelling (`scripts/step1…`, `scripts/step2…`)

| # | Problem | Fix |
|---|---|---|
| L1 | **Lee–Ready look-ahead.** Step 1 rows are labelled by bucket start but hold the book at the bucket **end**, and `merge_asof(backward)` matched each trade to quotes up to 1 s in its future. | Quotes are re-stamped at the bucket end before matching. |
| L2 | **Native aggressor side discarded.** The substring regex mapped Databento's `'A'` (sell aggressor) to NaN and `'-1'` to *buy*, and Lee–Ready overrode the native side anyway. | Exact token mapping (`map_aggressor_side`). The exchange side is used when present; Lee–Ready only labels the rest. `label_source` records which rule was used. |

Measured on the 70,565 trades with a known buy-aggressor flag: the original labels called **53.4%** of them buys
(a coin flip). The look-ahead fix alone gives **84.2%**, and with L2 they are correct by construction.

**Regenerated committed outputs (caveat):** the raw Databento files are not in the repo, so `output/step2_hawkes/` was
regenerated from the committed `trades_labeled.parquet` (`--trades` accepts it). That file kept only "buy aggressor or
unknown" (the old regex lost `'A'`), so 94k trades were labelled by the fixed Lee–Ready. With raw data, `make hawkes`
would label them from the native side. Buy share 53.6% → 59.1%. Hawkes fit: β 0.9458 → 0.9419, μ_sell 0.025 → 0.060.
Step 1 outputs are unchanged (its `buy_count/sell_count` still carry the old regex labels until rerun with raw data;
the engine does not use them).

## 3. New analysis

- **Fill-model calibration** (`scripts/calibrate_fill_model.py`). The touch intensity λ(δ) is estimated from
  next-second trades against the end-of-bucket mid, and fitted as ln λ = ln Λ₀ − kδ. Result: **k ≈ 58 per dollar**,
  against the 4–8 assumed in the sweep. At the legacy quote distance (16.5 ticks) the assumed model overstates the
  touch rate by several thousand times (≈7,000×, extrapolated beyond the 10-tick range where there is data), so the legacy PnL (~$5k/day) is mostly produced by the fill model. At the calibrated
  point (A≈0.066, k≈58) the strategy quotes about 1.7 ticks from mid and makes ~$200–260/day in simulation.
- **Hawkes calibration diagnostics** (`scripts/hawkes_diagnostics.py` → `docs/hawkes_calibration.md`). Covers occupancy,
  null models, stability, bucket-width and sub-window refits, and an exact continuous-time MLE (1–3 exponentials) with
  time-rescaling KS and Q–Q.
- **Step 4 rewritten for statistical honesty.** 10 seeds per configuration. Hawkes vs baseline is paired on
  (A, k, seed) with a 95% t-interval, and the trial count is printed. Markout horizons are re-measured from one run (the
  old sweep ran 54 "runs" that were 18 distinct trading configurations). Regimes are shared between modes. The
  calibrated (A, k) is added. An **ablation** switches the OFI signal off in quoting. Output file names are unchanged;
  new files: `step4_seed_runs.csv`, `step4_hawkes_vs_baseline_paired.csv`, `step4_ablation_signal_off.csv`.

## 4. Headline result, before → after

| | Original claim | Now |
|---|---|---|
| Evidence | best of 27 configs per mode, 1 seed | 10 seeds per config, paired CIs, 240 engine runs reported |
| Hawkes − baseline, best config | +869 PnL (11,592 vs 10,723) | +910, CI [841, 980] (A=0.12, k=4) |
| Hawkes − baseline, calibrated fill model | n/a | +46.5, CI [23, 70] |
| **…with the OFI signal unused in quoting** | n/a | **+0.2, CI [−51, 51] (default); −18.6, CI [−40, 2] (calibrated)** |

Interpretation: the whole Hawkes advantage comes from how each OFI signal drives the toxicity and skew logic. That
logic *loses* money in this simulator in both modes:

| PnL change from using the OFI signal (signal-on − signal-off, 10 seeds, ±95%) | Hawkes OFI | EWMA OFI |
|---|---:|---:|
| default (A=0.08, k=6) | −267 ± 43 | −644 ± 50 |
| calibrated | +7 ± 12 | −58 ± 14 |

The Hawkes OFI is smoother (q99 ≈ 0.14, against the ±0.2 toxicity threshold), so it triggers the costly logic less.
That is **not** evidence that Hawkes intensities predict order flow better. Whether they do is a prediction question
(plan.md Phase 3), not answerable with this fill simulator.

## 5. Tooling and hygiene

- `pyproject.toml` + `uv.lock` with exact pins (pandas 2.3 and matplotlib 3.10: pandas 3 / matplotlib 3.11 break the
  original scripts), plus `Makefile`, GitHub Actions CI, and `tests/` (25 tests: smoke, engine behaviour on
  synthetic inputs, no-look-ahead checks for the Hawkes recursion, EWMA, rolling vol and Lee–Ready).
- `bin/`, per-run `engine_results.csv` files (8.5 MB each) and `benchmark_results/runs/` are untracked and regenerated.
  Raw data, `.env` and the experiment store are gitignored. See `data/README.md` for how to obtain the data.
- `scripts/benchmark_engine.py` works without GNU `/usr/bin/time` (peak RSS reported as n/a).
- `step4`/plots: `boxplot(labels=…)` → `tick_labels=…`.
