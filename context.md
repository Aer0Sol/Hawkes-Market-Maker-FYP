# Project context

A short briefing for anyone joining or reviewing this project. Read this first, then `README.md` and `docs/changes.md` for detail.

## What the project is

A research simulator for **Hawkes-informed market making**:

- a market maker quotes buy and sell prices around the mid and earns the spread;
- trades arrive in bursts, so a **bivariate Hawkes model** (buys and sells exciting each other) is fitted to trade counts;
- its intensities are turned into an order-flow imbalance (OFI) signal;
- an **Avellaneda–Stoikov** quoting engine, written in C++, uses that signal to skew, widen or pull quotes;
- the question is whether the Hawkes signal helps compared with a simple moving-average (EWMA) baseline.

This is a final-year project fork of [Abhi-1122/Hawkes-Market-Maker](https://github.com/Abhi-1122/Hawkes-Market-Maker), in active development on the `repo-improvements` branch. It is **simulation only**: there is no live trading, and none is planned.

## Data

| | |
|---|---|
| Source | Databento, `XNAS.ITCH` (Nasdaq) |
| Instrument and day | AAPL, 2024-06-03, regular session 09:30–16:00 ET |
| Inputs | trades (164,922) and the top 10 order-book levels, resampled to 23,400 one-second rows |
| In the repo | only derived files (`output/step1_features/`, `output/step2_hawkes/`); raw data is licensed and git-ignored |

All results are **one day, one stock, in-sample, with simulated fills**. See `data/README.md` before requesting more data; it costs Databento credits.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Features | `scripts/step1_build_feature_table.py` | 1-second book features (needs raw data) |
| 2. Labels and Hawkes fit | `scripts/step2_trade_label_and_hawkes.py` | trade signs, Hawkes and EWMA intensities, OFI |
| 3. Engine | `engine/event_driven_mm_engine.cpp` → `bin/step3_mm` | per-second quotes, fills, PnL |
| 2b. Causal features | `scripts/step2b_causal_features.py` | walk-forward Hawkes, corrected labels, trade file for replay fills |
| 4. Evaluation (original) | `scripts/step4_analysis.py` | multi-seed sweep, paired CIs, signal-off ablation |
| 4b. Corrected evaluation | `scripts/step4b_fixed_comparison.py` | replay fills, standardised OFI, HAC CIs |
| 5. Alpha-signal ablation | `scripts/step5_alpha_signal.py` | walk-forward OOS ridge, Hawkes-vs-EWMA, `docs/alpha_signal.md` |
| 6. Multi-day validation | `scripts/download_binance.py`, `scripts/step6_binance_validation.py` | 7 BTCUSDT days, within- and cross-day, `docs/binance_validation.md` |
| Live | `scripts/live_monitor.py`, `scripts/live_dashboard.html` | live BTCUSDT dashboard and recorder |
| Diagnostics | `scripts/hawkes_diagnostics.py`, `scripts/calibrate_fill_model.py` | `docs/hawkes_calibration.md`, fill-model fit |
| Plots | `scripts/plot_graphs.py` | `plots/` |

```bash
uv sync --frozen --group live   # pinned Python 3.12 environment (+ websockets for the live monitor)
make all           # build, diagnostics, steps 4, 4b, 5 (steps 1-2 only if raw data is present)
make validate-binance   # step 6 (downloads ~550 MB once)
make test          # ruff + pytest
make live          # live dashboard on http://127.0.0.1:8765
```

## What changed from the original repo

Details and before/after numbers are in `docs/changes.md`.

**Engine bugs fixed:**
- **Markout sign was inverted:** "−0.25" meant fills were profitable.
- **"Adverse-selection cost" measured spread earned:** it now measures the mid moving against the position.
- **Toxicity logic protected the wrong side:** buy pressure now protects the ask.
- **The inventory-risk term was ≈ 0:** volatility was used in the wrong units. It is now in dollars, with a 60 s horizon. Max inventory fell from 500 (the hard limit) to 175.

**Trade labels:**
- The original labels used quotes up to 1 s *after* each trade.
- They also discarded the exchange's aggressor flag.
- Result: they agreed with the flag only 53% of the time. They now use the flag first and only past quotes otherwise.

**Fill model:** calibrated from trades, it gives k ≈ 58 per USD, against the 4–8 assumed. The old setup overstated fills about 7,000×, so most of the old 5–11k USD/day simulated PnL was an artefact. The calibrated PnL is about 200–260 USD/day.

**New evidence:**
- Hawkes diagnostics: occupancy, null models, stability, bucket-width and sub-window refits, and an exact event-time fit with a time-rescaling test.
- 10 paired seeds per configuration with 95% CIs (240 runs).
- A signal-off ablation.

**Tooling:** pinned environment (`uv.lock`), Makefile, GitHub Actions CI, 25 tests (engine behaviour, no look-ahead, side mapping).

## Current findings

**Market maker (AAPL, corrected comparison, `output/step4b_fixed/`).** Fills replayed from real trades, OFI
standardised causally, walk-forward Hawkes:

| Config | Hawkes − baseline $ (95% CI) | OFI rules on − off, Hawkes |
|---|---|---|
| Calibrated (k = 58) | +47 [−26, 120] | +115 [19, 210] |
| At the touch (k = 200) | +4 [−78, 87] | +285 [179, 392] |

- The strategy **loses** ~$930–1,800/day once fills come from real trades. The old $5–11k/day came from a fill model
  that never charges for adverse selection.
- Hawkes and baseline quoting are indistinguishable. The OFI protection rules now *help*; the old "they lose money"
  finding was also a fill-model artefact.

**Forecasting (walk-forward, Hawkes refit per training window, HAC tests):**

- AAPL, one day: null for Hawkes; event-time features help next-second flow but hurt returns (overfitting).
- BTCUSDT, seven days: small, consistent gains. Cross-day (train d−1, test d), event-time features beat the baseline on
  6 of 6 days for next-second flow (p = 0.0006) and 15 of 18 day×horizon return tests; Hawkes on 13 of 18. Hit-rate
  gains +1 to +3 pp on moves larger than the bid-ask bounce.

**Hawkes calibration:**

- AAPL: the full-day branching ratio of 0.83 falls to **0.46** with a 30-minute time-of-day baseline, and the
  half-life from 11.6 s to 3.7 s. About half of the "excitation" was intraday seasonality.
- BTCUSDT: branching 0.58–0.87 across days, 0.44–0.74 with an hour-of-day baseline; half-life 3–7 s.
- Excitation is strongest at millisecond scale; KS rejects every kernel. Hawkes is a descriptive feature, not a correct
  model.
- The branching ratio in this discrete model is ρ(α)/(1−β), **not** α/β.

## Open items

1. **More days per instrument**, then the single evaluation of the recorded holdout (BTCUSDT 2024-07-01 to 07-07,
   `config/holdout.json`; the scripts refuse those dates without `--allow-holdout`).
2. **AAPL beyond one day** needs a Databento purchase (ask first).
3. **Event-time features in the engine**, and a sub-second engine to use them.
4. **Queue-aware simulation** on MBO data, to replace the conservative replay fills.
5. **HJB + Hawkes controller** (plan.md Phase 6a): only after 1–3. A recent paper attempting it (arXiv:2510.26438)
   found no closed form and its solver did not converge.

## Live demo

`make live` → <http://127.0.0.1:8765>. It streams Binance BTCUSDT, runs the step 6 models on live data, scores them
each second and records to `data/live/`. `make live-replay` runs the same page on a downloaded day, with no network. It also shows a paper P&L for trading each forecast (option 1). Offline over 6 BTCUSDT days (`make taker`):
+$20–21k gross on $10k, but about −$3–4M at the 10 bp taker fee; break-even fee 0.05–0.07 bp. Option 2 (paper market
maker) is planned in `plan.md` §8a.

## Ground rules

- Never commit raw Databento data or API keys (`data/` and `.env` are git-ignored; see `.env.example`).
- No live trading.
- Report null and negative results as clearly as positive ones; state the trial count with any comparison.
- Regenerate numbers with `make all` rather than editing outputs by hand. `docs/hawkes_calibration.md` is generated.

## Where to look

| File | What it is |
|---|---|
| `README.md` | overview, quick start, results, limitations |
| `docs/changes.md` | every fix with before/after numbers |
| `docs/hawkes_calibration.md` | generated Hawkes diagnostics report |
| `docs/alpha_signal.md` | generated OOS alpha-signal ablation (AAPL) + HJB-literature check |
| `docs/binance_validation.md` | generated cross-asset OOS ablation (BTCUSDT) + Hawkes-fit comparison |
| `updates.md` | short summary of the improvements |
| `plan.md` | the original multi-phase research plan (only partly adopted) |
| `Report.md` | the original report; its numbers are superseded |
| `plots/07_hawkes_vs_baseline_ci.png` | the headline results chart |
