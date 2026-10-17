# Updates

This version keeps the original repo's structure and applies the fixes and improvements suggested in `plan.md`.
`make all` regenerates everything, and two clean runs gave identical numbers. All 25 tests pass (`make test`), and a
copy of the CI job passed in an Ubuntu 24.04 container. The work is on the local branch `repo-improvements`; nothing
is committed or pushed yet.

Full details with before/after numbers: [`docs/changes.md`](docs/changes.md).

## Engine bugs fixed

Flags and output columns are unchanged, so the analysis and plot scripts still work.

- **Markout sign was inverted.** The reported "−0.25 markout" actually meant fills were profitable.
- **"Adverse selection cost" measured gains.** That is why it equalled the realized spread. It now measures how far
  the mid moved against the position.
- **Toxicity logic protected the wrong side.** Buy pressure now widens or pulls the ask instead of the bid.
- **Inventory risk was effectively zero** because of a units mismatch: the term was at most about $0.000005.
  Volatility is now in dollars, and a 60-second risk horizon is configurable with `--risk-horizon-s`. Max inventory
  went from 500 (the hard limit) to 175.

## Trade-direction labels

The committed labels called only 53% of the exchange-flagged buy trades buys, close to a coin flip. Two causes:

- the labelling used quotes up to 1 second after each trade;
- the exchange's own buy/sell flag was being discarded.

The fixed version uses the exchange flag where it exists and past quotes otherwise. The raw Databento files are not
in the repo, so the step 2 outputs were regenerated from the committed trade file (caveat in `docs/changes.md`).

## New analysis

- **Fill model calibrated from trades** (`scripts/calibrate_fill_model.py`). The data give k ≈ 58 per dollar,
  against the 4–8 the sweep assumed. The old setup overstates fills about 7,000× at the distance where it quotes
  (about 16 ticks from mid). So most of the old ~$5–11k/day PnL comes from the fill model. With calibrated fills it
  is about $200–260/day.
- **Hawkes diagnostics report** ([`docs/hawkes_calibration.md`](docs/hawkes_calibration.md)). It covers bucket
  occupancy, comparison against simpler models, stability, refits over bucket widths and quarters of the day, and an
  exact event-time fit with a goodness-of-fit test. Findings:
  - clustering is very strong;
  - fitting the whole day overstates self-excitation, because the trade rate changes through the day;
  - the 1-second model misses millisecond-scale bursts.
- **Step 4 now runs 10 seeds per configuration.** It compares Hawkes and baseline on the same seeds with 95%
  confidence intervals, reports the trial count (240 runs), and includes a run with the order-flow signal switched off.

## Headline result

**Inconclusive about Hawkes as a signal, negative about the current signal logic.**

| Configuration | Hawkes − baseline PnL (95% CI) |
|---|---:|
| Best grid config (A=0.12, k=4) | +910 [841, 980] |
| Calibrated fill model (A=0.066, k=58) | +46 [23, 70] |
| Default config, order-flow signal unused in quoting | +0.2 [−51, 51] |
| Calibrated, order-flow signal unused in quoting | −19 [−40, 2] |

Hawkes beats baseline in every seed, but the gap disappears once the order-flow signal is unused in quoting. Using
the signal to skew, widen and pull quotes loses money in both modes. The smoother Hawkes signal triggers that logic
less, so it loses less. That is not evidence that Hawkes predicts order flow better; that needs an out-of-sample
prediction test.

All results are one day (AAPL, 2024-06-03), in-sample, with simulated fills. `Report.md` now carries a note that its
numbers are superseded.

## Housekeeping

- **Environment:** versions are pinned in `pyproject.toml` / `uv.lock`, because newer pandas and matplotlib break the
  original scripts. Added a `Makefile` and a GitHub Actions workflow.
- **Untracked artifacts:** the build output (`bin/`) and the large per-run result files are no longer tracked.
- **Benchmark:** runs without GNU `time`, reporting peak memory as n/a.
- **Size:** the directory is about 534 MB, mostly gitignored per-run result files that the plots read.
- **`plan.md`:** left untracked.
