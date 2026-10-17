# plan.md — Hawkes Market Maker → multi-strategy research and trading platform

> Audience: Claude Code, working inside a clone of `Abhi-1122/Hawkes-Market-Maker`.
> Written from a planning conversation, **not** from a fresh read of the code. Phase 0 exists to verify every assumption in this file against the actual repo before anything is changed. If the repo disagrees with this plan, the repo wins. Record the discrepancy in `docs/decision_log.md` and carry on.

---

## 0. Ground rules (read first, apply throughout)

1. **Honesty over impressive numbers.** Never tune until a number looks good and then report it. Report negative and null results. Every performance claim must name its data, period, seeds, number of configurations tried, and whether it is in-sample or out-of-sample.
2. **No live trading.** Nothing in this plan places real orders. The furthest stage is paper trading against a sandbox or testnet. Real-money trading needs an explicit, separate, written decision from the user. Do not add code paths that can send real orders without an off-by-default, clearly named safety flag.
3. **The C++ engine is to be re-architected, not preserved as-is.** The user wants it restructured around the new goals (pluggable strategies, venue abstraction, a separated risk layer, event-driven inputs) while keeping its functionality roughly the same. So: (a) freeze the current engine's behavior as a golden reference in Phase 0, since it is the specification of "same functionality"; (b) rebuild it in Phase 2b with parity tests against that reference; (c) record every intentional behavior difference in `docs/decision_log.md` with a reason and before/after metrics. Until the new engine passes parity, the old code stays buildable under `legacy/` so comparisons are always possible. The Python `step1`–`step4` scripts keep working throughout, and `step4` must keep reading the engine's results (see the output-compatibility requirement in Phase 2b).
4. **Small, reviewable steps.** One phase at a time. Each phase ends with its acceptance criteria met, tests passing, and a short entry in `docs/decision_log.md`. Stop and ask when a decision is genuinely the user's (data purchases, venue choice, any real-money step).
5. **Reproducibility.** Pin dependencies, seed every RNG, record data checksums and git SHA with each result file.
6. **Data hygiene.** Raw market data (Databento CSVs etc.) is licensed. Never commit it. Keep it gitignored and document how to obtain it. Never commit API keys, put them in env vars or a gitignored `.env`.
7. **Look-ahead and leakage are the main failure modes.** Every feature must be computable using only information available at its timestamp. Add explicit tests for this (see Phase 2).
8. **Context for the user.** The user is doing a coursework project with three reviews (see section 3). Review 1 (model and literature) is done. Review 2 needs substantial progress. Review 3 is light and converging. The user's own role is Role 3 (Hawkes calibration), which is Phase 1. Scope is governed by the checkpoint tiers in section 3, so check the tier before starting any task.

---

## 1. Background you need

### 1.1 What the repo does today (verify in Phase 0)

Offline, single-instrument, CSV-in/CSV-out research pipeline on one Databento MBP-10 sample day (AAPL, 2024-06-03):

| Stage | Script / binary | Role |
|---|---|---|
| 1 | `scripts/step1_build_feature_table.py` | Midprice, spread, microprice, L1/L10 imbalance, rolling realized vol (`rv_60`, `rv_300`) on a 1-second grid |
| 2 | `scripts/step2_trade_label_and_hawkes.py` | Lee–Ready trade labeling, 1s buy/sell counts, EWMA intensities, bivariate discrete Hawkes fit, `hawkes_ofi` |
| 3 | `engine/event_driven_mm_engine.cpp` → `bin/step3_mm` | Event-driven replay: Avellaneda–Stoikov-style quotes plus OFI/microprice skew, toxicity widening/pulling, queue and toxicity haircuts on a parametric fill model, inventory limit, PnL/markout/latency accounting |
| 4 | `scripts/step4_analysis.py`, `scripts/plot_graphs.py` | Parameter sweeps, best-run tables, PnL decomposition, markout, regime analysis |

Reported baseline (from README, to be reproduced, not trusted): Hawkes final MTM PnL about 11,592 vs baseline about 10,723; about 105k events/sec; p99 engine latency about 0.1 µs; best-run parameters `gamma=0.10, k=6.0, A=0.08`.

### 1.2 Key math (used by the code, keep consistent)

Discrete bivariate Hawkes recursion with a **shared** decay `beta` (7 free parameters: `mu_b, mu_s, alpha_bb, alpha_bs, alpha_sb, alpha_ss, beta`):

```
lambda_b[t] = mu_b + beta * lambda_b[t-1] + alpha_bb * N_b[t-1] + alpha_bs * N_s[t-1]
lambda_s[t] = mu_s + beta * lambda_s[t-1] + alpha_ss * N_s[t-1] + alpha_sb * N_b[t-1]
```

Fit is a **Poisson quasi-likelihood** on 1-second bucket counts, minimized with `scipy.optimize.minimize`, with a stability penalty as `beta` approaches 1 or goes negative. It is a quasi-MLE (approximate), not the exact continuous-time point-process MLE. Document it as such everywhere.

Quoting core (Avellaneda–Stoikov, extended):

```
r        = m - q * gamma * sigma^2 * tau + c_ofi * OFI + c_micro * (microprice - m)
half_spr = max( (1/gamma) * ln(1 + gamma/k) + 0.5 * gamma * sigma^2 * tau , min_spread_ticks * tick / 2 )
fill_p   = (1 - exp(-A * pressure * exp(-k * delta) * dt)) * queue_haircut * toxicity_haircut
```

`tau` is session-fraction-remaining, a substitution for time-to-horizon. `pressure` is the opposite side's intensity.

### 1.3 Honest assessment that motivates this plan

- Fills are **simulated** from an assumed parametric model, and the Hawkes signal also enters that model, so the backtest is partly circular.
- Granularity is 1 second. Real market making is decided at microsecond scale by queue position and latency.
- Evidence is one day, one instrument, best-run selection. The roughly 8% PnL gap between Hawkes and baseline is not shown to be significant.
- In liquid US equities a small team is structurally disadvantaged against Citadel/Virtu-class firms (latency, queue priority, fees, capital). The project's value is as a rigorous research platform and learning vehicle, not a proven money-maker.
- Hawkes is a valid **descriptive** model of order-flow clustering and a good **feature generator**. It is not, by itself, a strategy.
- A multi-strategy system is the right end state, but validate one strategy at a time because every extra strategy and sweep adds multiple-testing risk.

---

## 2. Target architecture

Strategies are plug-ins. They emit signals or target positions. One shared layer handles portfolio construction, risk and execution.

```
                +--------------------+
 data sources → | Market data layer  | (historical replay, later live/paper)
                +----------+---------+
                           v
                +--------------------+     +------------------+
                | Feature / signal   | →→→ | Persistence &    |
                | library (streaming)|     | experiment store |
                +----------+---------+     +------------------+
                           v
        +------------+  +------------+  +-------------+  +------------+
        | MM plugin  |  | StatArb    |  | Trend       |  | Execution  |
        | (Hawkes)   |  | plugin     |  | plugin      |  | algos      |
        +-----+------+  +-----+------+  +------+------+  +-----+------+
              +---------------+---------------+---------------+
                              v
                +----------------------------+
                | Portfolio + risk layer     |  limits, netting, per-strategy
                | (soft costs + hard limits) |  risk budgets, kill switch
                +-------------+--------------+
                              v
                +----------------------------+
                | OMS / execution layer      |  sim venue now; paper venue later
                +----------------------------+
```

Design principles:
- **Soft + hard risk.** Risk aversion prices inventory into quotes (soft). Hard limits override pricing (circuit breaker). Keep both.
- **One evaluation path.** The same strategy code runs in backtest and paper modes. Only the venue adapter changes.
- **Separate risk budgets per strategy.** A market maker wants flat inventory, stat arb deliberately holds a spread. They must not fight over one budget.
- **Language split.** Python for research, calibration, analytics. C++20 for the event-driven core, restructured as a library with a thin CLI (Phase 2b). Expose the same core to Python (pybind11 recommended) so research code and backtests run the *same* strategy code. Changing the core's language is a user decision (see Open Questions).

---

## 3. Review checkpoints (scope by review, not by calendar)

The project is judged in three reviews. **Review 1 is done** (the model and the literature were presented; nothing in the repo has changed yet). The user is now working toward **Review 2**, which must show substantial progress. **Review 3** must be light and converging: close gaps, absorb feedback, freeze and finalize. No new major components.

The dates of Reviews 2 and 3 are unknown (see Open Questions), so this section schedules by scope tiers and cut lines instead of calendar dates. Each phase heading in section 4 carries its checkpoint tag.

| Checkpoint | Window | Intent | Phases |
|---|---|---|---|
| **CP1 — Review 1** | done | Understand the model and the literature | none (baseline) |
| **CP2 — Review 2** | now → Review 2 | The heavy checkpoint: from a one-day demo to a verified, rigorous research platform | 0, 1, 2, 2b, 3, partial 9 (stretch: 4 prototype, 6a) |
| **CP3 — Review 3** | Review 2 → Review 3 | Converge: fix, freeze, finalize | final 9, leftovers from CP2, feedback fixes |
| Post-review | unscheduled | Direction only, nothing promised to any review | 4 (full), 5, 6b to 6d, 7, 8 |

Planning assumption: roughly 80% of the remaining effort goes to CP2 and 20% to CP3. Adjust once real dates are known.

### CP1 — Review 1 (done)

Delivered: model explanation, literature (Hawkes, Avellaneda–Stoikov, order-flow imbalance, stat-arb and other alternatives), the honest assessment of what the project does and does not show, the Obsidian study note (Parts I to III), and this plan. No reproduction or code changes have been done yet, so Phase 0 is the first task of CP2.

### CP2 — Review 1 → Review 2 (heavy)

Theme: **verify what exists, make it rigorous, rebuild the engine with parity, and get a first honest answer on whether Hawkes adds value.**

Work streams and dependencies:
- **Stream A (Python, independent of the engine):** Phase 1 (Hawkes calibration rigor) → Phase 2 (evaluation harness) → Phase 3 (alpha signal and Hawkes ablation).
- **Stream B (engine):** Phase 0 (golden master and `legacy/` oracle) → Phase 2b (re-architecture with parity).
- Phase 0 comes first for both streams. After it, A and B can run in parallel if the work is split across people. Phase 1 is the user's Role 3. The streams meet when the Phase 2 harness drives both the legacy and new engines.

Scope tiers (cut from the bottom if behind):

| Tier | Items |
|---|---|
| **MUST** (Review 2 is not credible without these) | Phase 0 in full. Phase 1 items 1 to 7, 10 and 11 (reference implementation, occupancy, null model, stability, sub-window refits, time-rescaling fit, exports, plots). Phase 2 core: multi-seed runs, trial registry, leakage tests, metrics module, walk-forward protocol on whatever data exists. Phase 2b with parity levels 1 and 2 passing. The Hawkes-vs-baseline comparison re-run under the harness with a plain conclusion, whichever way it comes out. Phase 3 minimal: a linear OFI model plus the with/without-Hawkes ablation. |
| **SHOULD** | Phase 1 items 8 and 9 (kernel alternatives, parameter uncertainty). Overfitting statistics (probability of backtest overfitting, deflated Sharpe). Parity level 3 complete and the Python bindings. A gradient-boosted model in Phase 3. More days or instruments of data if obtainable. |
| **STRETCH** | A queue-replay prototype (Phase 4) only if L3/MBO data is available. A first alpha-aware market-maker prototype (Phase 6a). |

Never cut, even when behind: leakage tests, the trial registry, parity tests, and honest reporting of null or negative results. Prefer finishing fewer items completely over many partial ones.

**Review 2 package (exit criteria):**
1. `make all` reproduces the pipeline from a clean checkout. CI is green.
2. The Hawkes calibration report with its figures: occupancy, null-model comparison, stability, sub-window parameters, Q–Q plot.
3. The Hawkes-vs-baseline result under the harness: mean and confidence interval over seeds, trial count, plain-language conclusion (positive, negative or inconclusive).
4. The new engine's parity report (levels passed, documented exceptions) and a throughput/latency comparison against legacy.
5. The alpha-signal ablation result (preliminary is acceptable if labeled as such).
6. An updated limitations list, and a short "what changed since Review 1 and why" note.
7. **A final holdout window set aside and recorded now, untouched until CP3** (see below).

### CP3 — Review 2 → Review 3 (light, converging)

Theme: **fix, freeze, finalize.**

Allowed work:
1. Triage every Review 2 comment into *fix now*, *document as a limitation*, or *defer*. Keep a feedback log mapping each comment to its resolution.
2. Finish small leftovers from CP2's SHOULD tier, only if they are small.
3. Freeze code and configurations and tag a release (for example `v1.0-review3`). No tuning after this point.
4. Run the final held-out evaluation **exactly once** with the frozen configuration and report it as is. If only one day of data exists, hold out the last fraction of the session and state clearly that this is weak evidence.
5. Finalize documentation: calibration report, evaluation methodology, engine spec, Obsidian Part IV, and a limitations section that says plainly which claims are established and which remain hypotheses.
6. Harden the demo: one-command reproduction, a README quick start, cached results so the demo does not depend on long runs, and all final figures regenerated by script.

Not allowed in CP3 without explicit user approval: new strategies, new venues, new data sources, architecture changes. Anything from the post-review list stays as documented direction.

**Review 3 package (exit criteria):** tagged release with one-command reproduction; the single held-out evaluation; the feedback log; the final report; parity level 3 complete or documented; every figure regenerable from script.

### Risks to the checkpoints

| Risk | Mitigation |
|---|---|
| Only one sample day of data | Say so everywhere. Treat results as demonstration, not evidence. Ask the user before any paid data purchase. |
| The legacy RNG sequence cannot be extracted | Fall back to parity levels 1 and 3, and record why. |
| The engine rebuild overruns | Ship parity level 1 first (pure quoting functions), then deterministic replay, then statistical parity. Keep `legacy/` working. |
| The Hawkes fit does not converge or is unstable | Report it. Try multiple initial conditions and bucket widths. Instability across windows is a finding, not a failure. |
| Hawkes shows no out-of-sample benefit | That is a valid result for the panel. Report it with the ablation. |

---

## 4. Phases

Each phase lists **Goal**, **Tasks**, and **Acceptance criteria**, plus a checkpoint tag from section 3. Do them in order within a checkpoint. Check with the user before starting any phase tagged post-review, and before any CP3 work that adds a new component.

### Phase 0 — Ground truth and reproducibility  `[CP2 · MUST]`

**Goal:** Know exactly what exists, and prove it runs.

Tasks:
- Read the whole repo (README, all scripts, the C++ engine, output dirs). Write `docs/repo_map.md`: modules, inputs/outputs, column names, CLI flags, `MMParams` table.
- Check every claim in section 1 against the code. Log differences in `docs/decision_log.md`.
- Create a reproducible environment (`requirements.txt` or `pyproject.toml` with pinned versions, a `Makefile` or `justfile` with `make features | hawkes | engine | analysis | all | test`).
- Run the pipeline end to end on the sample data. Compare to the README numbers. Record any gap and its cause.
- **Write `docs/engine_spec.md`:** a precise description of the legacy engine's behavior, extracted from the code: every formula, clamp, default, edge case (missing or non-positive volatility, zero depth, inventory at limit, one-sided pull), the order of operations within a tick, and the RNG usage (generator type, seeding, how many draws per tick and in what order). This is the parity specification for Phase 2b.
- **Capture a golden master.** Build the legacy engine unchanged, run it on the sample data over a list of seeds and a small parameter grid, and store the outputs in `tests/golden/` (keep it small: summaries plus checksums and a regeneration script if the full files are large).
- **Build a debug-instrumented copy** of the legacy engine under `legacy/` that additionally dumps per-tick intermediates (reservation price, half-spread, final quotes, fill probabilities, the uniform draws used, inventory, cash). Only add dumps, never change behavior. Verify that it reproduces the un-instrumented outputs exactly. Phase 2b uses it as a test oracle.
- Add smoke tests: feature table has expected columns and no NaN in required fields, Hawkes fit returns stable parameters, engine exits 0 and produces `engine_results.csv`.
- Add CI (GitHub Actions) running lint, the smoke tests, and a C++ build.

Acceptance:
- `make all` reproduces README results within a stated tolerance, or the discrepancy is explained.
- `docs/repo_map.md` and `docs/decision_log.md` exist.
- CI green.

### Phase 1 — Hawkes calibration rigor (serves the user's Role 3 directly)  `[CP2 · items 1-7, 10, 11 MUST; 8-9 SHOULD]`

**Goal:** Turn "we fit a Hawkes model" into a validated, defensible calibration.

Tasks (implement as a reusable `hawkes/` Python package plus a notebook or report, not inline script edits):
1. **Clean reference implementation.** `fit_discrete_hawkes(counts_b, counts_s, ...)` returning parameters, log-likelihood, convergence info, and the intensity paths. Keep it numerically identical to `step2` (assert this in a test), and vectorize or `numba`-compile the recursion if useful.
2. **Bucket occupancy check.** For buy and sell counts report the share of buckets with 0, 1, 2, 3+ events. This justifies the quasi-likelihood. Flag honestly if multi-event buckets are common.
3. **Bucket-width robustness.** Refit at multiple widths (for example 100 ms, 250 ms, 500 ms, 1 s, 2 s). Report how parameters and branching ratios move. Rescale parameters sensibly so widths are comparable.
4. **Null model comparison.** Fit homogeneous Poisson (closed form). Report log-likelihood gain, AIC/BIC, and a likelihood-ratio statistic (diagnostic only, since asymptotics are imperfect). Compare against a no-cross-excitation variant to show cross-excitation earns its keep.
5. **Stability analysis.** Report branching ratios (`alpha_xx / beta`, and the spectral radius of the effective excitation matrix). Confirm below 1. Document that the penalty approximates this constraint.
6. **Sub-window stability.** Refit on rolling or disjoint windows (for example 4 equal blocks). Report parameter ranges. Note open and close effects.
7. **Time-rescaling goodness of fit (Ogata).** Needs event-level timestamps, so go back to raw trades. Implement the continuous-time exponential-kernel Hawkes likelihood and compensator, compute rescaled inter-event times, and produce a Q–Q plot against Exponential(1) plus a KS test. Compare the discrete quasi-MLE parameters against an exact continuous-time MLE fit on the same data.
8. **Kernel alternatives (optional, flagged).** Add a power-law or sum-of-exponentials kernel and compare fit. Motivation: the literature points to slowly decaying kernels and near-critical branching ratios in order flow.
9. **Parameter uncertainty.** Bootstrap or observed-information standard errors for the 7 parameters.
10. **Export contract for downstream use.** Write `output/step2_hawkes/` with fitted parameters (JSON), intensity series, and `hawkes_ofi`, plus a short schema doc. The visualization role consumes `lambda_*_hawkes`, `hawkes_ofi`, the EWMA OFI, and raw event counts. Do not rename existing columns.
11. **Plots for the review.** (a) zoomed raw trade bursts vs fitted intensity, (b) Hawkes OFI vs EWMA OFI, (c) Q–Q plot, (d) occupancy bars, (e) parameter stability across windows.

Acceptance:
- A report (`docs/hawkes_calibration.md`) states the model, objective, optimizer and initial conditions, results, and every limitation. It explicitly labels the method a **quasi-MLE** and the **shared-beta** simplification.
- All checks above have code, tests, and recorded outputs.
- Existing `step2` outputs are unchanged unless the user approves a change.

### Phase 2 — Evaluation harness (credibility upgrade)  `[CP2 · core MUST; overfitting statistics SHOULD]`

**Goal:** Make every later claim statistically honest.

Tasks:
- **Multi-day, multi-instrument data ingestion.** Parameterize the pipeline by date and symbol. Add a data loader abstraction. Ask the user before acquiring paid data, and document the exact Databento request needed.
- **Walk-forward protocol.** Calibrate Hawkes and tune strategy parameters on a training window, evaluate on the next window, roll forward. Never tune on the evaluation window.
- **Multiple seeds.** The fill simulation is stochastic. Run N seeds per configuration and report mean, standard deviation, and confidence intervals.
- **Trial registry.** Log every configuration ever evaluated (parameters, data window, seed, metrics) to `experiments/registry.parquet` or SQLite. The count of trials must be available to any report.
- **Overfitting controls.** Implement the probability of backtest overfitting and the deflated Sharpe ratio (Bailey, López de Prado). Report them alongside Sharpe. Show best-run vs median-run, not just best.
- **Leakage tests.** Unit tests that shuffle or truncate the future and assert features at time t do not change. Test the Hawkes recursion and every rolling feature.
- **Standard metrics module.** PnL decomposition (spread capture, adverse selection, fees), markout curves at several horizons, inventory distribution, turnover, max drawdown, Sharpe with confidence interval.

Acceptance:
- One command produces a multi-day walk-forward report with confidence intervals and trial counts.
- The Hawkes-vs-baseline comparison is re-run under this harness and the conclusion (significant or not) is stated plainly, whichever way it comes out.

### Phase 2b — Engine re-architecture (behavior-preserving)  `[CP2 · parity levels 1-2 MUST; level 3 and bindings SHOULD]`

**Goal:** Replace the monolithic single-file replay engine with a modular core that supports the roadmap (several strategies, several venues, event-driven inputs, a separated risk layer) while reproducing the old engine's functionality. The evaluation harness from Phase 2 must call the engine through a stable interface (CLI or bindings), so it can run against `legacy/` and the new core interchangeably.

**What "same functionality" means.** Take the exact semantics from `docs/engine_spec.md`. This list is the checklist to verify:
- Quote generation: Avellaneda–Stoikov reservation price and half-spread, the OFI and microprice skew terms, session-fraction `tau`, `rv_60` volatility with its floor, and the Hawkes-vs-EWMA signal switch (`use_hawkes`).
- Toxic-regime handling: widening on the threatened side and the optional one-sided pull.
- Tick rounding and the no-cross clamp against the current best bid and ask.
- The hard inventory-limit check.
- The fill model: opposite-side pressure as intensity, queue haircut, toxicity haircut, Bernoulli draw.
- Accounting: cash, inventory, mark-to-market PnL, maker fee, per-fill markout at the configurable horizon, adverse-selection cost, spread capture, per-tick latency measurement and the budget flag.
- Interfaces: the same `MMParams` names and defaults, the same CLI flags, the same feature-table input, and the same `engine_results.csv` columns, so `step4` runs unchanged.

**Target structure** (C++20, CMake, GoogleTest or Catch2):
- `core/`: event types (book update, trade, timer, own-order ack/fill), a deterministic event loop, an injectable clock.
- `strategy/`: an `IStrategy` interface and a `MarketMakerStrategy` containing the quoting logic extracted from the old file as **pure functions** (state in, quotes out, no I/O, no RNG).
- `venue/`: an `IVenue` interface and a `ParametricFillVenue` holding the old fill model. The queue-replay venue arrives in Phase 4.
- `risk/`: pre-trade limit checks and kill-switch hooks, separated from quoting.
- `accounting/`: PnL, markout, spread capture, adverse selection.
- `io/`: a `FeatureTableSource` that replays the 1-second feature table as one event source among future ones, and a results writer that preserves the legacy schema.
- `app/`: a thin CLI.
- Python bindings (pybind11) so research code can drive the same core. Optional but recommended.

**Seams to introduce deliberately:** a pure quoting function; an injected RNG interface so fills can be forced in tests; an injected latency timer; configuration as a validated struct instead of loose globals.

**Parity tests (three levels):**
1. **Quote-level oracle.** Compile the legacy quote function as a test oracle. For a large property-based grid of random market states, inventories and parameters, the new quote function must match it to tick precision.
2. **Deterministic replay.** Feed the new engine the exact uniform draws dumped from the instrumented legacy build. Per-tick quotes, fill decisions, inventory, cash and PnL must match to a tight floating-point tolerance. If the legacy RNG sequence cannot be extracted, say so, fall back to level 3, and record why.
3. **Statistical parity.** Over many seeds on the sample day, the distributions of final PnL, fill counts, inventory statistics and markout means from the new and legacy engines must be statistically indistinguishable. Report mean differences with confidence intervals.

Also test output-schema compatibility: `step4` runs unmodified on the new results and reproduces the same tables within tolerance.

**Change discipline.** Port first, with parity. Only then make improvements, each as a separate commit with a `docs/decision_log.md` entry and before/after metrics. Do not bundle behavior changes into the port. Likely candidates to evaluate afterwards: replacing the session-fraction `tau` with a configurable risk horizon, replacing 1-second-row replay with true event-time handling, and calibrating `A` and `k` instead of assuming them.

**Performance.** Benchmark events per second and per-tick latency for new vs legacy. Report them and explain any large regression. The legacy per-tick figure was measured on 1-second rows with a parametric fill model, so it says little about real exchange latency.

Acceptance:
- Parity levels 1 to 3 pass, or each exception is documented with its cause.
- `step4` works unchanged on new outputs.
- `legacy/` is kept until the user approves removing it.
- `docs/repo_map.md` and `docs/engine_spec.md` are updated to the new structure.

### Phase 3 — Short-horizon alpha prediction from the order book  `[CP2 · linear model + Hawkes ablation MUST; boosted model SHOULD]`

**Goal:** Build a real predictive signal that can be judged without a fill simulator, and that Hawkes features can feed.

Tasks:
- **OFI features** per Cont, Kukanov & Stoikov (2014): best-level OFI, multi-level OFI, depth-normalized. Also imbalance, microprice deviation, spread, realized vol, and Hawkes-derived features (`lambda_b`, `lambda_s`, `hawkes_ofi`, branching-ratio proxy).
- **Targets:** midprice change over 1, 2, 5, 10 seconds (and in event-time as an option). Define strictly, with no look-ahead.
- **Models, simplest first:** linear/ridge on OFI (replicate the linear price-impact relation), gradient-boosted trees, then optionally a small LSTM/MLP trained on stationary order-flow inputs (the literature suggests order-flow inputs beat raw book states).
- **Evaluation:** out-of-sample R², hit rate, information coefficient, and crucially **markout-based economic value** after a realistic cost assumption. Ablate: with vs without Hawkes features. Report whether Hawkes features add out-of-sample value. That answers "is Hawkes useful?" empirically.
- Package as `signals/` with a common `Signal` interface (inputs: feature frame, outputs: expected return and confidence per horizon).

Acceptance:
- Walk-forward results with confidence intervals, an explicit with/without-Hawkes ablation, and a plain-language conclusion.

### Phase 4 — Better simulation: queue-aware replay  `[CP2 STRETCH prototype if L3 data exists; full version post-review]`

**Goal:** Remove the circularity between the signal and the fill model.

Tasks:
- Extend the `IVenue` interface from Phase 2b if needed (submit, cancel, replace, fills, book updates), keeping `ParametricFillVenue` working so strategies stay venue-agnostic.
- Build a **queue-position-aware replay simulator** on MBO/L3 data where available (ask the user about data). Track the resting order's queue position, assume FIFO price-time priority, advance the queue on trades and cancellations, fill when the queue empties.
- Add a **latency model** (configurable order-entry, ack and market-data delays, so stale quotes can be picked off) and **fee model** (maker/taker, rebates).
- Keep the parametric fill model as an alternative `Venue`. Compare both on the same strategy. Report how much the conclusions change. That sensitivity is itself a key finding.
- Calibrate the parametric `A`, `k` against replay fill statistics instead of assuming them.

Acceptance:
- The same market-maker plug-in runs on both venues. A report shows PnL, fill rates, adverse selection and markout under each, plus a conservative interpretation.

### Phase 5 — Strategy-plugin refactor and the portfolio/risk layer  `[post-review]`

**Goal:** Reach the multi-strategy architecture in section 2 without breaking Phase 0 reproducibility.

Tasks:
- Extend the `IStrategy` interface from Phase 2b to cover `on_market_event(state) -> actions` or `-> target_positions`, plus `risk_budget` and `required_features`, and add a strategy registry so several strategies run in one session.
- Keep the Phase 2b parity tests running as a regression suite for the market-maker plugin, since it is already the first plugin.
- **Portfolio/risk layer:** per-strategy and global position and notional limits, per-instrument limits, max order rate, fat-finger checks, daily loss limit, and a **kill switch** that flattens and cancels. Keep soft (risk aversion in pricing) and hard (limits) separate.
- **Allocator:** start simple (equal risk budget, volatility-scaled). Include the diversification identity for planning: with N strategies of Sharpe S and pairwise correlation rho, portfolio Sharpe is `S*sqrt(N)/sqrt(1+(N-1)*rho)`. Estimate realized correlations from out-of-sample strategy returns and report them, since correlations rise in stress.
- Add structured logging and an experiment store so runs are comparable.

Acceptance:
- The Phase 2b parity suite still passes. A second dummy strategy can be added without touching core code. Risk limits and kill switch have tests, including adversarial ones (limit breach, rapid fills, stale data).

### Phase 6 — Additional strategies (one at a time, each validated under the Phase 2 harness)  `[6a: CP2 STRETCH prototype; 6b-6d: post-review]`

6a. **Market maker with alpha signals.** Feed the Phase 3 signal into the reservation price (Cartea–Jaimungal style): lean quotes with the expected short-term price change, cut the threatened side when adverse-selection risk is high. Compare against the heuristic `c_ofi` and `c_micro` skews. Replace ad hoc coefficients with principled ones where possible.

6b. **Statistical arbitrage (medium frequency).** Following Avellaneda & Lee: PCA or sector-ETF factor residuals, Ornstein–Uhlenbeck fit per residual (`ds = theta*(mu - s)dt + sigma*dW`, half-life `ln2/theta`), s-score entry and exit thresholds, market-neutral construction, volume-adjusted "trading time" as an option. Include transaction costs and borrow assumptions explicitly. Expect alpha decay and report subperiod performance. Optional extension: a learned signal in the spirit of deep-learning stat arb, only after the classical version is working and honestly evaluated.

6c. **Systematic trend / time-series momentum.** Lookbacks of 1 to 12 months on liquid futures or ETFs, volatility targeting. Run it **with and without** volatility scaling and report both, because published critiques say the edge may come largely from vol scaling and weakened post-2009.

6d. **Execution algorithms.** Almgren–Chriss (static schedule from risk aversion, temporary and permanent impact), then an adaptive variant. Used to execute the target positions from the other strategies and measured by implementation shortfall against arrival price.

Acceptance for each: strategy module, tests, a walk-forward report under the Phase 2 harness (confidence intervals, trial count, deflated Sharpe), and a short "would I deploy this?" paragraph written plainly, including a clear "no" where warranted.

### Phase 7 — Multi-asset market making  `[post-review]`

**Goal:** Use correlation to carry more gross inventory at the same risk.

Tasks:
- Replace scalar inventory risk `q*gamma*sigma^2*tau` with `gamma*tau*(Sigma @ q)_i` using a covariance matrix (Guéant's multi-asset extension). Estimate `Sigma` with shrinkage, and test factor-model dimensionality reduction for larger universes.
- Extend the Hawkes layer to a **multivariate cross-asset** version (flow in asset A excites intensity in asset B). Keep the parameter count manageable (low rank, sparse, or shared decay) and test stability via the spectral radius.
- Compare single-asset vs multi-asset market making on the same data under Phase 2.

Acceptance: a clear report of whether the covariance-aware inventory term and cross-asset excitation add out-of-sample value.

### Phase 8 — Live-readiness (paper trading only)  `[post-review]`

**Goal:** Build and test the plumbing a live system needs, with no real money.

Tasks:
- **Market data gateway:** normalized book and trade schema, sequencing, gap detection, reconnect logic. Start with replay of recorded streams, then a public websocket feed for a crypto testnet or paper venue. Ask the user which venue before building an adapter.
- **Streaming feature engine:** incremental versions of every feature, verified numerically against the batch versions from Phase 1/3 (tests assert equality).
- **Online Hawkes:** an O(1) per-event intensity update using the already-fitted kernel, plus a periodic recalibration job on a rolling window. Test that online and batch intensities agree.
- **OMS:** order state machine, ack/reject handling, reconciliation against venue state, idempotent retries.
- **Exchange connectivity:** testnet/paper adapter only.
- **Monitoring:** PnL, inventory, latency, reject/cancel rates, data staleness, with alert thresholds feeding the kill switch.
- **Persistence:** raw ticks, features, Hawkes state, orders and fills, so live and paper results can be replayed offline and compared to the simulator (this closes the loop on simulator realism).

Acceptance: a multi-day paper-trading run with logs, a reconciliation report (paper fills vs simulator-predicted fills), and a failure-injection test report (feed drop, venue reject storm, clock skew). **Stop here** unless the user explicitly decides otherwise.

### Phase 9 — Documentation and write-up  `[CP2 partial, kept current as work lands; CP3 final]`

Tasks:
- Keep `docs/` current: repo map, decision log, calibration report, evaluation methodology, strategy reports.
- Maintain the user's Obsidian study note: add a **Part IV** covering the honest assessment, the alternatives, the alpha-signal and multi-asset extensions, and the bibliography below. Keep Obsidian-safe LaTeX: no literal `$` currency inside math, `<br/>` for Mermaid label line breaks, and heading links that resolve.
- A final summary: what works, what doesn't, what is unproven, and what each result is conditional on.

---

## 5. Repository layout to converge on

```
.
├── plan.md
├── Makefile
├── pyproject.toml / requirements.txt
├── data/                  # gitignored raw data, with data/README.md on how to obtain
├── hawkes/                # calibration package (Phase 1)
├── features/              # batch + streaming feature library
├── signals/               # alpha predictors (Phase 3)
├── strategies/            # market_maker, stat_arb, trend, execution (Phases 5-6)
├── portfolio/             # allocator + risk + kill switch (Phase 5)
├── venues/                # sim_parametric, sim_queue_replay, paper adapters (Phases 4, 8)
├── engine/                # re-architected C++20 core (Phase 2b): core/ strategy/ venue/ risk/ accounting/ io/ app/ bindings/ tests/
├── legacy/                # original engine + instrumented copy, kept as parity oracle until approved for removal
├── scripts/               # existing step1..step4 (keep working)
├── experiments/           # trial registry + result artifacts (gitignored large files)
├── tests/
├── docs/                  # repo_map.md, decision_log.md, reports
└── .github/workflows/
```

---

## 6. Cross-cutting engineering requirements

- Tests for every module. Prefer property tests for numerical code (for example the Hawkes recursion matches a naive O(n²) reference sum).
- Determinism: fixed seeds, documented BLAS/thread settings where results are sensitive.
- Performance: profile before optimizing. Compare against the legacy engine's throughput and latency as a baseline, but do not chase those numbers at the expense of clarity or testability.
- Type hints and docstrings on public Python APIs. Keep functions small.
- Every reported result file carries metadata: git SHA, data checksums, config hash, seed list, trial count.

---

## 7. Open questions for the user (ask, don't guess)

1. Which data is available beyond the one sample day, and is there budget for more (multi-day MBP-10 or MBO/L3 for queue replay)?
2. Which asset class and venue to target for paper trading (US equities need a data and broker setup, crypto testnets are easier)? Any regulatory constraints on automated trading that apply to them?
3. Is the deliverable a coursework review (prioritize Phases 0 to 2 and the write-up) or a longer-term platform (continue through Phases 5 to 8)?
4. The default in this plan is to restructure the engine in C++20 and add Python bindings. Should the core change language instead (for example Rust), or is C++ fine? If it changes, the Phase 2b parity tests still apply unchanged.
5. Preferred experiment tracking (plain files + parquet, SQLite, MLflow)?
6. What are the dates of Review 2 and Review 3? The checkpoint tiers in section 3 are scheduled by scope, so real dates decide how much of the SHOULD tier fits.
7. What does the panel expect to see at Review 2 and Review 3 (working demo, written report, slides, a specific result)? Adjust the exit criteria to match.
8. Is the work split across teammates (the slides mention separate roles)? If so, Phase 1 is the user's, and the other phases need owners so the Stream A / Stream B split is real.

---

## 8. References (verify each before citing; read the full paper, not just the abstract)

- Avellaneda & Stoikov (2008), *High-frequency trading in a limit order book*.
- Guéant (2016), *Optimal market making*, arXiv 1605.01862. Multi-asset closed-form approximations. See also Bergault & Guéant on factor-model dimensionality reduction.
- Cartea, Jaimungal & Penalva (2015), *Algorithmic and High-Frequency Trading*, Cambridge University Press.
- Cartea, Jaimungal & Ricci, *Buy Low, Sell High: A High Frequency Trading Perspective*. Authorship and venue from memory, verify.
- Ricci (2014), PhD thesis, Univ. of Toronto, SSRN 2504061. Includes self-exciting market-order dynamics.
- *Market Making with Alpha Signals* (Oxford ORA; believed Cartea, Donnelly, Jaimungal, verify).
- Bacry, Mastromatteo & Muzy (2015), *Hawkes processes in finance*, arXiv 1502.04592.
- Bacry, Jaisson & Muzy (2016), *Estimation of slowly decreasing Hawkes kernels: application to high frequency order book modelling*.
- Hawkes (1971), *Spectra of some self-exciting and mutually exciting point processes*.
- Ogata (1988), time-rescaling goodness-of-fit for point processes.
- Cont, Kukanov & Stoikov (2014), *The Price Impact of Order Book Events*, J. Financial Econometrics 12(1), arXiv 1011.6402.
- Kolm, Turiel & Westray (2021), *Deep Order Flow Imbalance*; and Kolm & Westray, *Deep learning alpha signals from limit order books: practical insights and lessons learned*.
- Avellaneda & Lee (2010), *Statistical arbitrage in the US equities market*, Quantitative Finance 10(7).
- Guijarro-Ordonez, Pelger & Zanotti (2021), *Deep Learning Statistical Arbitrage*, arXiv 2106.04028.
- Moskowitz, Ooi & Pedersen (2012), *Time series momentum*, JFE; and Kim, Tse & Wald, *Time series momentum and volatility scaling* (critique).
- Almgren & Chriss (2000/2001), *Optimal execution of portfolio transactions*.
- Spooner, Fearnley, Savani & Koukorinis (2018), *Market Making via Reinforcement Learning*, arXiv 1804.04216.
- Bailey, Borwein, López de Prado & Zhu (2014), *Pseudo-Mathematics and Financial Charlatanism*, Notices of the AMS.
- Lee & Ready (1991), *Inferring trade direction from intraday data*.

---

## 9. Definition of done (whole project)

This is the long-term target. Reviews 2 and 3 are judged against the checkpoint exit criteria in section 3, not against this whole list.

- A reproducible pipeline with CI and documented data acquisition.
- A validated Hawkes calibration with every diagnostic in Phase 1 and an honest limitations section.
- An evaluation harness whose headline numbers come with confidence intervals, trial counts, and overfitting-adjusted statistics.
- At least one predictive signal evaluated out of sample with a Hawkes ablation.
- A re-architected engine that reproduces the legacy engine's functionality (parity levels 1 to 3), with every intentional difference logged.
- A strategy-plugin architecture with a tested risk layer and kill switch.
- Each added strategy accompanied by an honest deploy/don't-deploy verdict.
- Paper-trading infrastructure (if Phase 8 is approved) with reconciliation against the simulator.
- Documentation that states, plainly, which claims are established and which remain hypotheses.
