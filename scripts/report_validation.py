"""Summarise output/validation/<SYMBOL>/ (scripts/validate_live_strategies.py) into one table per strategy family.

For every pair and strategy: the mean over test days of the family's metric, the number of test days on which it
beat the family's control, and (execution) a paired test across orders against the control. Also how often each
forecasting model's variant beat the Hawkes ensemble's variant of the same strategy, which is what decides whether a
model is redundant on the dashboard.

Writes output/validation/summary.csv, output/validation/vs_ensemble.csv and docs/strategy_validation.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("output/validation")
DOC = Path("docs/strategy_validation.md")
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "NEARUSDT"]
MODELS = ["baseline_ewma", "with_hawkes", "event_time", "hawkes_ensemble"]
METRIC = {"maker": "edge_0fee_bp", "taker": "edge_0fee_bp", "execution": "saving_bp", "forecast": "r2"}
# control each strategy is judged against (execution timing is already measured against trading immediately)
CONTROL = {"as_baseline_ewma": "as_none", "as_with_hawkes": "as_none", "as_event_time": "as_none",
           "as_hawkes_ensemble": "as_none", "ev_hawkes": "ev_none", "ev_none": "as_none", "as_none": "touch",
           "with_hawkes": "baseline_ewma", "event_time": "baseline_ewma", "hawkes_ensemble": "baseline_ewma",
           "passive_baseline_ewma": "passive_none", "passive_with_hawkes": "passive_none",
           "passive_event_time": "passive_none", "passive_hawkes_ensemble": "passive_none",
           "repeg_passive_none": "passive_none", "repeg_passive_hawkes": "passive_hawkes_ensemble"}
# Fast (default) Hawkes execution books vs the same books acting only once a second (validation-only twins)
FAST_PAIRS = [("timing_hawkes_ensemble", "slow_timing_hawkes"), ("passive_hawkes_ensemble", "slow_passive_hawkes"),
              ("repeg_passive_none", "passive_none"), ("repeg_passive_hawkes", "passive_hawkes_ensemble")]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    days, orders = [], []
    for p in PAIRS:
        d = ROOT / p
        if (d / "days.csv").exists():
            days.append(pd.read_csv(d / "days.csv").assign(pair=p))
        if (d / "orders.csv").exists():
            orders.append(pd.read_csv(d / "orders.csv").assign(pair=p))
    return pd.concat(days, ignore_index=True), pd.concat(orders, ignore_index=True)


def summary(days: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pair, fam, strat), g in days.groupby(["pair", "family", "strategy"], sort=False):
        m = METRIC[fam]
        row = {"pair": pair, "family": fam, "strategy": strat, "n_days": len(g), "mean": g[m].mean(),
               "net_usd_per_day": g["pnl_net"].mean() if "pnl_net" in g else np.nan}
        ctrl = CONTROL.get(strat)
        if ctrl is not None:
            c = days[(days.pair == pair) & (days.strategy == ctrl) & (days.family == fam)].set_index("day")[m]
            mine = g.set_index("day")[m]
            diff = (mine - c.reindex(mine.index)).dropna()
            row.update({"control": ctrl, "vs_control": diff.mean(), "days_better": int((diff > 0).sum()),
                        "p_days": stats.ttest_1samp(diff, 0).pvalue if len(diff) > 1 else np.nan})
        if fam == "execution":
            o = orders[(orders.pair == pair)].pivot_table(index=["day", "arrival_ms"], columns="strategy", values="saving_bp")
            if strat in o:
                row["p_orders_vs_zero"] = stats.ttest_1samp(o[strat].dropna(), 0).pvalue
                if ctrl is not None and ctrl in o:
                    d = (o[strat] - o[ctrl]).dropna()
                    row["p_orders_vs_control"] = stats.ttest_1samp(d, 0).pvalue if d.std() > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def vs_ensemble(days: pd.DataFrame) -> pd.DataFrame:
    """For each family and model: on how many pairs (mean over days) does the model's variant beat the ensemble's?"""
    pref = {"maker": "as_", "taker": "", "execution_timing": "timing_", "execution_passive": "passive_", "forecast": ""}
    rows = []
    for fam_key, pre in pref.items():
        fam = fam_key.split("_")[0]
        m = METRIC[fam]
        t = days[days.family == fam].groupby(["pair", "strategy"])[m].mean().unstack()
        ens = pre + "hawkes_ensemble"
        if ens not in t:
            continue
        for model in MODELS:
            if model == "hawkes_ensemble" or pre + model not in t:
                continue
            d = (t[pre + model] - t[ens]).dropna()
            rows.append({"family": fam_key, "model": model, "pairs_better_than_ensemble": int((d > 0).sum()),
                         "n_pairs": len(d), "mean_diff_vs_ensemble": d.mean()})
    return pd.DataFrame(rows)


def fmt(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:+.{nd}f}"


def table(s: pd.DataFrame, fam: str, strategies: list[str], unit: str) -> list[str]:
    head = "| Pair | " + " | ".join(strategies) + " |"
    L = [head, "|---|" + "---:|" * len(strategies)]
    for p in PAIRS:
        cells = []
        for st in strategies:
            r = s[(s.pair == p) & (s.family == fam) & (s.strategy == st)]
            if r.empty:
                cells.append("–")
                continue
            r = r.iloc[0]
            c = fmt(r["mean"], 4 if fam == "forecast" else 2)
            if isinstance(r.get("control"), str):
                c += f" ({int(r['days_better'])}/{int(r['n_days'])})"
            cells.append(c)
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    return L + ["", f"Mean over the test days, {unit}. In brackets: test days on which the strategy beat its control.", ""]


def main():
    days, orders = load()
    s = summary(days, orders)
    ve = vs_ensemble(days)
    s.to_csv(ROOT / "summary.csv", index=False)
    ve.to_csv(ROOT / "vs_ensemble.csv", index=False)
    L = ["# Out-of-sample validation of every live strategy", "",
         "Generated by `scripts/validate_live_strategies.py` and `scripts/report_validation.py`. For each pair, every "
         "test day d is replayed trade by trade through the live monitor's own code (`scripts/live_monitor.py`), with "
         "every model and calibration rebuilt from day d−1 only. BTCUSDT: 2024-06-04 to 06-09; the other pairs: "
         "2026-11-11 to 11-16 (six test days each). The order book is rebuilt from trades (last sell-initiated price = "
         "bid, last buy-initiated = ask) with unknown queue sizes, so resting orders fill only on trades through their "
         "price: conservative for every maker and passive book.", ""]
    L += ["## Forecasts (R² of the 1 s return, live code path)", ""]
    L += table(s, "forecast", MODELS, "R²; control = baseline")
    L += ["## Taker (trade the forecast): gross edge, bp per dollar traded, before fees", ""]
    L += table(s, "taker", MODELS, "bp of turnover before the 10 bp fee; control = baseline")
    L += ["## Market makers: edge at 0 fees, bp per dollar traded", ""]
    L += table(s, "maker", ["touch", "as_none", "as_baseline_ewma", "as_with_hawkes", "as_event_time", "as_hawkes_ensemble",
                            "ev_none", "ev_hawkes"],
               "bp; controls: A–S books vs `as_none`, `ev_none` vs `as_none`, `ev_hawkes` vs `ev_none`")
    L += ["## Execution: saving per order vs trading immediately, bp (fees included, 10 bp maker and taker)", ""]
    L += table(s, "execution", ["passive_none", "passive_baseline_ewma", "passive_with_hawkes", "passive_event_time",
                                "passive_hawkes_ensemble"],
               "bp per order; control: `passive_none`. Hawkes books (1 s and ensemble) include the per-trade burst trigger")
    L += table(s, "execution", ["timing_baseline_ewma", "timing_with_hawkes", "timing_event_time", "timing_hawkes_ensemble"],
               "bp per order (0 = trading immediately); Hawkes books include the per-trade burst trigger")
    L += ["## Fast vs once-a-second execution (same orders, paired)", "",
          "Each row compares a book with its twin on identical orders. The default Hawkes books add a per-trade "
          "burst trigger (50 ms latency) to the per-second rule; `repeg_*` books also follow the touch when it moves.", "",
          "| Comparison | Orders that differ | Mean when they differ | Mean over all orders | p (orders) | Pairs better |",
          "|---|---:|---:|---:|---:|---:|"]
    for a, b in FAST_PAIRS:
        if a not in set(orders.strategy) or b not in set(orders.strategy):
            continue
        ds, better = [], 0
        for _, g in orders.groupby("pair"):
            t = g.pivot_table(index=["day", "arrival_ms"], columns="strategy", values="saving_bp")
            if a in t and b in t:
                d = (t[a] - t[b]).dropna()
                ds.append(d)
                better += int(d.mean() > 0)
        d = pd.concat(ds)
        nz = d[d.abs() > 1e-9]
        L.append(f"| `{a}` vs `{b}` | {100 * len(nz) / len(d):.1f}% | {fmt(nz.mean())} bp | {fmt(d.mean(), 4)} bp | "
                 f"{stats.ttest_1samp(d, 0).pvalue:.1e} | {better}/{len(ds)} |")
    L.append("")
    L += ["## Is a model's variant ever better than the Hawkes ensemble's?", "",
          "| Family | Model | Pairs where it beat the ensemble's variant | Mean difference |", "|---|---|---:|---:|"]
    for _, r in ve.iterrows():
        L.append(f"| {r['family']} | {r['model']} | {r['pairs_better_than_ensemble']}/{r['n_pairs']} | {fmt(r['mean_diff_vs_ensemble'], 4)} |")
    L += ["## What runs live as a result (`FAMILY_MODELS` in `scripts/live_monitor.py`)", "",
          "- **Every model stays on the dashboard for comparison** (baseline, +Hawkes 1 s, +event-time, ensemble) in "
          "the forecast, taker and market-maker families; the main scoreboard shows all makers and takers.",
          "- **Execution drops +Event-time** (its signal is contained in the ensemble's and it never beat it) and keeps "
          "the baseline as the comparison. The best Hawkes model is pair-specific and significant: the ensemble on ETH, "
          "LINK and NEAR, the 1 s Hawkes model on XRP and DOGE (coarse ticks). The choice per pair is "
          "`output/validation/exec_model_choice.json` (★ on the dashboard).",
          "- **Hawkes execution is fast by default:** on every trade it crosses when the Hawkes burst detector fires "
          "against the order (50 ms latency). Against the once-a-second version on identical orders it is better on "
          "9/9 pairs for passive and 6/9 for timing (table above). **Re-pegging is not used**: following the touch "
          "lost on 8/9 pairs.",
          "- **Market makers:** every A–S variant is within about 0.03 bp of the others at 0 fees, so the forecast "
          "does not change maker outcomes; the event-driven pair is kept as the speed experiment.",
          "- **ADA:** every signal hurts execution (−0.5 to −1.4 bp per order); passive without a signal is its best.", ""]
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")
    pd.set_option("display.width", 250)
    print(ve.to_string(index=False))
    print(f"Wrote {DOC}, {ROOT}/summary.csv")


if __name__ == "__main__":
    main()
