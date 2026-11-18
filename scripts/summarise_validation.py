"""Summarise output/validation/<SYMBOL>/ (scripts/validate_live_strategies.py) and choose each pair's execution model.

For every pair and strategy: the mean over test days of the family's metric, the number of test days on which it
beat the family's control, and (execution) a paired test across orders against the control; how often each model's
variant beat the Hawkes ensemble's; and each event-driven execution book against its once-a-second twin on the same
orders. The Hawkes model with the better timing + passive execution per pair (ties within 0.02 bp go to the ensemble)
is written into that pair's live model file as display.exec_model, which the dashboard features (★).

Writes output/validation/summary.csv, vs_ensemble.csv, fast_vs_slow.csv and exec_model_choice.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("output/validation")
MODEL_FILES = {"BTCUSDT": Path("output/step6_binance_validation/live_model.json")}   # others: output/pairs/<SYMBOL>/
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


def fast_vs_slow(orders: pd.DataFrame) -> pd.DataFrame:
    """Each event-driven execution book against its twin on identical orders, pooled over pairs."""
    rows = []
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
        rows.append({"book": a, "twin": b, "share_differ": len(nz) / len(d), "mean_when_differ_bp": nz.mean(),
                     "mean_bp": d.mean(), "p_orders": stats.ttest_1samp(d, 0).pvalue, "pairs_better": better,
                     "n_pairs": len(ds)})
    return pd.DataFrame(rows)


def choose_exec_models(s: pd.DataFrame) -> dict[str, str]:
    """Per pair, the Hawkes model whose timing + passive execution saved more; written into the live model files."""
    choice = {}
    for pair, g in s[s.family == "execution"].groupby("pair"):
        m = g.set_index("strategy")["mean"]
        score = {k: m[f"timing_{k}"] + m[f"passive_{k}"] for k in ("with_hawkes", "hawkes_ensemble")}
        best = max(score, key=score.get)
        choice[pair] = "hawkes_ensemble" if abs(score["with_hawkes"] - score["hawkes_ensemble"]) < 0.02 else best
        path = MODEL_FILES.get(pair, Path("output/pairs") / pair / "live_model.json")
        if path.exists():
            model = json.loads(path.read_text())
            model["display"] = {"exec_model": choice[pair], "source": "output/validation/exec_model_choice.json"}
            path.write_text(json.dumps(model, indent=2))
    return choice


def main():
    days, orders = load()
    s = summary(days, orders)
    s.to_csv(ROOT / "summary.csv", index=False)
    vs_ensemble(days).to_csv(ROOT / "vs_ensemble.csv", index=False)
    fvs = fast_vs_slow(orders)
    fvs.to_csv(ROOT / "fast_vs_slow.csv", index=False)
    choice = choose_exec_models(s)
    (ROOT / "exec_model_choice.json").write_text(json.dumps(choice, indent=2))
    pd.set_option("display.width", 220)
    print(fvs.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("execution model per pair:", choice)


if __name__ == "__main__":
    main()
