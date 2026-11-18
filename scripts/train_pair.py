"""Train and validate the forecasting models for one Binance pair, and write its live-monitor model file.

The same cross-day test as steps 6-7 on BTCUSDT, for any symbol: on each pair of consecutive days, every
parameter (the 1 s discrete Hawkes fit, the exact 3-exponential Hawkes MLE, the ridge coefficients and the
ensemble's ridge penalty) is fitted on day d-1 and applied unchanged to day d. Five models are compared
against the baseline on the 1/5/10 s return and the 1 s order-flow targets: +Hawkes (1 s), +event-time
kernels, +exact Hawkes and the Hawkes ensemble.

Days are processed two at a time, so memory stays at two days of trades whatever the pair.

Writes output/pairs/<SYMBOL>/:
  live_model.json        models trained on the last day + maker calibration + tick (for live_monitor.py)
  cross_day_by_day.csv   one row per test day, target and model
  cross_day_summary.csv  days better, p-value across days, hit-rate gain, R^2
  taker_pnl_by_day.csv   1 s sign-taker P&L of each model per test day (gross and after fees)
  exact_hawkes_fits.json one exact fit per training day
  validation.json        compact summary the live hub shows next to the pair

  uv run python scripts/download_binance.py --symbol ETHUSDT --date 2026-11-10   # ... each date
  uv run python scripts/train_pair.py --symbol ETHUSDT --dates 2026-11-10 ... 2026-11-16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pair_model as bp  # noqa: E402
import download_binance as dl  # noqa: E402
import hawkes_diagnostics as hd  # noqa: E402
import step5_alpha_signal as s5  # noqa: E402
import step6_binance_validation as s6  # noqa: E402
import step7_hawkes_ensemble as s7  # noqa: E402
import taker_backtest as tb  # noqa: E402

OUT_ROOT = Path("output/pairs")


def _json(o):
    return o.tolist() if hasattr(o, "tolist") else float(o)


def load_day(sym: str, d: str):
    t0 = time.time()
    tr = s6.load_trades(dl.OUTDIR / f"{sym}-aggTrades-{d}.csv")
    table = s6.build_feature_table(tr)
    print(f"[{sym} {d}] {len(tr):,} trades, features {time.time() - t0:.0f}s", flush=True)
    return tr, table


def evaluate(p_df, n_df, hp, prev, nxt):
    """All models trained on p_df (day d-1), tested on n_df (day d)."""
    rows, pnl_rows, alphas = [], [], []
    thr = s6.bounce_threshold(n_df)
    for target in s6.TARGETS:
        preds, y_ref = {}, None
        for name, feats in s7.feature_sets(s6.BASELINE_FEATURES).items():
            h = hp if any(c in feats for c in s7.HAWKES_COLS) else None
            x_tr, y_tr = s7.design(p_df, feats, target, h)
            x_te, y_te = s7.design(n_df, feats, target, h)
            a = s7.tune_alpha(x_tr, y_tr) if name == "hawkes_ensemble" else s5.RIDGE_ALPHA
            if name == "hawkes_ensemble":
                alphas.append({"train_day": prev, "target": target, "ridge_alpha": a})
            preds[name] = s5.ridge_fit_predict(x_tr, y_tr, x_te, a)
            y_ref = y_te if y_ref is None else y_ref
            assert len(y_te) == len(y_ref)
        y = y_ref
        moved = np.abs(y) > (thr if target in s6.PRICE_TARGETS else 0.0)
        base_se = (y - preds["baseline_ewma"]) ** 2
        for name, pr in preds.items():
            rows.append({"test_day": nxt, "target": target, "model": name,
                         "mean_se_diff_vs_base": float(np.mean(base_se - (y - pr) ** 2)),
                         "r2": 1 - float(((y - pr) ** 2).sum() / (y ** 2).sum()),
                         "hit": float((np.sign(pr[moved]) == np.sign(y[moved])).mean()) if moved.any() else np.nan})
            if target == "future_ret_1s":
                pnl_rows.append({"test_day": nxt, "model": name, **tb.strategy_pnl(pr, y)})
    return rows, pnl_rows, alphas


def summarise(cd: pd.DataFrame) -> pd.DataFrame:
    base_hit = cd[cd.model == "baseline_ewma"].set_index(["test_day", "target"])["hit"]
    out = []
    for (name, target), g in cd.groupby(["model", "target"], sort=False):
        if name == "baseline_ewma":
            continue
        d = g["mean_se_diff_vs_base"].to_numpy()
        idx = g.set_index(["test_day", "target"]).index
        gain = 100 * (g.set_index(["test_day", "target"])["hit"] - base_hit.loc[idx])
        out.append({"model": name, "target": target, "days_better": int((d > 0).sum()), "n_days": len(d),
                    "p_across_days": float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) > 1 else np.nan,
                    "hit_gain_pp": float(gain.mean()), "r2": float(g["r2"].mean())})
    return pd.DataFrame(out)


def live_models(df: pd.DataFrame, tr: pd.DataFrame, hp: np.ndarray) -> tuple[dict, dict]:
    """Ridge models on the 1 s return, trained on the whole last day (as steps 6-7 do for BTCUSDT)."""
    xfit = s7.fit_exact(tr)
    dx = s7.exact_intensity(df, tr["ts"], tr["trade_sign"].to_numpy(), xfit)
    models = {}
    for name in ("baseline_ewma", "with_hawkes", "event_time", "hawkes_ensemble"):
        feats = s7.feature_sets(s6.BASELINE_FEATURES)[name]
        x, y = s7.design(dx, feats, "future_ret_1s", hp if any(c in feats for c in s7.HAWKES_COLS) else None)
        a = s7.tune_alpha(x, y) if name == "hawkes_ensemble" else s5.RIDGE_ALPHA
        m = s5.ridge_fit(x, y, a)
        models[name] = {"features": feats, "mu": m["mu"].tolist(), "sigma": m["sigma"].tolist(),
                        "beta": m["beta"].tolist(), "intercept": m["intercept"], "ridge_alpha": a}
    exact = {"mu": xfit["mu"], "alpha": xfit["alpha"], "omega": xfit["omega"], "half_life_s": xfit["half_life_s"]}
    return models, exact


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--tick", type=float, help="tick size; fetched from Binance if omitted")
    ap.add_argument("--exact-hours", type=float, default=s7.EXACT_WINDOW_H,
                    help="hours of the training day the exact Hawkes MLE is fitted on")
    args = ap.parse_args()
    sym = args.symbol.upper()
    for d in args.dates:
        dl.check_not_holdout(sym, d, False)
    missing = [d for d in args.dates if not (dl.OUTDIR / f"{sym}-aggTrades-{d}.csv").exists()]
    if missing:
        sys.exit(f"Missing data for {sym} {missing}. Run scripts/download_binance.py first.")
    tick = args.tick or bp.tick_size(sym)
    s6.TICK = tick                       # bounce filter: 1.5 ticks of this pair
    s7.EXACT_WINDOW_H = args.exact_hours
    out = OUT_ROOT / sym
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    rows, pnl_rows, alphas, fits_x = [], [], [], {}
    prev_tr, prev_table = load_day(sym, args.dates[0])
    for prev, nxt in zip(args.dates[:-1], args.dates[1:]):
        nxt_tr, nxt_table = load_day(sym, nxt)
        p_df, n_df = prev_table[0], nxt_table[0]
        t0 = time.time()
        hp = hd.fit_discrete(p_df["buy_events"].to_numpy(float), p_df["sell_events"].to_numpy(float))["x"]
        fits_x[prev] = s7.fit_exact(prev_tr)
        p_dx = s7.exact_intensity(p_df, prev_tr["ts"], prev_tr["trade_sign"].to_numpy(), fits_x[prev])
        n_dx = s7.exact_intensity(n_df, nxt_tr["ts"], nxt_tr["trade_sign"].to_numpy(), fits_x[prev])
        print(f"[{sym} {prev} -> {nxt}] exact fit {time.time() - t0:.0f}s, half-lives "
              f"{np.round(fits_x[prev]['half_life_s'], 4)}, branching {fits_x[prev]['branching']:.3f}", flush=True)
        r, pr, al = evaluate(p_dx, n_dx, hp, prev, nxt)
        rows += r
        pnl_rows += pr
        alphas += al
        prev_tr, prev_table = nxt_tr, nxt_table

    cd = pd.DataFrame(rows)
    cds = summarise(cd)
    pnl = pd.DataFrame(pnl_rows)
    cd.to_csv(out / "cross_day_by_day.csv", index=False)
    cds.to_csv(out / "cross_day_summary.csv", index=False)
    pnl.to_csv(out / "taker_pnl_by_day.csv", index=False)
    pd.DataFrame(alphas).to_csv(out / "ridge_alpha.csv", index=False)
    (out / "exact_hawkes_fits.json").write_text(json.dumps(fits_x, indent=2, default=_json))

    # Live model: maker calibration and features from the last day, plus the forecasting models.
    last = args.dates[-1]
    model, curve = bp.build(sym, last, tick, trades=prev_tr, table=prev_table)
    hp_last = np.array([model["hawkes_params"][k] for k in hd.PARAM_NAMES])
    model["models"], model["exact_hawkes"] = live_models(prev_table[0], prev_tr, hp_last)
    model["exact_hawkes"]["fitted_on"] = f"{last}, last {s7.EXACT_WINDOW_H:g} h"
    model["validated_on"] = args.dates
    (out / "live_model.json").write_text(json.dumps(model, indent=2, default=_json))
    curve.to_csv(out / "touch_curve.csv", index=False)

    ens = cds[cds.model == "hawkes_ensemble"].set_index("target")
    ret1 = ens.loc["future_ret_1s"]
    pn = pnl.groupby("model")[["gross", "traded"]].sum()
    validation = {
        "symbol": sym, "dates": args.dates, "n_test_days": int(ret1["n_days"]), "tick": tick,
        "tick_bp": model["notes"]["tick_bp"], "median_price": model["notes"]["median_price"],
        "ensemble": {t: {"days_better": int(ens.loc[t, "days_better"]), "n_days": int(ens.loc[t, "n_days"]),
                         "p": float(ens.loc[t, "p_across_days"]), "hit_gain_pp": float(ens.loc[t, "hit_gain_pp"]),
                         "r2": float(ens.loc[t, "r2"])} for t in ens.index},
        "tests_won": int((cds["days_better"] == cds["n_days"]).sum()), "tests": len(cds),
        "ensemble_tests_won": int((ens["days_better"] == ens["n_days"]).sum()), "ensemble_tests": len(ens),
        "ensemble_beats_baseline_1s": bool(ret1["days_better"] == ret1["n_days"] and ret1["p_across_days"] < s5.BONFERRONI_P),
        "taker_gross_per_day_usd": {m: float(pn.loc[m, "gross"]) / int(ret1["n_days"]) for m in pn.index},
        "exact_half_lives_s": model["exact_hawkes"]["half_life_s"],
        "minutes": round((time.time() - t_start) / 60, 1),
    }
    (out / "validation.json").write_text(json.dumps(validation, indent=2, default=_json))
    print(cds.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(json.dumps(validation, indent=2, default=_json))


if __name__ == "__main__":
    main()
