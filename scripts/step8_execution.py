"""Step 8: Hawkes-signal execution. Does the order-flow forecast make a trade you must do cheaper?

A fund must trade $10,000 of BTCUSDT every minute (alternating buy / sell), each within a 10-second window.
Benchmark: trade immediately at arrival (taker). Strategies, one per forecasting model:
  - timing:  a taker that waits while the 1 s forecast says the price is moving its way (falling for a buy,
             rising for a sell) and crosses as soon as it is not, or at the deadline;
  - passive: rests at the best bid (buy) / ask (sell) as a maker; fills if a trade prints through that price
             (queue position unknown, so at-price trades never fill it: conservative); crosses the remainder as
             a taker as soon as the forecast turns against it, or at the deadline.
Controls: "passive, no signal" (rest, cross only at the deadline) isolates what the forecast adds.

Cost of an order = side * (fill price - arrival price) / arrival price, in bp. Savings = benchmark cost - strategy
cost. At Binance's standard tier maker and taker fees are both 10 bp, so fees cancel against the benchmark; the
table also reports the result with the maker side free (0 bp maker, 10 bp taker). Historical aggTrades carry
no order book, so bid / ask are taken as the last trade price -/+ half a tick (BTCUSDT's spread is one cent).

Models: trained on day d-1, tested on day d, as in steps 6-7 (the ensemble uses step 7's code).
Writes output/step8_execution/, plots/11_execution_savings.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_binance as dl  # noqa: E402
import hawkes_diagnostics as hd  # noqa: E402
import step5_alpha_signal as s5  # noqa: E402
import step6_binance_validation as s6  # noqa: E402
import step7_hawkes_ensemble as s7  # noqa: E402

OUT = Path("output/step8_execution")
PLOT = Path("plots/11_execution_savings.png")
EVERY_S, WINDOW_S, NOTIONAL, HALF_TICK = 60, 10, 10_000.0, 0.005
MODELS = ["baseline_ewma", "with_hawkes", "event_time", "hawkes_ensemble"]
LABEL = {"baseline_ewma": "Baseline", "with_hawkes": "+Hawkes", "event_time": "+Event-time",
         "hawkes_ensemble": "Hawkes ensemble", "none": "No signal"}


def day_predictions(prev_df, prev_tr, df, tr) -> dict[str, np.ndarray]:
    """1 s return forecasts for every second of `df`, models fitted on the previous day (as in step 7)."""
    xfit = s7.fit_exact(prev_tr)
    p_df = s7.exact_intensity(prev_df, prev_tr["ts"], prev_tr["trade_sign"].to_numpy(), xfit)
    n_df = s7.exact_intensity(df, tr["ts"], tr["trade_sign"].to_numpy(), xfit)
    hp = hd.fit_discrete(p_df["buy_events"].to_numpy(float), p_df["sell_events"].to_numpy(float))["x"]
    sets = s7.feature_sets(s6.BASELINE_FEATURES)
    preds = {}
    for name in MODELS:
        feats = sets[name]
        use_h = hp if any(c in feats for c in s7.HAWKES_COLS) else None
        x_tr, y_tr = s6.day_design(p_df, feats, "future_ret_1s", use_h)
        a = s7.tune_alpha(x_tr, y_tr) if name == "hawkes_ensemble" else s5.RIDGE_ALPHA
        model = s5.ridge_fit(x_tr, y_tr, a)
        frame = n_df.copy()
        if use_h is not None:
            h = s5.hawkes_one_step(use_h, frame["buy_events"].to_numpy(float), frame["sell_events"].to_numpy(float),
                                   index=frame.index)
            for c in h.columns:
                frame[c] = h[c]
        x = frame[feats].to_numpy(float)
        ok = np.isfinite(x).all(axis=1)
        p = np.full(len(frame), np.nan)
        p[ok] = s5.ridge_predict(model, x[ok])
        preds[name] = p
    return preds


def per_second_extremes(tr: pd.DataFrame, ts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Lowest sell-initiated and highest buy-initiated trade price in each second (NaN if none)."""
    sec = tr["ts"].dt.floor("1s")
    lo_sell = tr[tr["trade_sign"] == -1].groupby(sec[tr["trade_sign"] == -1])["price"].min()
    hi_buy = tr[tr["trade_sign"] == 1].groupby(sec[tr["trade_sign"] == 1])["price"].max()
    return lo_sell.reindex(ts).to_numpy(float), hi_buy.reindex(ts).to_numpy(float)


def simulate_day(price: np.ndarray, lo_sell: np.ndarray, hi_buy: np.ndarray, preds: dict) -> pd.DataFrame:
    rows = []
    n = len(price)
    side = 1
    for a in range(120, n - WINDOW_S - 1, EVERY_S):
        if not np.isfinite(price[a]):
            continue
        p0 = price[a]
        bench = p0 + side * HALF_TICK
        cost = lambda px: side * (px - p0) / p0 * 1e4  # noqa: E731
        base = {"arrival_s": a, "side": side, "bench_cost_bp": cost(bench)}
        for model in ["none", *MODELS]:
            pr = preds.get(model)
            # timing (taker): wait while the forecast favours waiting
            if model != "none":
                t_exec = a + WINDOW_S - 1
                for k in range(WINDOW_S):
                    t = a + k
                    if np.isfinite(pr[t]) and side * pr[t] > 0:   # price expected to move against us: go now
                        t_exec = t
                        break
                px = price[t_exec] + side * HALF_TICK
                rows.append({**base, "strategy": "timing", "model": model, "cost_bp": cost(px), "maker": False,
                             "wait_s": t_exec - a})
            # passive (maker first)
            limit = p0 - side * HALF_TICK
            filled_at, maker = None, False
            for k in range(1, WINDOW_S + 1):
                t = a + k
                through = (lo_sell[t] < limit - 1e-9) if side == 1 else (hi_buy[t] > limit + 1e-9)
                if np.isfinite(lo_sell[t] if side == 1 else hi_buy[t]) and through:
                    filled_at, maker = limit, True
                    break
                adverse = model != "none" and np.isfinite(pr[t]) and side * pr[t] > 0
                if adverse or k == WINDOW_S:
                    filled_at = price[t] + side * HALF_TICK
                    break
            rows.append({**base, "strategy": "passive", "model": model, "cost_bp": cost(filled_at), "maker": maker,
                         "wait_s": k})
        side = -side
    df = pd.DataFrame(rows)
    df["saving_bp"] = df["bench_cost_bp"] - df["cost_bp"]
    # Maker fills pay the maker fee instead of the taker fee: 0 at the standard tier (both 10 bp), +10 bp if maker is free.
    df["saving_bp_free_maker"] = df["saving_bp"] + np.where(df["maker"], 10.0, 0.0)
    return df


def summarise(all_orders: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (strat, model), g in all_orders.groupby(["strategy", "model"], sort=False):
        s = g["saving_bp"].to_numpy()
        daily = g.groupby("test_day")["saving_bp"].mean().to_numpy()
        ci = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
        out.append({"strategy": strat, "model": model, "orders": len(s), "mean_saving_bp": s.mean(),
                    "ci95_lo": s.mean() - ci, "ci95_hi": s.mean() + ci,
                    "p_orders": stats.ttest_1samp(s, 0).pvalue, "days_positive": int((daily > 0).sum()),
                    "n_days": len(daily), "p_days": stats.ttest_1samp(daily, 0).pvalue if len(daily) > 1 else np.nan,
                    "usd_per_order": s.mean() / 1e4 * NOTIONAL, "maker_fill_share": g["maker"].mean(),
                    "mean_saving_bp_free_maker": g["saving_bp_free_maker"].mean(), "mean_wait_s": g["wait_s"].mean()})
    return pd.DataFrame(out)


def paired(all_orders: pd.DataFrame) -> pd.DataFrame:
    """Same orders, different signals: per-order saving differences, tested across orders and across days."""
    rows = []
    for strat in ("timing", "passive"):
        t = all_orders[all_orders.strategy == strat].pivot_table(index=["test_day", "arrival_s"], columns="model",
                                                                 values="saving_bp")
        pairs = [(m, "baseline_ewma") for m in ("with_hawkes", "event_time", "hawkes_ensemble")]
        if strat == "passive":
            pairs.append(("hawkes_ensemble", "none"))
        for a, b in pairs:
            d = t[a] - t[b]
            daily = d.groupby(level=0).mean()
            rows.append({"strategy": strat, "a": a, "b": b, "mean_bp": d.mean(), "p_orders": stats.ttest_1samp(d, 0).pvalue,
                         "days_a_better": int((daily > 0).sum()), "n_days": len(daily),
                         "p_days": stats.ttest_1samp(daily, 0).pvalue})
    return pd.DataFrame(rows)


def plot(all_orders: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = {"none": "#8b91a0", "baseline_ewma": "#5b6478", "with_hawkes": "#1f5fe0", "event_time": "#6fa80f",
              "hawkes_ensemble": "#ff1a3c"}
    for (strat, model), g in all_orders.groupby(["strategy", "model"], sort=False):
        if strat != "passive" and model == "none":
            continue
        cum = g.sort_values(["test_day", "arrival_s"])["saving_bp"].cumsum().to_numpy() / 1e4 * NOTIONAL
        ax.plot(cum, color=colors[model], linewidth=2.4 if model == "hawkes_ensemble" else 1.4,
                linestyle="-" if strat == "passive" else "--", label=f"{strat}: {LABEL[model]}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Orders ($10k each, 6 test days)")
    ax.set_ylabel("Cumulative saving vs. trading immediately ($)")
    ax.set_title("Execution with order-flow forecasts, BTCUSDT out of sample")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = s6.DEFAULT_DATES
    days, trades = {}, {}
    for d in dates:
        tr = s6.load_trades(dl.OUTDIR / f"BTCUSDT-aggTrades-{d}.csv")
        days[d], _, _ = s6.build_feature_table(tr)
        trades[d] = tr
        print(f"[{d}] features", flush=True)
    orders = []
    for prev, nxt in zip(dates[:-1], dates[1:]):
        preds = day_predictions(days[prev], trades[prev], days[nxt], trades[nxt])
        df = days[nxt]
        lo, hi = per_second_extremes(trades[nxt], df["ts"])
        sim = simulate_day(df["mid"].to_numpy(float), lo, hi, preds).assign(test_day=nxt)
        orders.append(sim)
        print(f"[{nxt}] {len(sim)} order-strategy rows", flush=True)
    all_orders = pd.concat(orders, ignore_index=True)
    summary = summarise(all_orders)
    all_orders.to_csv(OUT / "orders.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "config.json").write_text(json.dumps({"every_s": EVERY_S, "window_s": WINDOW_S, "notional": NOTIONAL}, indent=2))
    pair = paired(all_orders)
    pair.to_csv(OUT / "paired_vs_baseline.csv", index=False)
    plot(all_orders)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
