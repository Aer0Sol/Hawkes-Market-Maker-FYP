"""Step 6: multi-day, cross-asset validation on Binance BTCUSDT.

Steps 1-5 use one day of one instrument (AAPL, 2024-06-03). This reruns the step 5 test with the
same code on several BTCUSDT days (24 h, another asset class and venue), three ways:

1. Hawkes fit per day: are the parameters stable across days? How much of the branching ratio is
   an hour-of-day baseline?
2. Within-day walk-forward (step 5's test) on each day separately: baseline vs +Hawkes, and
   baseline vs +event-time multi-scale features.
3. Cross-day: fit Hawkes and ridge on day d-1, predict day d. Days are close to independent, so a
   t-test across the daily loss differentials is the strongest evidence here.

It also exports the models fitted on the last day for the live dashboard (scripts/live_monitor.py).

Binance aggTrades carry no order book, so the feature set is trade flow only, and the price is
the last trade of each second (BTCUSDT's 1-cent spread on ~$65k makes bid-ask bounce negligible).
Dates in config/holdout.json are refused unless --allow-holdout is passed.
"""

from __future__ import annotations

import argparse
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
import step2_trade_label_and_hawkes as s2  # noqa: E402
import step5_alpha_signal as s5  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")

OUT_ROOT = Path("output/step6_binance_validation")
DOC_PATH = Path("docs/binance_validation.md")
PLOTS_DIR = Path("plots")
AAPL_SUMMARY = Path("output/step2_hawkes/step2_summary.json")
DEFAULT_DATES = ["2024-06-03", "2024-06-04", "2024-06-05", "2024-06-06", "2024-06-07", "2024-06-08", "2024-06-09"]

BASELINE_FEATURES = ["order_flow_imbalance", "lambda_buy_ewm", "lambda_sell_ewm", "rv_60", "trade_imbalance"]
HAWKES_FEATURES = BASELINE_FEATURES + ["hawkes_ofi", "lambda_buy_hawkes", "lambda_sell_hawkes"]
EVENT_FEATURES = BASELINE_FEATURES + s5.event_time_feature_names()
PRICE_TARGETS = s5.PRICE_TARGETS
TARGETS = PRICE_TARGETS + [s5.FLOW_TARGET]
MODELS = {"with_hawkes": HAWKES_FEATURES, "event_time": EVENT_FEATURES}
LABEL = {"with_hawkes": "+Hawkes", "event_time": "+event-time"}

verdict = s5.verdict
TICK = 0.01
# 88% of BTCUSDT's non-zero 1 s last-trade moves are exactly one tick: the bid-ask bounce. A sign hit on
# a bounce says which side traded last, not where the price went, so hit rates count only moves larger
# than 1.5 ticks. The squared-error tests are unaffected (they weight by size).
BOUNCE_TICKS = 1.5


def bounce_threshold(df: pd.DataFrame) -> float:
    return BOUNCE_TICKS * TICK / float(df["mid"].median())


def map_binance_aggressor(is_buyer_maker: pd.Series) -> np.ndarray:
    """Binance convention: if the buyer was the resting (maker) order, the trade was initiated by
    a sell order hitting the bid, so the aggressor is the SELLER. is_buyer_maker=False => aggressor
    is the buyer. Returns +1 buy-aggressor / -1 sell-aggressor, matching step2's trade_sign."""
    return np.where(is_buyer_maker.astype(bool), -1, 1)


def load_trades(path: Path) -> pd.DataFrame:
    cols = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
            "transact_time", "is_buyer_maker", "is_best_match"]
    df = pd.read_csv(path, header=None, names=cols)
    # Binance switched its daily files from milliseconds to microseconds (16 digits) in 2025.
    unit = "us" if df["transact_time"].max() > 1e14 else "ms"
    df["ts"] = pd.to_datetime(df["transact_time"], unit=unit, utc=True)
    df["trade_sign"] = map_binance_aggressor(df["is_buyer_maker"])
    return df[["ts", "price", "quantity", "trade_sign"]]


def build_feature_table(trades: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    counts_df = s2.build_bucketed_event_counts(trades, bucket="1s")

    tr = trades.set_index("ts").sort_index()
    tr["notional"] = tr["price"] * tr["quantity"]
    bucket = tr.resample("1s").agg(
        vwap_num=("notional", "sum"), volume=("quantity", "sum"),
        last_price=("price", "last"), trade_count=("price", "size"),
    )
    bucket["signed_volume"] = (tr["trade_sign"] * tr["quantity"]).resample("1s").sum()
    bucket["vwap"] = bucket["vwap_num"] / bucket["volume"]
    # Price = last trade of the bucket (end-of-bucket, like AAPL's end-of-bucket mid). A bucket VWAP
    # averages over the second, which makes the next VWAP return mechanically predictable from the
    # latest flow (lag-1 autocorrelation 0.36 vs 0.14 here). Bid-ask bounce in the last trade is
    # negligible for BTCUSDT: a 1-cent spread on a ~$65k price is ~0.0015 bp.
    bucket["mid"] = bucket["last_price"].ffill()
    bucket["trade_imbalance"] = np.where(bucket["volume"] > 0, bucket["signed_volume"] / bucket["volume"], 0.0)
    bucket = bucket.reset_index().rename(columns={"index": "ts"})

    hawkes_fit = s2.fit_discrete_hawkes(counts_df)
    baseline_int = s2.estimate_baseline_intensities(counts_df)
    hawkes_int = s2.build_hawkes_intensity_df(counts_df, hawkes_fit)

    df = bucket.merge(baseline_int[["ts", "lambda_buy_ewm", "lambda_sell_ewm", "order_flow_imbalance"]], on="ts")
    df = df.merge(hawkes_int[["ts", "lambda_buy_hawkes", "lambda_sell_hawkes", "hawkes_ofi"]], on="ts")
    df = df.merge(counts_df, on="ts")

    df["log_mid"] = np.log(df["mid"])
    df["ret_1"] = df["log_mid"].diff()
    df["rv_60"] = df["ret_1"].rolling(60).std()
    for n, name in [(1, "future_ret_1s"), (5, "future_ret_5s"), (10, "future_ret_10s")]:
        df[name] = df["log_mid"].shift(-n) - df["log_mid"]

    df = s5.add_event_time_features(df, trades["ts"], trades["trade_sign"].to_numpy())
    df = s5.add_flow_target(df)
    return df, hawkes_fit, counts_df


def describe_fit(date: str, counts_df: pd.DataFrame, hawkes_fit: dict) -> dict:
    p = hawkes_fit["params"]
    x = np.array([p[k] for k in hd.PARAM_NAMES])
    st = hd.stability(x)
    yb = counts_df["buy_events"].to_numpy(float)
    ys = counts_df["sell_events"].to_numpy(float)
    hour = ((counts_df["ts"] - counts_df["ts"].iloc[0]).dt.total_seconds().to_numpy() // 3600).astype(int)
    tv = hd.fit_discrete_tv(yb, ys, hour)
    return {"date": date, "events": int(yb.sum() + ys.sum()), "beta": p["beta"],
            "half_life_s": st["half_life_buckets"], "branching": st["spectral_radius_branching"],
            "companion": st["spectral_radius_companion"], "branching_hour_of_day_mu": tv["spectral_radius_branching"],
            "half_life_hour_of_day_mu": tv["half_life_buckets"],
            "share_buy_buckets_ge3": hd.occupancy(yb)["3+"]}


def within_day(df: pd.DataFrame, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, paired = [], []
    thr = bounce_threshold(df)
    for alt, feats in MODELS.items():
        for target in TARGETS:
            r = s5.run_target(df, target, None, BASELINE_FEATURES, feats, regime_col=None, alt_label=alt,
                              hit_min_abs=thr if target in PRICE_TARGETS else 0.0)
            summaries.append(r["summary"].assign(date=date, alt_model=alt))
            paired.append({**r["paired"], "date": date})
    return pd.concat(summaries, ignore_index=True), pd.DataFrame(paired)


def day_design(df: pd.DataFrame, feats: list[str], target: str, hawkes_params: np.ndarray | None) -> tuple:
    d = df.copy()
    if hawkes_params is not None:
        h = s5.hawkes_one_step(hawkes_params, d["buy_events"].to_numpy(float), d["sell_events"].to_numpy(float),
                               index=d.index)
        for c in h.columns:
            d[c] = h[c]
    v = d[list(dict.fromkeys(feats + [target]))].dropna()
    return v[feats].to_numpy(float), v[target].to_numpy(float)


def cross_day(days: dict[str, pd.DataFrame], dates: list[str]) -> pd.DataFrame:
    """Train on day d-1 (Hawkes parameters and ridge), test on day d."""
    rows = []
    for prev, nxt in zip(dates[:-1], dates[1:]):
        p_df, n_df = days[prev], days[nxt]
        fit = hd.fit_discrete(p_df["buy_events"].to_numpy(float), p_df["sell_events"].to_numpy(float))["x"]
        for alt, feats in MODELS.items():
            hp = fit if alt == "with_hawkes" else None
            for target in TARGETS:
                xb_tr, y_tr = day_design(p_df, BASELINE_FEATURES, target, None)
                xb_te, y_te = day_design(n_df, BASELINE_FEATURES, target, None)
                xa_tr, ya_tr = day_design(p_df, feats, target, hp)
                xa_te, ya_te = day_design(n_df, feats, target, hp)
                n = min(len(y_te), len(ya_te))  # identical rows: the event-time/Hawkes columns have no NaNs
                pb = s5.ridge_fit_predict(xb_tr, y_tr, xb_te, s5.RIDGE_ALPHA)[:n]
                pa = s5.ridge_fit_predict(xa_tr, ya_tr, xa_te, s5.RIDGE_ALPHA)[:n]
                y = ya_te[:n]
                nz = np.abs(y) > (bounce_threshold(n_df) if target in PRICE_TARGETS else 0.0)
                d = (y - pb) ** 2 - (y - pa) ** 2
                rows.append({"train_day": prev, "test_day": nxt, "alt_model": alt, "target": target,
                             "mean_se_diff": float(d.mean()),
                             "r2_base": 1 - float(((y - pb) ** 2).sum() / (y ** 2).sum()),
                             "r2_alt": 1 - float(((y - pa) ** 2).sum() / (y ** 2).sum()),
                             "hit_base": float((np.sign(pb[nz]) == np.sign(y[nz])).mean()),
                             "hit_alt": float((np.sign(pa[nz]) == np.sign(y[nz])).mean())})
    return pd.DataFrame(rows)


def cross_day_summary(cd: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (alt, target), g in cd.groupby(["alt_model", "target"], sort=False):
        d = g["mean_se_diff"].to_numpy()
        t = stats.ttest_1samp(d, 0.0) if len(d) > 1 else None
        out.append({"alt_model": alt, "target": target, "n_test_days": len(d),
                    "days_alt_better": int((d > 0).sum()), "p_value_across_days": float(t.pvalue) if t else np.nan,
                    "mean_hit_gain_pp": float(100 * (g["hit_alt"] - g["hit_base"]).mean()),
                    "mean_r2_base": float(g["r2_base"].mean()), "mean_r2_alt": float(g["r2_alt"].mean())})
    return pd.DataFrame(out)


def export_live_model(df: pd.DataFrame, date: str):
    """Models trained on one full day, for scripts/live_monitor.py (1 s return of the last trade price)."""
    yb, ys = df["buy_events"].to_numpy(float), df["sell_events"].to_numpy(float)
    fit = hd.fit_discrete(yb, ys)
    models = {}
    for name, feats in [("baseline_ewma", BASELINE_FEATURES), *MODELS.items()]:
        x, y = day_design(df, feats, "future_ret_1s", fit["x"] if name == "with_hawkes" else None)
        m = s5.ridge_fit(x, y, s5.RIDGE_ALPHA)
        models[name] = {"features": feats, "mu": m["mu"].tolist(), "sigma": m["sigma"].tolist(),
                        "beta": m["beta"].tolist(), "intercept": m["intercept"]}
    out = {"symbol": "BTCUSDT", "trained_on": date, "target": "ln(last price at t+1 s) - ln(last price at t)",
           "hit_min_abs_move": BOUNCE_TICKS * TICK,
           "hawkes_params": dict(zip(hd.PARAM_NAMES, map(float, fit["x"]))), "ewma_span": 60, "rv_window": 60,
           "event_half_lives_s": list(s5.EVENT_HALF_LIVES_S), "models": models}
    (OUT_ROOT / "live_model.json").write_text(json.dumps(out, indent=2))


def aapl_fit_row() -> dict | None:
    if not AAPL_SUMMARY.exists():
        return None
    p = json.loads(AAPL_SUMMARY.read_text())["hawkes_fit"]["params"]
    st = hd.stability(np.array([p[k] for k in hd.PARAM_NAMES]))
    tv_csv = Path("output/hawkes_diagnostics/diagnostics.json")
    tv = json.loads(tv_csv.read_text()).get("time_varying_baseline", {}) if tv_csv.exists() else {}
    return {"beta": p["beta"], "half_life_s": st["half_life_buckets"], "branching": st["spectral_radius_branching"],
            "companion": st["spectral_radius_companion"],
            "branching_tv": tv.get("spectral_radius_branching", np.nan)}


def write_report(dates, fits: pd.DataFrame, within: pd.DataFrame, within_sum: pd.DataFrame,
                 cd: pd.DataFrame, cds: pd.DataFrame):
    L = [
        "# Multi-day validation: Binance BTCUSDT",
        "",
        f"Generated by `scripts/step6_binance_validation.py` on {len(dates)} BTCUSDT days "
        f"({dates[0]} to {dates[-1]}, UTC, 24 h each, {int(fits['events'].sum()):,} aggressor events). "
        "Same Hawkes-fit code and same walk-forward test as the AAPL analysis (`docs/alpha_signal.md`). "
        "The dates in `config/holdout.json` (2024-07-01 to 07-07) have not been downloaded or evaluated.",
        "",
        "Not a like-for-like replication of AAPL: aggTrades has no order book, so the baseline is trade flow "
        "only (`order_flow_imbalance`, `lambda_*_ewm`, `rv_60`, `trade_imbalance`); the price is the last trade "
        "of each second. A Binance aggTrade merges the fills of one taker order at one price, so its events "
        "are closer to orders than AAPL's prints.",
        "",
        "## 1. Hawkes fit by day",
        "",
        "| Day | Events | β | Half-life (s) | Branching | Companion | Branching, hour-of-day μ | Half-life, hour-of-day μ (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    a = aapl_fit_row()
    if a:
        L.append(f"| AAPL 2024-06-03 (6.5 h; 30-min μ) | 164,922 | {a['beta']:.3f} | {a['half_life_s']:.1f} | "
                 f"{a['branching']:.3f} | {a['companion']:.3f} | {a['branching_tv']:.3f} | — |")
    for _, r in fits.iterrows():
        L.append(f"| {r['date']} | {r['events']:,} | {r['beta']:.3f} | {r['half_life_s']:.1f} | {r['branching']:.3f} | "
                 f"{r['companion']:.3f} | {r['branching_hour_of_day_mu']:.3f} | {r['half_life_hour_of_day_mu']:.1f} |")
    L += [
        "",
        f"Across days the constant-baseline branching ratio ranges {fits['branching'].min():.2f}–"
        f"{fits['branching'].max():.2f} and the half-life {fits['half_life_s'].min():.1f}–"
        f"{fits['half_life_s'].max():.1f} s. With an hour-of-day baseline the branching ratio falls to "
        f"{fits['branching_hour_of_day_mu'].min():.2f}–{fits['branching_hour_of_day_mu'].max():.2f}: part of the "
        "apparent excitation is daily seasonality, as on AAPL.",
        "",
        "## 2. Within-day walk-forward, each day separately",
        "",
        "Step 5's test (5 chronological folds, Hawkes refit per training window, one-step-ahead features) "
        "run on each day. Each cell is that day's verdict for the alternative model against the baseline: "
        "**R** robust (both paired tests p < 0.0125), **S** suggestive (either p < 0.05), **·** null; "
        "**+** the alternative was better, **−** the baseline was better. Hit rates here and below count only moves "
        "larger than 1.5 ticks: 88% of non-zero one-second last-trade moves are a single tick, i.e. the bid-ask "
        "bounce, and a sign hit on a bounce says which side traded last rather than where the price went.",
        "",
    ]
    for alt in MODELS:
        L.append(f"### Baseline vs {LABEL[alt]}")
        L.append("")
        L.append("| Day | " + " | ".join(f"`{t}`" for t in TARGETS) + " |")
        L.append("|---|" + "---|" * len(TARGETS))
        for date in dates:
            cells = []
            for t in TARGETS:
                r = within[(within["date"] == date) & (within["alt_model"] == alt) & (within["target"] == t)].iloc[0]
                v = verdict(r)
                sym = "R" if v.startswith("robust") else "S" if v.startswith("suggestive") else "·"
                sgn = "" if sym == "·" else ("+" if r["mean_se_diff_fold_level"] > 0 else "−")
                cells.append(sym + sgn)
            L.append(f"| {date} | " + " | ".join(cells) + " |")
        L.append("")
        s = within_sum[within_sum["alt_model"] == alt]
        hits = s.pivot_table(index=["date", "target"], columns="model", values="hit_rate_mean").reset_index()
        hits["gain"] = 100 * (hits[alt] - hits["baseline_ewma"])
        L.append("Mean within-day hit-rate gain over days (pp): " + ", ".join(
            f"`{t}` {hits.loc[hits['target'] == t, 'gain'].mean():+.2f}" for t in TARGETS) + ".")
        L.append("")
    L += [
        "## 3. Cross-day: train on day d−1, test on day d",
        "",
        "Hawkes parameters and ridge coefficients are fitted on the whole previous day, then applied unchanged "
        "to the next. Each test day gives one mean loss differential; days are close to independent, so the "
        "t-test across days needs no autocorrelation correction. Positive means the alternative beat the baseline.",
        "",
        "| Model | Target | Test days | Days better | p across days | Mean hit gain (pp) | R² base | R² alt |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cds.iterrows():
        L.append(f"| {LABEL[r['alt_model']]} | `{r['target']}` | {r['n_test_days']} | {r['days_alt_better']} | "
                 f"{r['p_value_across_days']:.3g} | {r['mean_hit_gain_pp']:+.2f} | {r['mean_r2_base']:.4f} | "
                 f"{r['mean_r2_alt']:.4f} |")
    L += ["", "## Conclusion", ""]
    for alt in MODELS:
        g = cds[cds["alt_model"] == alt].set_index("target")
        price = g.loc[PRICE_TARGETS]
        pv = ", ".join(f"{t.replace('future_', '')} {g.loc[t, 'p_value_across_days']:.3g}" for t in TARGETS)
        L.append(f"- **{LABEL[alt]}:** cross-day it beats the baseline on {int(price['days_alt_better'].sum())} of "
                 f"{int(price['n_test_days'].sum())} day×horizon return tests and on "
                 f"{int(g.loc[s5.FLOW_TARGET, 'days_alt_better'])} of {int(g.loc[s5.FLOW_TARGET, 'n_test_days'])} days "
                 f"for next-second flow. p across days: {pv} (Bonferroni threshold 0.0125; six test days give the "
                 f"test little power). Hit-rate gain on real moves: {price['mean_hit_gain_pp'].min():+.1f} to "
                 f"{price['mean_hit_gain_pp'].max():+.1f} pp.")
    L += [
        "",
        "These are gross forecasting gains of a linear model on one week of one crypto pair, before any trading "
        "cost. They say the Hawkes / event-time features carry information the EWMA baseline lacks on this "
        "instrument; they do not say a strategy built on them is profitable (see `output/step4b_fixed/` for "
        "what happens to an AAPL market maker once fills are replayed from real trades).",
        "",
        "## Limitations",
        "",
        "- One instrument and one week; AAPL (`docs/alpha_signal.md`) gives a different answer, and that "
        "difference is not explained by the weaker BTCUSDT baseline (AAPL with the same flow-only features "
        "is still null on returns).",
        "- No order book here: no spread, imbalance or microprice features, and no trading cost.",
        "- The hour-of-day baseline is fitted on the same day it describes.",
        "- Ridge α = 1 is not tuned; models are linear.",
    ]
    DOC_PATH.parent.mkdir(exist_ok=True)
    DOC_PATH.write_text("\n".join(L) + "\n")


def plot_cross_day(cd: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(TARGETS))
    for off, (alt, color) in zip([-0.18, 0.18], [("with_hawkes", "#DD8452"), ("event_time", "#55A868")]):
        g = cd[cd["alt_model"] == alt]
        gains = [100 * (g.loc[g["target"] == t, "hit_alt"] - g.loc[g["target"] == t, "hit_base"]).to_numpy()
                 for t in TARGETS]
        means = [v.mean() for v in gains]
        ses = [v.std(ddof=1) / np.sqrt(len(v)) * stats.t.ppf(0.975, len(v) - 1) for v in gains]
        ax.bar(x + off, means, 0.34, yerr=ses, capsize=4, color=color, label=LABEL[alt])
        for i, v in enumerate(gains):
            ax.scatter(np.full(len(v), x[i] + off), v, s=10, color="black", zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS, rotation=12)
    ax.set_ylabel("Hit-rate gain over EWMA baseline (pp)")
    ax.set_title("BTCUSDT cross-day test (train d−1, test d): gain per test day, mean ± 95% CI")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_binance_oos_r2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--allow-holdout", action="store_true", help="only for the single final evaluation")
    args = ap.parse_args()
    for d in args.dates:
        dl.check_not_holdout(args.symbol, d, args.allow_holdout)
    missing = [d for d in args.dates if not (dl.OUTDIR / f"{args.symbol}-aggTrades-{d}.csv").exists()]
    if missing:
        sys.exit(f"Missing data for {missing}. Run: uv run python scripts/download_binance.py --date <date>")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    days, fits, within_p, within_s = {}, [], [], []
    for d in args.dates:
        print(f"[{d}] features ...", flush=True)
        trades = load_trades(dl.OUTDIR / f"{args.symbol}-aggTrades-{d}.csv")
        df, hawkes_fit, counts_df = build_feature_table(trades)
        days[d] = df
        fits.append(describe_fit(d, counts_df, hawkes_fit))
        print(f"[{d}] within-day walk-forward ...", flush=True)
        s, p = within_day(df, d)
        within_s.append(s)
        within_p.append(p)
    fits = pd.DataFrame(fits)
    within = pd.concat(within_p, ignore_index=True)
    within_sum = pd.concat(within_s, ignore_index=True)
    print("cross-day ...", flush=True)
    cd = cross_day(days, args.dates)
    cds = cross_day_summary(cd)

    fits.to_csv(OUT_ROOT / "hawkes_fit_by_day.csv", index=False)
    within.to_csv(OUT_ROOT / "within_day_paired.csv", index=False)
    within_sum.to_csv(OUT_ROOT / "within_day_summary.csv", index=False)
    cd.to_csv(OUT_ROOT / "cross_day_by_day.csv", index=False)
    cds.to_csv(OUT_ROOT / "cross_day_summary.csv", index=False)
    export_live_model(days[args.dates[-1]], args.dates[-1])
    write_report(args.dates, fits, within, within_sum, cd, cds)
    plot_cross_day(cd)
    print(f"Wrote {OUT_ROOT}/, {DOC_PATH}, plots/09_binance_oos_r2.png")


if __name__ == "__main__":
    main()
