"""Step 7: the Hawkes ensemble, combining every Hawkes signal from steps 2-6 into one forecaster.

Features (20):
  - baseline flow (5): EWMA buy/sell intensities and imbalance, 60 s realised vol, trade imbalance;
  - discrete 1 s Hawkes (3): one-step-ahead lambda_{t+1|t} and its imbalance (step 2 model);
  - exact event-time Hawkes (3): the intensity of a 3-exponential bivariate Hawkes process fitted by
    exact maximum likelihood on event timestamps (hawkes_diagnostics.fit_continuous), evaluated at
    each second's end -- log buy, log sell intensity and their imbalance;
  - event-time kernel features (9): decayed buy/sell counts at 0.1 / 1 / 10 s half-lives.
The ridge penalty is chosen by chronological cross-validation inside the training data, so the
larger feature set cannot overfit the way the event-time model did on one AAPL day.

Evaluation:
  - BTCUSDT: cross-day (all parameters fitted on day d-1, applied unchanged to day d), against the
    baseline, +Hawkes, +event-time and +exact Hawkes models, with the paper taker P&L of each.
  - AAPL: walk-forward within the one day, exact Hawkes refitted on each training window.
Writes output/step7_hawkes_ensemble/, docs/hawkes_ensemble.md, plots/10_hawkes_ensemble.png, and adds
the ensemble to output/step6_binance_validation/live_model.json for the live monitor.
"""

from __future__ import annotations

import json
import sys
import time
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
import taker_backtest as tb  # noqa: E402

OUT = Path("output/step7_hawkes_ensemble")
DOC = Path("docs/hawkes_ensemble.md")
PLOT = Path("plots/10_hawkes_ensemble.png")
LIVE_MODEL = Path("output/step6_binance_validation/live_model.json")

EXACT_K = 3
EXACT_OMEGA0 = [2000.0, 2.0, 0.05]
EXACT_WINDOW_H = 6          # fit the exact model on the last 6 h of the training data
ALPHA_GRID = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
EXACT_FEATURES = ["hx_log_buy", "hx_log_sell", "hx_ofi"]
HAWKES_COLS = ["hawkes_ofi", "lambda_buy_hawkes", "lambda_sell_hawkes"]
ET = s5.event_time_feature_names()


def feature_sets(baseline: list[str]) -> dict[str, list[str]]:
    return {
        "baseline_ewma": baseline,
        "with_hawkes": baseline + HAWKES_COLS,
        "event_time": baseline + ET,
        "exact_hawkes": baseline + EXACT_FEATURES,
        "hawkes_ensemble": baseline + HAWKES_COLS + EXACT_FEATURES + ET,
    }


LABEL = {"baseline_ewma": "Baseline", "with_hawkes": "+Hawkes (1 s)", "event_time": "+Event-time kernels",
         "exact_hawkes": "+Exact Hawkes", "hawkes_ensemble": "Hawkes ensemble"}


# ------------------------------------------------------------------ exact Hawkes
def fit_exact(trades: pd.DataFrame) -> dict:
    """3-exponential bivariate Hawkes by exact MLE on the last EXACT_WINDOW_H hours of `trades`."""
    end = trades["ts"].max()
    start = max(trades["ts"].min(), end - pd.Timedelta(hours=EXACT_WINDOW_H))
    sub = trades[trades["ts"] >= start]
    times, types, _ = hd.event_stream(sub, start)
    T = float((end - start).total_seconds())
    fit = None
    # The likelihood surface has divergent directions (alpha -> inf); retry from other starting decays.
    for omega0 in (EXACT_OMEGA0, [1000.0, 1.0, 0.02], [5000.0, 5.0, 0.1]):
        try:
            cand, _, _ = hd.fit_continuous(times, types, T, EXACT_K, omega0)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue
        if np.all(np.isfinite(cand["alpha"])) and np.all(np.isfinite(cand["omega_per_s"])) and cand["spectral_radius_branching"] < 1:
            fit = cand
            break
    if fit is None:
        raise RuntimeError(f"exact Hawkes fit failed from every starting point ({len(times)} events)")
    return {"mu": fit["mu_per_s"], "alpha": fit["alpha"], "omega": fit["omega_per_s"],
            "half_life_s": fit["half_life_s"], "branching": fit["spectral_radius_branching"],
            "n_events": int(len(times)), "success": fit["success"]}


def exact_intensity(df: pd.DataFrame, ev_ts: pd.Series, ev_sign: np.ndarray, p: dict) -> pd.DataFrame:
    """Fitted exact intensity lambda_i(t) = mu_i + sum_k sum_j alpha[k,i,j] R_kj(t) at each bucket end."""
    ev = pd.DataFrame({"ts": ev_ts.to_numpy(), "side": (np.asarray(ev_sign) < 0).astype(np.int64)})
    ev = ev.drop_duplicates().sort_values("ts", kind="stable")
    origin = df["ts"].iloc[0]
    ev_t = (ev["ts"] - origin).dt.total_seconds().to_numpy()
    grid_t = (df["ts"] - origin).dt.total_seconds().to_numpy() + 1.0
    R = s5._decayed_event_sums(ev_t, ev["side"].to_numpy(), grid_t, np.asarray(p["omega"], dtype=float))
    lam = np.asarray(p["mu"])[None, :] + np.einsum("gkj,kij->gi", R, np.asarray(p["alpha"]))
    lam = np.maximum(lam, 1e-8)
    out = df.copy()
    out["hx_log_buy"] = np.log(lam[:, 0])
    out["hx_log_sell"] = np.log(lam[:, 1])
    out["hx_ofi"] = (lam[:, 0] - lam[:, 1]) / (lam[:, 0] + lam[:, 1])
    return out


# ------------------------------------------------------------------ ridge with tuned penalty
def tune_alpha(x: np.ndarray, y: np.ndarray, n_folds: int = 4) -> float:
    """Chronological expanding-window CV inside the training data; returns the best ridge penalty."""
    folds = s5.walk_forward_folds(len(y), n_folds, 0.4)
    best, best_mse = ALPHA_GRID[0], np.inf
    for a in ALPHA_GRID:
        mse = np.mean([np.mean((y[te] - s5.ridge_fit_predict(x[tr], y[tr], x[te], a)) ** 2) for tr, te in folds])
        if mse < best_mse:
            best, best_mse = a, mse
    return best


def design(df: pd.DataFrame, feats: list[str], target: str, hawkes_params) -> tuple[np.ndarray, np.ndarray]:
    return s6.day_design(df, feats, target, hawkes_params)


# ------------------------------------------------------------------ BTCUSDT cross-day
def btc_cross_day(dates: list[str]):
    days, trades, fits_d, fits_x = {}, {}, {}, {}
    for d in dates:
        t0 = time.time()
        tr = s6.load_trades(dl.OUTDIR / f"BTCUSDT-aggTrades-{d}.csv")
        df, _, _ = s6.build_feature_table(tr)
        days[d], trades[d] = df, tr
        print(f"[{d}] features {time.time() - t0:.0f}s", flush=True)
    rows, pnl_rows, alphas = [], [], []
    for prev, nxt in zip(dates[:-1], dates[1:]):
        t0 = time.time()
        fits_d[prev] = hd.fit_discrete(days[prev]["buy_events"].to_numpy(float),
                                       days[prev]["sell_events"].to_numpy(float))["x"]
        fits_x[prev] = fit_exact(trades[prev])
        p_df = exact_intensity(days[prev], trades[prev]["ts"], trades[prev]["trade_sign"].to_numpy(), fits_x[prev])
        n_df = exact_intensity(days[nxt], trades[nxt]["ts"], trades[nxt]["trade_sign"].to_numpy(), fits_x[prev])
        print(f"[{prev} -> {nxt}] exact fit {time.time() - t0:.0f}s, half-lives "
              f"{np.round(fits_x[prev]['half_life_s'], 4)}", flush=True)
        thr = s6.bounce_threshold(n_df)
        for target in s6.TARGETS:
            preds, y_ref = {}, None
            for name, feats in feature_sets(s6.BASELINE_FEATURES).items():
                hp = fits_d[prev] if any(c in feats for c in HAWKES_COLS) else None
                x_tr, y_tr = design(p_df, feats, target, hp)
                x_te, y_te = design(n_df, feats, target, hp)
                a = tune_alpha(x_tr, y_tr) if name == "hawkes_ensemble" else s5.RIDGE_ALPHA
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
                             "hit": float((np.sign(pr[moved]) == np.sign(y[moved])).mean())})
                if target == "future_ret_1s":
                    pnl_rows.append({"test_day": nxt, "model": name, **tb.strategy_pnl(pr, y)})
    return pd.DataFrame(rows), pd.DataFrame(pnl_rows), pd.DataFrame(alphas), fits_x, days, trades, fits_d


def summarise_cross_day(cd: pd.DataFrame) -> pd.DataFrame:
    base_hit = cd[cd.model == "baseline_ewma"].set_index(["test_day", "target"])["hit"]
    out = []
    for (name, target), g in cd.groupby(["model", "target"], sort=False):
        if name == "baseline_ewma":
            continue
        d = g["mean_se_diff_vs_base"].to_numpy()
        gain = 100 * (g.set_index(["test_day", "target"])["hit"] - base_hit.loc[g.set_index(["test_day", "target"]).index])
        out.append({"model": name, "target": target, "days_better": int((d > 0).sum()), "n_days": len(d),
                    "p_across_days": float(stats.ttest_1samp(d, 0.0).pvalue), "hit_gain_pp": float(gain.mean()),
                    "r2": float(g["r2"].mean())})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ AAPL walk-forward
def aapl_walk_forward():
    df = s5.load_features()
    trades = pd.read_parquet(s5.TRADES_PARQUET)
    base = s5.BASELINE_FEATURES
    sets = feature_sets(base)
    rows = []
    fit_cache: dict[int, dict] = {}
    for target in s5.PRICE_TARGETS + [s5.FLOW_TARGET]:
        cols = list(dict.fromkeys(sum(sets.values(), []) + [target]))
        cols = [c for c in cols if c not in EXACT_FEATURES]
        valid = df[cols].dropna()
        pos = df.index.get_indexer(valid.index)
        folds = s5.walk_forward_folds(len(valid), s5.N_FOLDS, s5.WARMUP_FRAC)
        y_all = valid[target].to_numpy(float)
        se = {m: [] for m in sets}
        for tr, te in folds:
            end = int(pos[tr].max()) + 1
            train_end_ts = df["ts"].iloc[end - 1] + pd.Timedelta(seconds=1)
            if end not in fit_cache:
                fit_cache[end] = fit_exact(trades[trades["ts"] < train_end_ts])
            xfit = fit_cache[end]
            dx = exact_intensity(df, trades["ts"], trades["trade_sign"].to_numpy(), xfit).iloc[pos]
            dh = s5.hawkes_features_fitted_on(df, end).iloc[pos]
            frame = valid.copy()
            for c in EXACT_FEATURES:
                frame[c] = dx[c].to_numpy()
            for c in HAWKES_COLS:
                frame[c] = dh[c].to_numpy()
            for name, feats in sets.items():
                x = frame[feats].to_numpy(float)
                a = tune_alpha(x[tr], y_all[tr]) if name == "hawkes_ensemble" else s5.RIDGE_ALPHA
                se[name].append((y_all[te] - s5.ridge_fit_predict(x[tr], y_all[tr], x[te], a)) ** 2)
        for name in sets:
            if name == "baseline_ewma":
                continue
            d = [b - a for b, a in zip(se["baseline_ewma"], se[name])]
            fold_means = np.array([x.mean() for x in d])
            _, p_dm = s5.diebold_mariano_hac(np.concatenate(d), s5.TARGET_HORIZON[target] + 10)
            rows.append({"model": name, "target": target, "folds_better": int((fold_means > 0).sum()),
                         "p_fold": float(stats.ttest_1samp(fold_means, 0.0).pvalue), "p_dm": p_dm})
        print(f"[AAPL] {target} done", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ live export, report, plot
def export_live(days, trades, dates, fits_d):
    last = dates[-1]
    xfit = fit_exact(trades[last])
    df = exact_intensity(days[last], trades[last]["ts"], trades[last]["trade_sign"].to_numpy(), xfit)
    hp = hd.fit_discrete(df["buy_events"].to_numpy(float), df["sell_events"].to_numpy(float))["x"]
    feats = feature_sets(s6.BASELINE_FEATURES)["hawkes_ensemble"]
    x, y = design(df, feats, "future_ret_1s", hp)
    a = tune_alpha(x, y)
    m = s5.ridge_fit(x, y, a)
    live = json.loads(LIVE_MODEL.read_text())
    live["exact_hawkes"] = {"mu": xfit["mu"], "alpha": xfit["alpha"], "omega": xfit["omega"],
                            "half_life_s": xfit["half_life_s"], "fitted_on": f"{last}, last {EXACT_WINDOW_H} h"}
    live["models"]["hawkes_ensemble"] = {"features": feats, "mu": m["mu"].tolist(), "sigma": m["sigma"].tolist(),
                                         "beta": m["beta"].tolist(), "intercept": m["intercept"], "ridge_alpha": a}
    LIVE_MODEL.write_text(json.dumps(live, indent=2))
    return a


def write_report(cds, pnl, alphas, fits_x, aapl, dates):
    L = ["# Hawkes ensemble", "",
         "Generated by `scripts/step7_hawkes_ensemble.py`. One forecaster combining every Hawkes signal in the project: "
         "the discrete 1 s Hawkes intensity, the **exact event-time Hawkes intensity** (3-exponential bivariate Hawkes "
         f"fitted by maximum likelihood on the last {EXACT_WINDOW_H} h of the training data), and the event-time kernel "
         "features, on top of the 5 baseline flow features (20 features). Its ridge penalty is chosen by chronological "
         f"cross-validation inside the training data (grid {list(ALPHA_GRID)}); the single-family models keep α = 1.", "",
         "## Exact Hawkes fits (BTCUSDT, one per training day)", "",
         "| Fitted on | Events | Kernel half-lives | Branching ratio |", "|---|---:|---|---:|"]
    for d, f in fits_x.items():
        hl = ", ".join(f"{h * 1000:.1f} ms" if h < 1 else f"{h:.1f} s" for h in f["half_life_s"])
        L.append(f"| {d} (last {EXACT_WINDOW_H} h) | {f['n_events']:,} | {hl} | {f['branching']:.2f} |")
    L += ["", f"## BTCUSDT cross-day test ({dates[1]} to {dates[-1]}): train on day d−1, test on day d", "",
          "Each row compares a model with the baseline on six test days. Hit-rate gains count only moves larger than "
          "the bid-ask bounce.", "",
          "| Model | Target | Days better | p across days | Hit gain (pp) | R² |", "|---|---|---:|---:|---:|---:|"]
    for _, r in cds.iterrows():
        bold = "**" if r["model"] == "hawkes_ensemble" else ""
        L.append(f"| {bold}{LABEL[r['model']]}{bold} | `{r['target']}` | {r['days_better']}/{r['n_days']} | "
                 f"{r['p_across_days']:.3g} | {r['hit_gain_pp']:+.2f} | {r['r2']:.4f} |")
    L += ["", "Ridge penalty chosen for the ensemble (by training day and target): " +
          ", ".join(f"{int(a):g}" for a in alphas["ridge_alpha"]) + ".", "",
          "## Paper P&L of trading the 1 s forecast ($10k long/short, six test days)", "",
          "| Model | Gross | Traded | Net at 10 bp | Break-even fee (bp) |", "|---|---:|---:|---:|---:|"]
    tot = pnl.groupby("model", sort=False)[["gross", "traded", "net_10bp"]].sum()
    for name, r in tot.iterrows():
        L.append(f"| {LABEL[name]} | ${r['gross']:,.0f} | ${r['traded'] / 1e9:.2f} bn | ${r['net_10bp']:,.0f} | "
                 f"{1e4 * r['gross'] / r['traded']:.3f} |")
    L += ["", "## AAPL (one day, walk-forward, exact Hawkes refitted per training window)", "",
          "| Model | Target | Folds better | Fold p | DM p |", "|---|---|---:|---:|---:|"]
    for _, r in aapl.iterrows():
        L.append(f"| {LABEL[r['model']]} | `{r['target']}` | {r['folds_better']}/5 | {r['p_fold']:.3f} | {r['p_dm']:.3g} |")
    ens = cds[cds["model"] == "hawkes_ensemble"].set_index("target")
    best_single = cds[cds["model"] != "hawkes_ensemble"].groupby("target")["hit_gain_pp"].max()
    L += ["", "## Reading", "",
          "- BTCUSDT: the ensemble beats the baseline on "
          f"{int(ens['days_better'].sum())} of {int(ens['n_days'].sum())} day×target tests, with hit-rate gains of "
          + ", ".join(f"`{t}` {ens.loc[t, 'hit_gain_pp']:+.2f} pp (best single family {best_single.loc[t]:+.2f})"
                      for t in s6.TARGETS) + ".",
          "- It is a forecaster, not a strategy: traded naively it still pays far more in fees than it earns (table above).",
          "- One crypto pair, six test days; AAPL is one day. The holdout week in `config/holdout.json` is untouched.", ""]
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text("\n".join(L) + "\n")


def plot(cds):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    models = [m for m in LABEL if m != "baseline_ewma"]
    colors = {"with_hawkes": "#1f5fe0", "event_time": "#6fa80f", "exact_hawkes": "#0b2a6b", "hawkes_ensemble": "#d23c2c"}
    x = np.arange(len(s6.TARGETS))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        g = cds[cds["model"] == m].set_index("target").loc[s6.TARGETS]
        ax.bar(x - 0.4 + w * (i + 0.5), g["hit_gain_pp"], w, color=colors[m], label=LABEL[m])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(s6.TARGETS)
    ax.set_ylabel("Hit-rate gain over baseline (pp)")
    ax.set_title("BTCUSDT cross-day test: mean gain over six test days")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = s6.DEFAULT_DATES
    cd, pnl, alphas, fits_x, days, trades, fits_d = btc_cross_day(dates)
    cds = summarise_cross_day(cd)
    cd.to_csv(OUT / "cross_day_by_day.csv", index=False)
    cds.to_csv(OUT / "cross_day_summary.csv", index=False)
    pnl.to_csv(OUT / "taker_pnl_by_day.csv", index=False)
    alphas.to_csv(OUT / "ridge_alpha.csv", index=False)
    (OUT / "exact_hawkes_fits.json").write_text(json.dumps(fits_x, indent=2))
    print(cds.to_string(index=False, float_format=lambda v: f"{v:.4g}"), flush=True)
    a_live = export_live(days, trades, dates, fits_d)
    print("[AAPL] walk-forward ...", flush=True)
    aapl = aapl_walk_forward()
    cd.to_csv(OUT / "cross_day_by_day.csv", index=False)
    cds.to_csv(OUT / "cross_day_summary.csv", index=False)
    pnl.to_csv(OUT / "taker_pnl_by_day.csv", index=False)
    alphas.to_csv(OUT / "ridge_alpha.csv", index=False)
    aapl.to_csv(OUT / "aapl_walk_forward.csv", index=False)
    (OUT / "exact_hawkes_fits.json").write_text(json.dumps(fits_x, indent=2))
    write_report(cds, pnl, alphas, fits_x, aapl, dates)
    plot(cds)
    print(cds.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"live ensemble ridge alpha: {a_live}")
    print(f"Wrote {OUT}/, {DOC}, {PLOT}")


if __name__ == "__main__":
    main()
