"""Step 5: out-of-sample alpha prediction and the Hawkes-vs-EWMA ablation (plan.md Phase 3, minimal).

Question: does the Hawkes order-flow signal forecast anything -- future price moves, future order
flow -- better than the EWMA baseline already in the feature table? This is independent of the
step 3 fill simulator: it fits a linear (ridge) model walk-forward on the single sample day and
reports out-of-sample R^2, hit rate, and a markout-style economic value, each with and without the
Hawkes features.

No new dependency is added (no scikit-learn): ridge is the closed-form solution on standardized,
train-fold-only statistics.
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
from numba import njit  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hawkes_diagnostics as hd  # noqa: E402

plt.style.use("seaborn-v0_8-whitegrid")

OUT_ROOT = Path("output/step5_alpha_signal")
FEATURES_CSV = Path("output/step2_hawkes/feature_table_step2_1s.csv")
TRADES_PARQUET = Path("output/step2_hawkes/trades_labeled.parquet")
DOC_PATH = Path("docs/alpha_signal.md")
PLOTS_DIR = Path("plots")

N_FOLDS = 5
WARMUP_FRAC = 0.20  # initial training window before the first OOS fold
RIDGE_ALPHA = 1.0

BASELINE_FEATURES = [
    "order_flow_imbalance", "lambda_buy_ewm", "lambda_sell_ewm",
    "imbalance_l1", "imbalance_l10", "microprice_alpha", "spread_bps", "rv_60", "trade_imbalance",
]
HAWKES_ONLY_FEATURES = ["hawkes_ofi", "lambda_buy_hawkes", "lambda_sell_hawkes"]
HAWKES_FEATURES = BASELINE_FEATURES + HAWKES_ONLY_FEATURES

PRICE_TARGETS = ["future_ret_1s", "future_ret_5s", "future_ret_10s"]
FLOW_TARGET = "future_ofi_1s"  # built here: next-second realized (buy-sell)/(buy+sell) event imbalance
TARGET_HORIZON = {"future_ret_1s": 1, "future_ret_5s": 5, "future_ret_10s": 10, FLOW_TARGET: 1}
HAWKES_COLS = ["lambda_buy_hawkes", "lambda_sell_hawkes", "hawkes_ofi"]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def add_flow_target(df: pd.DataFrame) -> pd.DataFrame:
    # Next-second order-flow imbalance target: uses only row t+1's own event counts (already
    # recorded at t+1), shifted back so it lines up with features known at row t. No feature is
    # built from it; it exists only as a target column.
    total = df["buy_events"] + df["sell_events"]
    flow_imb = np.where(total > 0, (df["buy_events"] - df["sell_events"]) / total.replace(0, np.nan), np.nan)
    df = df.copy()
    df[FLOW_TARGET] = pd.Series(flow_imb, index=df.index).shift(-1)
    return df


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    # Step 1's trade_imbalance was built from the legacy labels (most sells lost); rebuild it from
    # the corrected labels in trades_labeled.parquet.
    import step2b_causal_features as s2b
    trades = pd.read_parquet(TRADES_PARQUET)
    vols = s2b.bucket_trade_volumes(trades, df["ts"])
    for col in vols.columns:
        df[col] = vols[col].to_numpy()
    df = add_event_time_features(df, trades["ts"], trades["trade_sign"].to_numpy())
    return add_flow_target(df)


def ridge_fit(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict:
    """Closed-form ridge on standardized features (train mean/std only), intercept via centering."""
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    xs = (x_train - mu) / sigma
    y_bar = float(y_train.mean())
    beta = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ (y_train - y_bar))
    return {"mu": mu, "sigma": sigma, "beta": beta, "intercept": y_bar}


def ridge_predict(model: dict, x: np.ndarray) -> np.ndarray:
    return model["intercept"] + ((x - model["mu"]) / model["sigma"]) @ model["beta"]


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    return ridge_predict(ridge_fit(x_train, y_train, alpha), x_test)


def walk_forward_folds(n: int, n_folds: int, warmup_frac: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window, chronological: train is everything strictly before the test block."""
    warmup_end = int(n * warmup_frac)
    remaining = n - warmup_end
    block = remaining // n_folds
    folds = []
    for i in range(n_folds):
        test_start = warmup_end + i * block
        test_end = n if i == n_folds - 1 else warmup_end + (i + 1) * block
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        folds.append((train_idx, test_idx))
    return folds


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan, np.nan
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    h = float(stats.t.ppf(0.5 + level / 2, len(x) - 1) * se)
    return m, se, m - h, m + h


def hawkes_one_step(params: np.ndarray, yb: np.ndarray, ys: np.ndarray, index=None) -> pd.DataFrame:
    """One-step-ahead intensities lambda_{t+1|t} (counts up to t) and their OFI, for given parameters."""
    lam_b_t, lam_s_t = hd.discrete_intensities(params, yb, ys)
    mu_b, mu_s, a_bb, a_bs, a_ss, a_sb, beta = params
    lam_b = np.maximum(mu_b + beta * lam_b_t + a_bb * yb + a_bs * ys, 1e-8)
    lam_s = np.maximum(mu_s + beta * lam_s_t + a_ss * ys + a_sb * yb, 1e-8)
    denom = lam_b + lam_s
    ofi = np.where(denom > 0, (lam_b - lam_s) / denom, 0.0)
    return pd.DataFrame({"lambda_buy_hawkes": lam_b, "lambda_sell_hawkes": lam_s, "hawkes_ofi": ofi}, index=index)


def hawkes_features_fitted_on(df: pd.DataFrame, end: int) -> pd.DataFrame:
    """One-step-ahead Hawkes intensities lambda_{t+1|t} for every row, parameters fitted on rows [0, end).

    Row t's features are known at the end of bucket t, and the EWMA baseline already includes
    bucket t's counts. Step 2's lambda_t is the forecast *for* bucket t (counts up to t-1), so
    using it would leave the Hawkes features one second staler than the baseline. lambda_{t+1}
    uses counts up to t only, so it is the like-for-like feature. Fitting the parameters on the
    training window keeps test-fold data out of the Hawkes features.
    """
    yb = df["buy_events"].to_numpy(dtype=float)
    ys = df["sell_events"].to_numpy(dtype=float)
    fit = hd.fit_discrete(yb[:end], ys[:end])
    return hawkes_one_step(fit["x"], yb, ys, index=df.index)


EVENT_HALF_LIVES_S = (0.1, 1.0, 10.0)


@njit(cache=True)
def _decayed_event_sums(ev_t, ev_side, grid_t, omegas):
    """R[g, k, side] = sum over events strictly before grid_t[g] of omega_k * exp(-omega_k (grid_t[g] - t_i))."""
    K = omegas.shape[0]
    out = np.zeros((grid_t.shape[0], K, 2))
    R = np.zeros((K, 2))
    t_last = 0.0
    j = 0
    n_ev = ev_t.shape[0]
    for g in range(grid_t.shape[0]):
        while j < n_ev and ev_t[j] < grid_t[g]:
            for k in range(K):
                decay = np.exp(-omegas[k] * (ev_t[j] - t_last))
                R[k, 0] *= decay
                R[k, 1] *= decay
                R[k, ev_side[j]] += omegas[k]
            t_last = ev_t[j]
            j += 1
        for k in range(K):
            decay = np.exp(-omegas[k] * (grid_t[g] - t_last))
            out[g, k, 0] = R[k, 0] * decay
            out[g, k, 1] = R[k, 1] * decay
    return out


def event_time_feature_names(half_lives=EVENT_HALF_LIVES_S) -> list[str]:
    names = []
    for h in half_lives:
        tag = f"{h:g}s".replace(".", "p")
        names += [f"et_log_buy_{tag}", f"et_log_sell_{tag}", f"et_ofi_{tag}"]
    return names


def add_event_time_features(df: pd.DataFrame, ev_ts: pd.Series, ev_sign: np.ndarray,
                            half_lives=EVENT_HALF_LIVES_S) -> pd.DataFrame:
    """Multi-scale event-time order-flow intensities sampled at each bucket's end.

    Uses individual event timestamps, so it sees the millisecond-scale clustering that 1 s buckets
    average away. Each scale is an exponential kernel with the given half-life; a linear model on
    these columns is a learned sum-of-exponentials Hawkes intensity. Events sharing a timestamp
    and side (one order sweeping several resting orders) count once. Row t uses events strictly
    before the end of bucket t, the same information set as the EWMA and one-step-ahead Hawkes.
    """
    ev = pd.DataFrame({"ts": ev_ts.to_numpy(), "side": (np.asarray(ev_sign) < 0).astype(np.int64)})
    ev = ev.drop_duplicates().sort_values("ts", kind="stable")
    origin = df["ts"].iloc[0]
    ev_t = (ev["ts"] - origin).dt.total_seconds().to_numpy()
    grid_t = (df["ts"] - origin).dt.total_seconds().to_numpy() + 1.0
    omegas = np.log(2.0) / np.asarray(half_lives, dtype=float)
    R = _decayed_event_sums(ev_t, ev["side"].to_numpy(), grid_t, omegas)
    df = df.copy()
    names = event_time_feature_names(half_lives)
    for k in range(len(half_lives)):
        b, s = R[:, k, 0], R[:, k, 1]
        df[names[3 * k]] = np.log1p(b)
        df[names[3 * k + 1]] = np.log1p(s)
        df[names[3 * k + 2]] = np.where(b + s > 1e-12, (b - s) / (b + s), 0.0)
    return df


def diebold_mariano_hac(d: np.ndarray, lag: int) -> tuple[float, float]:
    """Diebold-Mariano test of mean(d) = 0 with a Newey-West (Bartlett) long-run variance."""
    n = len(d)
    if n < lag + 2:
        return float("nan"), float("nan")
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt(lrv / n)
    return float(t), float(2.0 * stats.norm.sf(abs(t)))


BONFERRONI_P = 0.05 / 4  # four targets per instrument


def verdict(r: pd.Series) -> str:
    """robust: both paired tests clear Bonferroni; suggestive: either below 0.05; else null."""
    p_fold, p_dm, m = r["p_value_fold_level"], r["p_value_dm_hac"], r["mean_se_diff_fold_level"]
    alt = {"with_hawkes": "Hawkes", "event_time": "event-time"}.get(r.get("alt_model", "with_hawkes"), "alt")
    who = f"{alt} better" if m > 0 else "EWMA better"
    if p_fold < BONFERRONI_P and p_dm < BONFERRONI_P:
        return f"robust ({who})"
    if min(p_fold, p_dm) < 0.05:
        return f"suggestive ({who})"
    return "null"


def toxicity_regime_labels(df: pd.DataFrame, buy_col: str = "lambda_buy_ewm", sell_col: str = "lambda_sell_ewm") -> pd.Series:
    """Tertiles of |EWMA buy - sell intensity|, same definition step4_analysis.toxicity_regimes() uses."""
    proxy = (df[buy_col] - df[sell_col]).abs()
    q1, q2 = proxy.quantile(0.33), proxy.quantile(0.66)
    return pd.Series(np.where(proxy <= q1, "low", np.where(proxy <= q2, "medium", "high")), index=df.index)


def _fold_metrics(pred: np.ndarray, y_te: np.ndarray, cost_te: np.ndarray, hit_min_abs: float = 0.0) -> dict:
    ss_res = float(np.sum((y_te - pred) ** 2))
    ss_tot_zero = float(np.sum(y_te ** 2))  # OOS R^2 against the "predict zero" benchmark
    r2 = 1.0 - ss_res / ss_tot_zero if ss_tot_zero > 0 else np.nan
    # Zero targets are common (AAPL: ~48% of 1 s returns are exactly 0 at a 1-cent tick), and no sign
    # prediction can "hit" them, so hit rate is measured on non-zero targets only.
    # hit_min_abs > 0 also drops moves no bigger than a bid-ask bounce (used for last-trade prices).
    scored = (np.sign(pred) != 0) & (np.abs(y_te) > hit_min_abs)
    hit = float((np.sign(pred[scored]) == np.sign(y_te[scored])).mean()) if scored.any() else np.nan
    gross = np.sign(pred) * y_te
    net = gross - cost_te * np.abs(np.sign(pred))
    return {
        "oos_r2": r2, "hit_rate": hit,
        "mean_gross_pnl_per_row": float(gross.mean()), "mean_net_pnl_per_row": float(net.mean()),
    }


def evaluate_pair(df: pd.DataFrame, baseline_feats: list[str], hawkes_feats: list[str],
                   target: str, cost_col: str | None,
                   regime_col: tuple[str, str] | None = None, refit_hawkes: bool = True,
                   alt_label: str = "with_hawkes", hit_min_abs: float = 0.0) -> dict:
    """Fit baseline and hawkes models on identical rows/folds so their errors are paired.

    With refit_hawkes, the Hawkes parameters are re-estimated on each fold's training window
    (hawkes_features_fitted_on), so no test-fold data reaches the Hawkes features; the EWMA
    baseline has no fitted parameters, so it needs no equivalent. Returns fold-level metrics for
    both models, a paired per-fold squared-error difference (positive = hawkes lower error), a
    Diebold-Mariano test on the pooled per-row differences with a Newey-West variance (serial
    correlation and overlapping multi-step targets make a plain t-test far too optimistic), and,
    if regime_col is given, the same difference split by toxicity-regime tertile.
    """
    cols = list(dict.fromkeys(hawkes_feats + baseline_feats + [target] + ([cost_col] if cost_col else [])))
    valid = df[cols].dropna()
    valid_pos = df.index.get_indexer(valid.index)
    x_base = valid[baseline_feats].to_numpy(dtype=float)
    x_hawkes = valid[hawkes_feats].to_numpy(dtype=float)
    hawkes_pos = [i for i, c in enumerate(hawkes_feats) if c in HAWKES_COLS]
    y_all = valid[target].to_numpy(dtype=float)
    cost_all = valid[cost_col].to_numpy(dtype=float) / 2.0 / 10000.0 if cost_col else np.zeros(len(valid))
    regimes = None
    if regime_col is not None:
        regimes = toxicity_regime_labels(df.loc[valid.index], *regime_col).to_numpy()

    folds = walk_forward_folds(len(valid), N_FOLDS, WARMUP_FRAC)
    fold_rows, se_diff_rows, regime_rows = [], [], []
    for fold_id, (tr, te) in enumerate(folds):
        if len(tr) < 50 or len(te) < 10:
            continue
        x_h = x_hawkes
        if refit_hawkes and hawkes_pos:
            refit = hawkes_features_fitted_on(df, end=int(valid_pos[tr].max()) + 1).iloc[valid_pos]
            x_h = x_hawkes.copy()
            for i in hawkes_pos:
                x_h[:, i] = refit[hawkes_feats[i]].to_numpy()
        pred_base = ridge_fit_predict(x_base[tr], y_all[tr], x_base[te], RIDGE_ALPHA)
        pred_hawkes = ridge_fit_predict(x_h[tr], y_all[tr], x_h[te], RIDGE_ALPHA)
        y_te = y_all[te]

        m_base = _fold_metrics(pred_base, y_te, cost_all[te], hit_min_abs)
        m_hawkes = _fold_metrics(pred_hawkes, y_te, cost_all[te], hit_min_abs)
        fold_rows.append({"fold": fold_id, "model": "baseline_ewma", **m_base})
        fold_rows.append({"fold": fold_id, "model": alt_label, **m_hawkes})

        se_base = (y_te - pred_base) ** 2
        se_hawkes = (y_te - pred_hawkes) ** 2
        se_diff = se_base - se_hawkes  # positive => hawkes has lower squared error
        se_diff_rows.append(se_diff)

        if regimes is not None:
            reg_te = regimes[te]
            for label in ["low", "medium", "high"]:
                mask = reg_te == label
                if mask.sum() < 5:
                    continue
                regime_rows.append({
                    "fold": fold_id, "regime": label, "n": int(mask.sum()),
                    "mean_se_diff": float(se_diff[mask].mean()),
                })

    fold_df = pd.DataFrame(fold_rows)
    fold_se_diff_means = np.array([float(d.mean()) for d in se_diff_rows])
    pooled_se_diff = np.concatenate(se_diff_rows) if se_diff_rows else np.array([])

    def t_test(x: np.ndarray) -> tuple[float, float]:
        if len(x) < 2 or np.all(x == x[0]):
            return float("nan"), float("nan")
        res = stats.ttest_1samp(x, 0.0)
        return float(res.statistic), float(res.pvalue)

    fold_stat, fold_p = t_test(fold_se_diff_means)
    # Bartlett lag: the target's own overlap (h - 1) plus a margin for short-range dependence.
    hac_lag = TARGET_HORIZON.get(target, 1) + 10
    pooled_stat, pooled_p = diebold_mariano_hac(pooled_se_diff, hac_lag)
    m, se, lo, hi = mean_ci(fold_se_diff_means)

    regime_df = pd.DataFrame(regime_rows)
    regime_summary = None
    if len(regime_df):
        regime_summary = (
            regime_df.groupby("regime")["mean_se_diff"]
            .apply(lambda x: pd.Series(mean_ci(x.to_numpy()), index=["mean", "se", "ci95_lo", "ci95_hi"]))
            .unstack()
            .reindex(["low", "medium", "high"])
            .reset_index()
        )

    return {
        "fold_df": fold_df,
        "paired_fold_mean_se_diff": m, "paired_fold_ci95_lo": lo, "paired_fold_ci95_hi": hi,
        "paired_fold_t_stat": fold_stat, "paired_fold_p_value": fold_p, "n_folds": len(fold_se_diff_means),
        "paired_pooled_t_stat": pooled_stat, "paired_pooled_p_value": pooled_p, "n_pooled_rows": len(pooled_se_diff),
        "regime_summary": regime_summary,
    }


def summarize(fold_df: pd.DataFrame, label: str) -> dict:
    sub = fold_df[fold_df["model"] == label]
    out = {"model": label, "n_folds": len(sub)}
    for metric in ["oos_r2", "hit_rate", "mean_gross_pnl_per_row", "mean_net_pnl_per_row"]:
        m, se, lo, hi = mean_ci(sub[metric].dropna().to_numpy())
        out[f"{metric}_mean"] = m
        out[f"{metric}_ci95_lo"] = lo
        out[f"{metric}_ci95_hi"] = hi
    return out


def run_target(df: pd.DataFrame, target: str, cost_col: str | None,
                baseline_feats: list[str] = BASELINE_FEATURES, hawkes_feats: list[str] = HAWKES_FEATURES,
                regime_col: tuple[str, str] | None = ("lambda_buy_ewm", "lambda_sell_ewm"),
                alt_label: str = "with_hawkes", hit_min_abs: float = 0.0) -> dict:
    result = evaluate_pair(df, baseline_feats, hawkes_feats, target, cost_col, regime_col, alt_label=alt_label,
                           hit_min_abs=hit_min_abs)
    fold_df = result["fold_df"].copy()
    fold_df["target"] = target
    summary_rows = [
        {"target": target, **summarize(fold_df, "baseline_ewma")},
        {"target": target, **summarize(fold_df, alt_label)},
    ]
    paired_row = {
        "target": target,
        "alt_model": alt_label,
        "n_folds": result["n_folds"],
        "mean_se_diff_fold_level": result["paired_fold_mean_se_diff"],
        "ci95_lo": result["paired_fold_ci95_lo"], "ci95_hi": result["paired_fold_ci95_hi"],
        "t_stat_fold_level": result["paired_fold_t_stat"], "p_value_fold_level": result["paired_fold_p_value"],
        "dm_hac_stat": result["paired_pooled_t_stat"], "p_value_dm_hac": result["paired_pooled_p_value"],
        "n_pooled_rows": result["n_pooled_rows"],
    }
    regime_df = result["regime_summary"]
    if regime_df is not None:
        regime_df = regime_df.copy()
        regime_df["target"] = target
    return {
        "fold_df": fold_df, "summary": pd.DataFrame(summary_rows),
        "paired": paired_row, "regime": regime_df,
    }


def write_report(summary: pd.DataFrame, paired: pd.DataFrame, regime: pd.DataFrame, n_rows: int, n_folds_used: int,
                 extra: list[str] | None = None):
    lines = [
        "# Alpha-signal ablation (plan.md Phase 3, minimal)",
        "",
        "Generated by `scripts/step5_alpha_signal.py`. Walk-forward (expanding window, chronological, "
        f"{N_FOLDS} folds, {WARMUP_FRAC:.0%} warm-up) ridge regression on AAPL 2024-06-03, {n_rows} "
        "one-second rows. One day, one instrument -- treat as a first pass, not a validated edge.",
        "",
        "**Question:** does the Hawkes order-flow signal (`hawkes_ofi`, `lambda_*_hawkes`) forecast "
        "future price moves or future order flow better than the EWMA baseline already in the feature "
        "table (`order_flow_imbalance`, `lambda_*_ewm`)? Two models are compared at each target: "
        "`baseline_ewma` (9 microstructure/EWMA features) vs `with_hawkes` (same 9 plus the 3 Hawkes "
        "features). Both are plain ridge (alpha=1, not tuned) on standardized features; no look-ahead: "
        "train-fold mean/std only, each fold trains strictly on rows before its test block, and the Hawkes "
        "parameters are re-estimated on each fold's training window (fitting them on the whole day, as "
        "step 2 does, would hand the Hawkes features information from the test folds that the "
        "parameter-free EWMA does not get).",
        "",
        "## Results",
        "",
        "OOS R^2 is against the \"predict zero\" benchmark (standard for short-horizon return forecasting, "
        "since the unconditional mean return is ~0). Hit rate is scored on rows whose target is non-zero "
        "(about half of AAPL's 1 s returns are exactly zero at a 1-cent tick, and no sign call can hit "
        "those). Economic value is mean(sign(prediction) * realized return) per row, trading every row; "
        "\"net\" subtracts a half-spread crossing cost. Trading every second at the spread is not a "
        "strategy anyone would run, so net PnL is dominated by cost for both models; read it as a "
        "relative comparison only. It is not defined for the order-flow target. CIs are a t-interval "
        "over the folds actually used "
        f"({n_folds_used} per target) -- small n, treat the width as a floor on the uncertainty, not the "
        "whole of it (one day's worth of folds is not independent data).",
        "",
    ]
    for target in summary["target"].unique():
        t = summary[summary["target"] == target]
        lines.append(f"### `{target}`")
        lines.append("")
        lines.append("| Model | OOS R² (95% CI) | Hit rate (95% CI) | Net PnL/row (95% CI) |")
        lines.append("|---|---|---|---|")
        for _, r in t.iterrows():
            pnl = "n/a" if target == FLOW_TARGET else (
                f"{r['mean_net_pnl_per_row_mean']:.2e} "
                f"[{r['mean_net_pnl_per_row_ci95_lo']:.2e}, {r['mean_net_pnl_per_row_ci95_hi']:.2e}]")
            lines.append(
                f"| {r['model']} | {r['oos_r2_mean']:.4f} [{r['oos_r2_ci95_lo']:.4f}, {r['oos_r2_ci95_hi']:.4f}] "
                f"| {r['hit_rate_mean']:.3f} [{r['hit_rate_ci95_lo']:.3f}, {r['hit_rate_ci95_hi']:.3f}] "
                f"| {pnl} |"
            )
        lines.append("")

    lines += [
        "## Paired significance test",
        "",
        "Comparing two overlapping confidence intervals is a weak test (it is biased toward \"no "
        "difference\"). This tests the per-row squared-error difference directly: both models are fit "
        "on identical rows and identical folds, so `se_diff = (baseline error)^2 - (hawkes error)^2` is "
        "a paired observation -- positive means Hawkes had the lower error on that row. Two versions are "
        "reported: **fold-level** (one mean per fold, t-test with n = folds -- conservative, few degrees "
        "of freedom) and **Diebold-Mariano** (all OOS rows pooled, with a Newey-West long-run variance, "
        "Bartlett lag = horizon + 10). A plain pooled t-test would be badly over-confident here: adjacent "
        "1 s rows are serially correlated and the 5 s / 10 s targets overlap by construction.",
        "",
        "| Target | Mean se_diff (fold, 95% CI) | Fold t-test p | Diebold-Mariano (HAC) p | n pooled rows | Verdict |",
        "|---|---|---|---:|---:|---|",
    ]
    for _, r in paired.iterrows():
        lines.append(
            f"| `{r['target']}` | {r['mean_se_diff_fold_level']:.3e} "
            f"[{r['ci95_lo']:.3e}, {r['ci95_hi']:.3e}] | {r['p_value_fold_level']:.3f} | "
            f"{r['p_value_dm_hac']:.3f} | {r['n_pooled_rows']} | {verdict(r)} |"
        )
    lines += [
        "",
        "Four targets are tested, so the Bonferroni threshold for a family-wise 5% level is p < 0.0125. "
        "**robust** = both tests clear it; **suggestive** = at least one test is below 0.05; **null** "
        "otherwise. A negative mean means the Hawkes features made the forecast *worse* (extra, noisier "
        "regressors).",
        "",
        "## Does Hawkes help more in specific regimes?",
        "",
        "Same paired squared-error difference (positive = Hawkes better), split by the toxicity-regime "
        "tertiles `step4_analysis.toxicity_regimes()` already uses (`|lambda_buy_ewm - lambda_sell_ewm|` "
        "tertiles: low/medium/high order-flow imbalance). The Hawkes calibration report found the fit "
        "itself is unstable across the day -- if Hawkes adds value only during bursty/high-imbalance "
        "periods, averaging over the whole day would hide it; this checks that directly rather than "
        "assuming a flat effect. (The tertile cut-offs use the whole day; they only label rows for this "
        "breakdown and never enter a model.)",
        "",
        "| Target | Regime | Mean se_diff over folds (95% CI) |",
        "|---|---|---|",
    ]
    if regime is not None and len(regime):
        for _, r in regime.iterrows():
            lines.append(
                f"| `{r['target']}` | {r['regime']} | {r['mean']:.3e} [{r['ci95_lo']:.3e}, {r['ci95_hi']:.3e}] |"
            )
    else:
        lines.append("| -- | -- | no regime data |")
    lines += [
        "",
        "## Reading this table",
        "",
        "If `with_hawkes`'s CI does not clearly separate from `baseline_ewma`'s on OOS R^2 and net PnL, "
        "**and** the paired test above does not reject zero, the Hawkes features are not shown to add "
        "predictive value beyond the EWMA baseline on this day. That is consistent with the corrected "
        "engine comparison (`output/step4b_fixed/step4b_report.txt`), where Hawkes and baseline quoting "
        "are statistically indistinguishable once fills are replayed from real trades. A result that only "
        "shows up in the high-toxicity regime would be a materially different finding from a flat null, "
        "and is what the regime table above is for.",
        "",
        *(extra or []),
        "## Literature check: is an HJB + Hawkes optimal-control reformulation the next step?",
        "",
        "Checked against Jain, Firoozye, Kochems & Treleaven, *\"An Impulse Control Approach to Market "
        "Making in a Hawkes LOB Market\"* (arXiv:2510.26438v2, Oct 2025), the most directly relevant "
        "recent paper attempting exactly this. Findings relevant here, read directly from the paper "
        "rather than taken from its abstract:",
        "",
        "- They write the HJB quasi-variational inequality for a Hawkes-driven market maker, then state "
        "a closed-form solution is infeasible. Their numerical PDE solver (Deep Galerkin) does not "
        "converge for the full multivariate Hawkes model; their headline result instead comes from a "
        "PPO reinforcement-learning agent trained in their own simulator.",
        "- No real market data is used anywhere in the paper -- validation is against their own "
        "simulator's stylized facts, not a historical dataset.",
        "- No comparison against Avellaneda-Stoikov or any standard heuristic quoting strategy is "
        "reported -- only against their own rule-based baseline.",
        "- No confidence intervals, seed counts, or significance tests accompany the headline Sharpe "
        "ratios.",
        "",
        "Conclusion: a validated, closed-form HJB+Hawkes reformulation is not a two-day task -- it was "
        "not a converged result for a funded team attempting exactly this. The more defensible use of "
        "limited time is the direct empirical test above: does the signal predict anything, checked "
        "ourselves, rather than assumed from a paper's framing. An HJB/RL reformulation remains the "
        "right next step (plan.md Phase 6a) **conditional on** this ablation showing the Hawkes signal "
        "adds out-of-sample value; optimizing control of a signal not shown to predict anything would "
        "not be a meaningful next step.",
        "",
        "## Cross-asset check",
        "",
        "`scripts/step6_binance_validation.py` / `docs/binance_validation.md` reruns this exact walk-forward "
        "test on an independent instrument (Binance BTCUSDT) over several days, within each day and across "
        "days, to check whether this null result is AAPL-specific. It is not: see that report.",
        "",
        "## Limitations",
        "",
        "- One day, one instrument -- folds are time-adjacent, not independent; CIs understate true "
        "uncertainty.",
        "- Ridge alpha=1.0 is fixed, not cross-validated.",
        "- The economic-value cost assumption (half the quoted spread) is a simplification, not a fitted "
        "fill model; it is not the step 3 engine's fill probabilities.",
        "- `future_ofi_1s` is a realized one-second-ahead order-flow imbalance, not a tradeable quantity "
        "by itself; it isolates whether Hawkes intensities forecast flow, separately from price.",
    ]
    DOC_PATH.parent.mkdir(exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n")


def plot_oos_r2(summary_df: pd.DataFrame):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    targets = list(summary_df["target"].unique())
    x = np.arange(len(targets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, (label, color) in zip([-1, 1], [("baseline_ewma", "#4C72B0"), ("with_hawkes", "#DD8452")]):
        sub = summary_df[summary_df["model"] == label].set_index("target").loc[targets]
        means = sub["oos_r2_mean"].to_numpy()
        lo = means - sub["oos_r2_ci95_lo"].to_numpy()
        hi = sub["oos_r2_ci95_hi"].to_numpy() - means
        ax.bar(x + offset * width / 2, means, width, label=label, color=color,
               yerr=[lo, hi], capsize=4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15)
    ax.set_ylabel("Out-of-sample R² (vs. predict-zero)")
    ax.set_title("Hawkes vs. EWMA ablation: walk-forward OOS R² by target (95% CI)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_alpha_signal_oos_r2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_all_targets(df: pd.DataFrame, baseline_feats: list[str], alt_feats: list[str], alt_label: str,
                    price_cost_col: str | None, regime_col=("lambda_buy_ewm", "lambda_sell_ewm")):
    folds, summary, paired, regime = [], [], [], []
    for target, cost in [(t, price_cost_col) for t in PRICE_TARGETS] + [(FLOW_TARGET, None)]:
        r = run_target(df, target, cost, baseline_feats, alt_feats, regime_col, alt_label)
        folds.append(r["fold_df"])
        summary.append(r["summary"])
        paired.append(r["paired"])
        if r["regime"] is not None:
            regime.append(r["regime"])
    return (pd.concat(folds, ignore_index=True), pd.concat(summary, ignore_index=True), pd.DataFrame(paired),
            pd.concat(regime, ignore_index=True) if regime else None)


def event_time_section(summary: pd.DataFrame, paired: pd.DataFrame, instrument: str) -> list[str]:
    hl = ", ".join(f"{h:g} s" for h in EVENT_HALF_LIVES_S)
    lines = [
        "## Event-time multi-scale features",
        "",
        "The discrete Hawkes model sees only 1 s counts, but the exact fit in `docs/hawkes_calibration.md` puts "
        "most of the excitation at millisecond scale. This model replaces the three Hawkes features with "
        f"exponentially decayed counts of buy and sell aggressor events at half-lives of {hl}, computed from "
        "the individual trade timestamps and sampled at each bucket's end (log intensity per side, plus an "
        "imbalance, per scale). A linear model on these is a learned sum-of-exponentials Hawkes intensity. "
        "Same baseline, folds and tests as above.",
        "",
        f"| Target ({instrument}) | R² base | R² +event-time | Hit base | Hit +event-time | Fold p | DM p | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, p in paired.iterrows():
        s = summary[summary["target"] == p["target"]].set_index("model")
        b, e = s.loc["baseline_ewma"], s.loc["event_time"]
        lines.append(f"| `{p['target']}` | {b['oos_r2_mean']:.4f} | {e['oos_r2_mean']:.4f} | {b['hit_rate_mean']:.3f} | "
                     f"{e['hit_rate_mean']:.3f} | {p['p_value_fold_level']:.3f} | {p['p_value_dm_hac']:.3g} | "
                     f"{verdict(p)} |")
    lines.append("")
    return lines


def main():
    ensure_dir(OUT_ROOT)
    df = load_features()

    fold_df, summary_df, paired_df, regime_df = run_all_targets(
        df, BASELINE_FEATURES, HAWKES_FEATURES, "with_hawkes", "spread_bps")
    et_feats = BASELINE_FEATURES + event_time_feature_names()
    et_folds, et_summary, et_paired, _ = run_all_targets(df, BASELINE_FEATURES, et_feats, "event_time",
                                                         "spread_bps", regime_col=None)
    fold_df.to_csv(OUT_ROOT / "step5_fold_metrics.csv", index=False)
    summary_df.to_csv(OUT_ROOT / "step5_ablation_summary.csv", index=False)
    paired_df.to_csv(OUT_ROOT / "step5_paired_significance.csv", index=False)
    et_summary.to_csv(OUT_ROOT / "step5_event_time_summary.csv", index=False)
    et_paired.to_csv(OUT_ROOT / "step5_event_time_paired.csv", index=False)
    if regime_df is not None:
        regime_df.to_csv(OUT_ROOT / "step5_regime_breakdown.csv", index=False)

    n_folds_used = int(fold_df.groupby("target")["fold"].nunique().min())
    write_report(summary_df, paired_df, regime_df, n_rows=len(df), n_folds_used=n_folds_used,
                 extra=event_time_section(et_summary, et_paired, "AAPL"))
    plot_oos_r2(summary_df)

    meta = {
        "n_rows": len(df), "n_folds_requested": N_FOLDS, "warmup_frac": WARMUP_FRAC,
        "ridge_alpha": RIDGE_ALPHA, "baseline_features": BASELINE_FEATURES, "hawkes_features": HAWKES_FEATURES,
        "price_targets": PRICE_TARGETS, "flow_target": FLOW_TARGET,
    }
    (OUT_ROOT / "step5_run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {OUT_ROOT}/step5_fold_metrics.csv, step5_ablation_summary.csv, step5_run_meta.json")
    print(f"Wrote {DOC_PATH}")


if __name__ == "__main__":
    main()
