"""Hawkes calibration diagnostics for the step 2 model.

Occupancy, null models, stability, bucket-width and sub-window refits of the
discrete fit, plus an exact continuous-time fit (1-3 exponentials) with a
time-rescaling KS test. Writes output/hawkes_diagnostics/ and docs/hawkes_calibration.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from numba import njit
from scipy import optimize, signal, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import step2_trade_label_and_hawkes as s2  # noqa: E402

STEP2 = ROOT / "output" / "step2_hawkes"
OUTDIR = ROOT / "output" / "hawkes_diagnostics"
REPORT = ROOT / "docs" / "hawkes_calibration.md"
WIDTHS_S = [0.25, 0.5, 1.0, 2.0, 5.0]
N_WINDOWS = 4
PARAM_NAMES = ["mu_buy", "mu_sell", "alpha_bb", "alpha_bs", "alpha_ss", "alpha_sb", "beta"]


# ---------------------------------------------------------------- discrete model

def discrete_intensities(params: np.ndarray, yb: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same recursion as step 2 (lambda[0] = mu, lambda[t] uses counts up to t-1), via lfilter.

    Step 2's 1e-8 floor never binds for non-negative parameters, so it is omitted.
    """
    mu_b, mu_s, a_bb, a_bs, a_ss, a_sb, beta = params
    xb = np.empty_like(yb)
    xs = np.empty_like(ys)
    xb[0], xs[0] = mu_b, mu_s
    xb[1:] = mu_b + a_bb * yb[:-1] + a_bs * ys[:-1]
    xs[1:] = mu_s + a_ss * ys[:-1] + a_sb * yb[:-1]
    lam_b = signal.lfilter([1.0], [1.0, -beta], xb)
    lam_s = signal.lfilter([1.0], [1.0, -beta], xs)
    return np.maximum(lam_b, 1e-8), np.maximum(lam_s, 1e-8)


def poisson_loglik(y: np.ndarray, lam: np.ndarray) -> float:
    """Poisson log-likelihood without the log(y!) constant (as in step 2)."""
    return float(np.sum(y * np.log(lam + 1e-12) - lam))


def negloglik_fast(params: np.ndarray, yb: np.ndarray, ys: np.ndarray) -> float:
    """Step 2's objective (same penalties), with the vectorised recursion."""
    lam_b, lam_s = discrete_intensities(params, yb, ys)
    penalty = 0.0
    beta = params[-1]
    if beta >= 0.999:
        penalty += 1e6 * (beta - 0.999 + 1e-6)
    if beta < 0:
        penalty += 1e6 * abs(beta)
    excitation_sum = params[2] + params[3] + params[4] + params[5]
    if excitation_sum > 5.0:
        penalty += 1e3 * (excitation_sum - 5.0)
    return -(poisson_loglik(yb, lam_b) + poisson_loglik(ys, lam_s)) + penalty


def fit_discrete(yb: np.ndarray, ys: np.ndarray, cross: bool = True) -> dict:
    """Step 2's fit (same x0, bounds, optimizer). cross=False pins alpha_bs = alpha_sb = 0."""
    x0 = np.array([max(yb.mean() * 0.5, 1e-3), max(ys.mean() * 0.5, 1e-3), 0.10, 0.03, 0.10, 0.03, 0.60])
    bounds = [(1e-6, 50.0), (1e-6, 50.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.0, 5.0), (0.0, 0.999)]
    if not cross:
        x0[[3, 5]] = 0.0
        bounds[3] = bounds[5] = (0.0, 0.0)
    res = optimize.minimize(negloglik_fast, x0, args=(yb, ys), method="L-BFGS-B", bounds=bounds,
                            options={"maxiter": 500})
    p = res.x
    lam_b, lam_s = discrete_intensities(p, yb, ys)
    ll = poisson_loglik(yb, lam_b) + poisson_loglik(ys, lam_s)
    return {"params": dict(zip(PARAM_NAMES, map(float, p))), "x": p, "loglik": ll, "success": bool(res.success),
            "message": str(res.message), "n_free": 7 if cross else 5, **stability(p)}


def stability(p: np.ndarray) -> dict:
    """Companion matrix beta*I + alpha (one-step mean dynamics) and branching matrix alpha / (1 - beta)."""
    alpha = np.array([[p[2], p[3]], [p[5], p[4]]])  # rows: target (buy, sell); cols: source (buy, sell)
    beta = p[6]
    comp = beta * np.eye(2) + alpha
    branch = alpha / (1.0 - beta)
    return {"spectral_radius_companion": float(max(abs(np.linalg.eigvals(comp)))),
            "spectral_radius_branching": float(max(abs(np.linalg.eigvals(branch)))),
            "branching_matrix": branch.round(6).tolist(),
            "half_life_buckets": float(np.log(0.5) / np.log(beta)) if 0 < beta < 1 else None}


def counts_at_width(trades: pd.DataFrame, width_s: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    edges = pd.date_range(start, end, freq=pd.Timedelta(seconds=width_s))
    b = np.histogram(trades.loc[trades["trade_sign"] == 1, "ts"].astype("int64"), edges.astype("int64"))[0]
    s = np.histogram(trades.loc[trades["trade_sign"] == -1, "ts"].astype("int64"), edges.astype("int64"))[0]
    return pd.DataFrame({"ts": edges[:-1], "buy_events": b.astype(float), "sell_events": s.astype(float)})


def occupancy(y: np.ndarray) -> dict:
    n = len(y)
    return {"0": float((y == 0).sum() / n), "1": float((y == 1).sum() / n), "2": float((y == 2).sum() / n),
            "3+": float((y >= 3).sum() / n), "mean": float(y.mean()), "var_over_mean": float(y.var() / y.mean())}


# ---------------------------------------------------------- continuous-time model

@njit(cache=True)
def _cont_loglik(times, types, T, mu, alpha, omega, want_resid):
    """Bivariate Hawkes, kernel phi_ij(t) = sum_k alpha[k,i,j] * omega[k] * exp(-omega[k] t).

    alpha[k, i, j]: excitation of type i by an event of type j through decay component k;
    with this normalisation sum_k alpha[k, i, j] is the branching ratio.
    Returns (loglik, per-event compensator increments since the previous event of the same type).
    """
    K = omega.shape[0]
    R = np.zeros((K, 2))          # R[k, j] = sum over past type-j events of omega_k exp(-omega_k (t - t_j))
    comp_since = np.zeros(2)      # compensator of each type accumulated since its last event
    resid = np.zeros(times.shape[0])
    ll = 0.0
    t_prev = 0.0
    for n in range(times.shape[0]):
        t = times[n]
        dt = t - t_prev
        for i in range(2):
            inc = mu[i] * dt
            for k in range(K):
                decay = np.exp(-omega[k] * dt)
                for j in range(2):
                    inc += alpha[k, i, j] * R[k, j] * (1.0 - decay) / omega[k]
            comp_since[i] += inc
        for k in range(K):
            decay = np.exp(-omega[k] * dt)
            for j in range(2):
                R[k, j] *= decay
        i = types[n]
        lam = mu[i]
        for k in range(K):
            for j in range(2):
                lam += alpha[k, i, j] * R[k, j]
        ll += np.log(lam)
        if want_resid:
            resid[n] = comp_since[i]
        comp_since[i] = 0.0
        for k in range(K):
            R[k, i] += omega[k]
        t_prev = t
    # compensator up to T
    total = (mu[0] + mu[1]) * T
    for n in range(times.shape[0]):
        j = types[n]
        for k in range(K):
            tail = 1.0 - np.exp(-omega[k] * (T - times[n]))
            total += (alpha[k, 0, j] + alpha[k, 1, j]) * tail
    return ll - total, resid


def _unpack(theta: np.ndarray, K: int):
    mu = np.exp(theta[:2])
    alpha = np.exp(theta[2:2 + 4 * K]).reshape(K, 2, 2)
    omega = np.exp(theta[2 + 4 * K:])
    return mu, alpha, omega


def fit_continuous(times: np.ndarray, types: np.ndarray, T: float, K: int, omega0: list[float]) -> dict:
    """Exact MLE (log-parameterised, L-BFGS-B) of the K-exponential bivariate Hawkes process."""
    rate = np.array([(types == 0).sum(), (types == 1).sum()]) / T
    theta0 = np.concatenate([np.log(0.3 * rate), np.log(np.full(4 * K, 0.6 / (2 * K))), np.log(omega0)])

    def nll(th):
        mu, alpha, omega = _unpack(th, K)
        val, _ = _cont_loglik(times, types, T, mu, alpha, omega, False)
        return -val if np.isfinite(val) else 1e300

    res = optimize.minimize(nll, theta0, method="L-BFGS-B", options={"maxiter": 2000, "maxfun": 200000})
    mu, alpha, omega = _unpack(res.x, K)
    ll, resid = _cont_loglik(times, types, T, mu, alpha, omega, True)
    branch = alpha.sum(axis=0)
    first = np.zeros(2, dtype=bool)
    keep = np.ones(len(types), dtype=bool)
    for n, i in enumerate(types):  # the first event of each type has no preceding inter-event interval
        if not first[i]:
            keep[n] = False
            first[i] = True
    out = {"K": K, "loglik": float(ll), "n_free": 2 + 5 * K, "success": bool(res.success),
           "message": str(res.message), "mu_per_s": mu.tolist(), "omega_per_s": omega.tolist(),
           "half_life_s": (np.log(2) / omega).tolist(), "alpha": alpha.tolist(),
           "branching_matrix": branch.tolist(),
           "spectral_radius_branching": float(max(abs(np.linalg.eigvals(branch))))}
    gof = {}
    for i, name in ((0, "buy"), (1, "sell")):
        r = resid[(types == i) & keep]
        ks = stats.kstest(r, "expon")
        gof[name] = {"n": int(len(r)), "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                     "mean": float(r.mean()), "var": float(r.var())}
    out["gof"] = gof
    return out, resid, keep


def event_stream(trades: pd.DataFrame, start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, int]:
    """Collapse trades with identical timestamp and sign into one event (one aggressive order
    sweeping several resting orders prints several trades at the same nanosecond)."""
    ev = trades[["ts", "trade_sign"]].drop_duplicates()
    times = (ev["ts"] - start).dt.total_seconds().to_numpy()
    order = np.argsort(times, kind="stable")
    times = times[order]
    types = np.where(ev["trade_sign"].to_numpy()[order] == 1, 0, 1).astype(np.int64)
    # distinct-sign events at the same instant: nudge by 1 ns to keep strict ordering
    dup = np.concatenate([[False], np.diff(times) <= 0])
    times = times + np.cumsum(dup) * 1e-9
    return times, types, len(trades) - len(ev)


# ---------------------------------------------------------------------- driver

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    trades = pd.read_parquet(STEP2 / "trades_labeled.parquet")
    counts = pd.read_parquet(STEP2 / "event_counts_1s.parquet")
    feats = pd.read_parquet(STEP2 / "feature_table_step2_1s.parquet")
    committed = json.loads((STEP2 / "step2_summary.json").read_text())["hawkes_fit"]
    yb = counts["buy_events"].to_numpy(float)
    ys = counts["sell_events"].to_numpy(float)
    session_start = counts["ts"].iloc[0]
    session_end = counts["ts"].iloc[-1] + pd.Timedelta("1s")
    out: dict = {"data": {"symbol": "AAPL", "date": "2024-06-03", "n_trades": int(len(trades)),
                          "n_buckets_1s": int(len(yb)), "n_buy": int(yb.sum()), "n_sell": int(ys.sum())}}

    # 1. reference implementation check
    p_c = np.array([committed["params"][k] for k in PARAM_NAMES])
    lb_loop, ls_loop = s2.simulate_discrete_hawkes(p_c, yb, ys)
    lb_fast, ls_fast = discrete_intensities(p_c, yb, ys)
    max_dev = float(max(np.abs(lb_loop - lb_fast).max(), np.abs(ls_loop - ls_fast).max()))
    assert max_dev < 1e-9, max_dev
    full = fit_discrete(yb, ys)
    out["reference"] = {"max_abs_intensity_dev_vs_step2_loop": max_dev,
                        "refit_params": full["params"], "committed_params": committed["params"],
                        "refit_negloglik": -full["loglik"], "committed_fun": committed["fun"]}

    # 2. occupancy
    out["occupancy_1s"] = {"buy": occupancy(yb), "sell": occupancy(ys)}

    # 3. bucket width robustness
    width_rows = []
    for w in WIDTHS_S:
        c = counts_at_width(trades, w, session_start, session_end)
        f = fit_discrete(c["buy_events"].to_numpy(), c["sell_events"].to_numpy())
        p = f["x"]
        width_rows.append({"width_s": w, "n_buckets": len(c),
                           "mu_buy_per_s": p[0] / w, "mu_sell_per_s": p[1] / w,
                           "alpha_bb": p[2], "alpha_bs": p[3], "alpha_ss": p[4], "alpha_sb": p[5],
                           "beta": p[6], "decay_rate_per_s": -np.log(p[6]) / w if p[6] > 0 else np.nan,
                           "half_life_s": f["half_life_buckets"] * w if f["half_life_buckets"] else np.nan,
                           "spectral_radius_branching": f["spectral_radius_branching"],
                           "spectral_radius_companion": f["spectral_radius_companion"],
                           "share_buckets_ge2_buy": float((c["buy_events"] >= 2).mean()),
                           "success": f["success"]})
    width_df = pd.DataFrame(width_rows)
    width_df.to_csv(OUTDIR / "bucket_width_robustness.csv", index=False)

    # 4. null models
    n_obs = 2 * len(yb)
    ll_pois = poisson_loglik(yb, np.full_like(yb, yb.mean())) + poisson_loglik(ys, np.full_like(ys, ys.mean()))
    nocross = fit_discrete(yb, ys, cross=False)
    models = {"poisson": (ll_pois, 2), "hawkes_no_cross": (nocross["loglik"], 5), "hawkes_full": (full["loglik"], 7)}
    out["model_comparison"] = {
        name: {"loglik": ll, "n_free": k, "aic": 2 * k - 2 * ll, "bic": k * np.log(n_obs) - 2 * ll}
        for name, (ll, k) in models.items()}
    out["model_comparison"]["lr_full_vs_poisson"] = 2 * (full["loglik"] - ll_pois)
    out["model_comparison"]["lr_full_vs_no_cross"] = 2 * (full["loglik"] - nocross["loglik"])
    out["model_comparison"]["no_cross_params"] = nocross["params"]
    out["model_comparison"]["note"] = ("Quasi-likelihood on bucket counts; LR statistics are diagnostic only "
                                       "(chi-square asymptotics need the model to be correctly specified).")

    # 5. stability
    out["stability_full"] = {k: full[k] for k in ("spectral_radius_companion", "spectral_radius_branching",
                                                  "branching_matrix", "half_life_buckets")}

    # 6. sub-windows
    win_rows = []
    for w, idx in enumerate(np.array_split(np.arange(len(yb)), N_WINDOWS)):
        f = fit_discrete(yb[idx], ys[idx])
        win_rows.append({"window": w + 1, "start": str(counts["ts"].iloc[idx[0]]),
                         "end": str(counts["ts"].iloc[idx[-1]]), **f["params"],
                         "spectral_radius_branching": f["spectral_radius_branching"],
                         "spectral_radius_companion": f["spectral_radius_companion"],
                         "buy_rate_per_s": float(yb[idx].mean()), "sell_rate_per_s": float(ys[idx].mean()),
                         "success": f["success"]})
    win_df = pd.DataFrame(win_rows)
    win_df.to_csv(OUTDIR / "subwindow_params.csv", index=False)

    # 7. continuous time
    times, types, n_collapsed = event_stream(trades, session_start)
    T = float((session_end - session_start).total_seconds())
    cont1, resid1, keep1 = fit_continuous(times, types, T, K=1, omega0=[0.5])
    cont2, resid2, keep2 = fit_continuous(times, types, T, K=2, omega0=[1000.0, 0.05])
    cont3, _, _ = fit_continuous(times, types, T, K=3, omega0=[2000.0, 2.0, 0.05])
    out["continuous"] = {"n_events": int(len(times)), "n_trades_collapsed_same_ts_sign": int(n_collapsed),
                         "exp1": cont1, "exp2": cont2, "exp3": cont3,
                         "discrete_implied_decay_per_s": float(-np.log(full["x"][6])),
                         "discrete_branching_spectral_radius": full["spectral_radius_branching"],
                         "aic_exp1": 2 * cont1["n_free"] - 2 * cont1["loglik"],
                         "aic_exp2": 2 * cont2["n_free"] - 2 * cont2["loglik"],
                         "aic_exp3": 2 * cont3["n_free"] - 2 * cont3["loglik"]}
    out["runtime_s"] = time.time() - t_start
    (OUTDIR / "diagnostics.json").write_text(json.dumps(out, indent=2, default=float) + "\n")

    make_plots(counts, feats, full, out, width_df, win_df, (resid1, keep1, types), (resid2, keep2, types))
    write_report(out, width_df, win_df)
    print(f"Wrote {OUTDIR} and {REPORT} in {out['runtime_s']:.0f}s")


def make_plots(counts, feats, full, out, width_df, win_df, r1, r2) -> None:
    lam_b, lam_s = discrete_intensities(full["x"], counts["buy_events"].to_numpy(float),
                                        counts["sell_events"].to_numpy(float))
    # (a) zoomed bursts vs fitted intensity
    sl = slice(3600, 4200)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(np.arange(sl.start, sl.stop), counts["buy_events"].iloc[sl], width=1.0, color="#9ecae1",
           label="buy trades per second")
    ax.plot(np.arange(sl.start, sl.stop), lam_b[sl], color="#08519c", lw=1.5, label="fitted Hawkes λ_buy")
    ax.plot(np.arange(sl.start, sl.stop), feats["lambda_buy_ewm"].iloc[sl], color="#e6550d", lw=1.2,
            label="EWMA (span 60)")
    ax.set_ylim(0, np.quantile(counts["buy_events"].iloc[sl], 0.99) * 1.2)
    ax.set_xlabel("seconds since open (10:30–10:40 ET)")
    ax.set_ylabel("events / s")
    ax.set_title("Buy trade bursts vs fitted intensity (zoom)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "a_bursts_vs_intensity.png", dpi=160)
    plt.close(fig)

    # (b) Hawkes OFI vs EWMA OFI
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    axes[0].plot(feats["order_flow_imbalance"].to_numpy(), lw=0.4, color="#e6550d", label="EWMA OFI")
    axes[0].plot(feats["hawkes_ofi"].to_numpy(), lw=0.6, color="#08519c", label="Hawkes OFI")
    for y in (0.2, -0.2):
        axes[0].axhline(y, color="grey", ls="--", lw=0.8)
    axes[0].set_xlabel("second of session")
    axes[0].set_title("OFI signals (dashed: engine toxicity threshold ±0.2)")
    axes[0].legend()
    axes[1].scatter(feats["order_flow_imbalance"], feats["hawkes_ofi"], s=1, alpha=0.2)
    axes[1].set_xlabel("EWMA OFI")
    axes[1].set_ylabel("Hawkes OFI")
    rho = feats[["order_flow_imbalance", "hawkes_ofi"]].corr().iloc[0, 1]
    axes[1].set_title(f"corr = {rho:.2f}")
    fig.tight_layout()
    fig.savefig(OUTDIR / "b_hawkes_ofi_vs_ewma_ofi.png", dpi=160)
    plt.close(fig)

    # (c) Q-Q vs Exp(1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (resid, keep, types), label in ((axes[0], r1, "1 exponential"), (axes[1], r2, "2 exponentials")):
        for i, name, col in ((0, "buy", "#08519c"), (1, "sell", "#e6550d")):
            r = np.sort(resid[(types == i) & keep])
            q = stats.expon.ppf((np.arange(1, len(r) + 1) - 0.5) / len(r))
            ax.plot(q, r, ".", ms=1.5, color=col, label=name)
        m = np.quantile(q, 0.9999)
        ax.plot([0, m], [0, m], "k--", lw=1)
        ax.set_xlim(0, m)
        ax.set_ylim(0, m * 1.5)
        ax.set_xlabel("Exp(1) quantile")
        ax.set_ylabel("rescaled inter-event time")
        ax.set_title(f"Time-rescaling Q–Q, kernel: {label}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "c_qq_time_rescaling.png", dpi=160)
    plt.close(fig)

    # (d) occupancy
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cats = ["0", "1", "2", "3+"]
    x = np.arange(len(cats))
    for off, side, col in ((-0.2, "buy", "#08519c"), (0.2, "sell", "#e6550d")):
        vals = [out["occupancy_1s"][side][c] for c in cats]
        bars = ax.bar(x + off, vals, width=0.4, color=col, label=side)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, cats)
    ax.set_xlabel("events in a 1-second bucket")
    ax.set_ylabel("share of buckets")
    ax.set_title("Bucket occupancy (1 s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "d_occupancy.png", dpi=160)
    plt.close(fig)

    # (e) parameter stability
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    w = win_df["window"]
    for col in ("alpha_bb", "alpha_bs", "alpha_ss", "alpha_sb"):
        axes[0].plot(w, win_df[col], "o-", label=col)
    axes[0].set_title("excitation by window")
    axes[0].legend(fontsize=8)
    axes[1].plot(w, win_df["beta"], "o-", label="beta")
    axes[1].plot(w, win_df["spectral_radius_branching"], "s-", label="ρ(branching)")
    axes[1].axhline(1, color="grey", ls="--", lw=0.8)
    axes[1].set_title("decay and branching by window")
    axes[1].legend(fontsize=8)
    axes[2].plot(width_df["width_s"], width_df["spectral_radius_branching"], "o-", label="ρ(branching)")
    axes[2].plot(width_df["width_s"], width_df["half_life_s"] / width_df["half_life_s"].max(), "s-",
                 label="half-life (scaled)")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("bucket width (s)")
    axes[2].set_title("bucket-width robustness")
    axes[2].legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_xlabel("window (quarter of session)")
        ax.set_xticks(w)
    fig.tight_layout()
    fig.savefig(OUTDIR / "e_parameter_stability.png", dpi=160)
    plt.close(fig)


def _fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else str(x)


def write_report(out: dict, width_df: pd.DataFrame, win_df: pd.DataFrame) -> None:
    ref, occ, mc, st, cont = out["reference"], out["occupancy_1s"], out["model_comparison"], out["stability_full"], \
        out["continuous"]
    c1, c2 = cont["exp1"], cont["exp2"]
    p = ref["refit_params"]
    L = []
    L.append("# Hawkes calibration report\n")
    L.append("_Generated by `scripts/hawkes_diagnostics.py` (`make diagnostics`). Do not edit by hand._\n")
    L.append(f"Data: AAPL, 2024-06-03, regular session, {out['data']['n_trades']:,} Lee–Ready-labelled trades "
             f"({out['data']['n_buy']:,} buy, {out['data']['n_sell']:,} sell), {out['data']['n_buckets_1s']:,} "
             "one-second buckets. **One day, one symbol, in-sample: everything below is descriptive.**\n")
    L.append("## Model and estimator\n")
    L.append("Step 2 fits a discrete bivariate Hawkes recursion with a **shared decay β**:\n")
    L.append("```\nλ_b[t] = μ_b + β λ_b[t−1] + α_bb N_b[t−1] + α_bs N_s[t−1]\n"
             "λ_s[t] = μ_s + β λ_s[t−1] + α_ss N_s[t−1] + α_sb N_b[t−1]      λ[0] = μ\n```\n")
    L.append("The estimator is a **Poisson quasi-MLE** on bucket counts: it maximises Σ N log λ − λ, treating each "
             "bucket count as Poisson given the past. This is an approximation of the exact point-process "
             "likelihood, not the exact MLE. It is minimised with L-BFGS-B (numerical gradients, maxiter 500) "
             "from x0 = [½·mean N_b, ½·mean N_s, 0.10, 0.03, 0.10, 0.03, 0.60], with bounds μ ∈ [1e-6, 50], "
             "α ∈ [0, 5], β ∈ [0, 0.999]. Penalties are added for β ≥ 0.999, β < 0 and Σα > 5. The bounds already "
             "keep β < 0.999, so the β penalty never binds, and **no constraint on the spectral radius is "
             "enforced**. Stationarity is checked after the fit instead (below).\n")
    L.append(f"The vectorised recursion in this script matches step 2's loop to {ref['max_abs_intensity_dev_vs_step2_loop']:.1e}. "
             f"Refit: −loglik {ref['refit_negloglik']:.2f} vs committed {ref['committed_fun']:.2f}.\n")
    L.append("| parameter | refit | committed |\n|---|---:|---:|")
    for k in PARAM_NAMES:
        L.append(f"| {k} | {p[k]:.5f} | {ref['committed_params'][k]:.5f} |")
    L.append("")
    L.append("## 1. Bucket occupancy\n")
    L.append("| side | 0 | 1 | 2 | 3+ | mean | var/mean |\n|---|---:|---:|---:|---:|---:|---:|")
    for side in ("buy", "sell"):
        o = occ[side]
        L.append(f"| {side} | {o['0']:.3f} | {o['1']:.3f} | {o['2']:.3f} | {o['3+']:.3f} | {o['mean']:.2f} | "
                 f"{o['var_over_mean']:.1f} |")
    L.append("")
    ge3 = max(occ['buy']['3+'], occ['sell']['3+'])
    L.append(f"**Finding:** multi-event buckets are common, not rare: up to {ge3:.0%} of buckets hold 3 or more "
             "trades. Counts are also heavily over-dispersed (var/mean ≫ 1). So the 1-second quasi-likelihood "
             "throws away the within-bucket timing that drives self-excitation. That is why the exact "
             "continuous-time fit below disagrees with it so much.\n")
    L.append("## 2. Null models\n")
    L.append("| model | free params | loglik | AIC | BIC |\n|---|---:|---:|---:|---:|")
    for name in ("poisson", "hawkes_no_cross", "hawkes_full"):
        m = mc[name]
        L.append(f"| {name} | {m['n_free']} | {m['loglik']:.1f} | {m['aic']:.1f} | {m['bic']:.1f} |")
    L.append("")
    L.append(f"LR(full vs Poisson) = {mc['lr_full_vs_poisson']:.1f}; LR(full vs no-cross) = "
             f"{mc['lr_full_vs_no_cross']:.1f} (2 restrictions). {mc['note']} Clustering is overwhelming. "
             "Cross-excitation improves AIC and BIC "
             + ("as well." if mc['hawkes_full']['bic'] < mc['hawkes_no_cross']['bic'] else
                "only marginally or not at all.") + "\n")
    L.append("## 3. Stability\n")
    L.append(f"- Spectral radius of the one-step companion matrix βI + α: **{st['spectral_radius_companion']:.4f}** "
             "(must be < 1 for a stationary mean).")
    L.append(f"- Branching matrix α/(1−β) = {np.round(st['branching_matrix'], 3).tolist()}. Its spectral radius is "
             f"**{st['spectral_radius_branching']:.3f}** (expected events triggered per event, all generations "
             "counted through the dominant mode).")
    L.append(f"- Decay half-life: {st['half_life_buckets']:.1f} buckets (seconds).\n")
    L.append("Note: in this discrete parameterisation the branching ratio is α/(1−β), not α/β.\n")
    L.append("## 4. Bucket-width robustness\n")
    L.append("μ is rescaled to events per second, and β to a per-second decay rate (−ln β / Δ) and half-life. "
             "α and the branching ratio are unitless, so they are directly comparable across widths.\n")
    cols = ["width_s", "mu_buy_per_s", "mu_sell_per_s", "alpha_bb", "alpha_sb", "beta", "half_life_s",
            "spectral_radius_branching", "share_buckets_ge2_buy"]
    L.append("| " + " | ".join(cols) + " |\n|" + "---:|" * len(cols))
    for _, r in width_df.iterrows():
        L.append("| " + " | ".join(_fmt(r[c], 3) for c in cols) + " |")
    L.append("")
    L.append(f"**Finding:** the branching ratio is fairly robust to the bucket width, ranging from "
             f"{width_df['spectral_radius_branching'].min():.2f} to "
             f"{width_df['spectral_radius_branching'].max():.2f}. The half-life is less robust: it grows from "
             f"{width_df['half_life_s'].min():.1f} s to {width_df['half_life_s'].max():.1f} s as the buckets "
             "widen. That drift is mild evidence that one exponential time scale is a compromise across several "
             "scales, which the exact fit below confirms.\n")
    L.append("## 5. Sub-window stability (4 equal blocks)\n")
    cols = ["window", "start", "mu_buy", "mu_sell", "alpha_bb", "alpha_bs", "alpha_ss", "alpha_sb", "beta",
            "spectral_radius_branching", "buy_rate_per_s"]
    L.append("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for _, r in win_df.iterrows():
        L.append("| " + " | ".join(_fmt(r[c], 4) if c != "start" else str(r[c])[11:19] for c in cols) + " |")
    L.append("")
    rho_full = out["stability_full"]["spectral_radius_branching"]
    L.append(f"**Finding:** the parameters move a lot across the day. The per-window branching ratio ranges from "
             f"{win_df['spectral_radius_branching'].min():.2f} to {win_df['spectral_radius_branching'].max():.2f}, "
             f"all below the full-day value of {rho_full:.2f}. The trade rate also changes "
             f"({win_df['buy_rate_per_s'].min():.1f}–{win_df['buy_rate_per_s'].max():.1f} buys/s, highest in the "
             "block containing the open). A constant baseline μ cannot follow that intraday seasonality, so the "
             "full-day fit attributes part of it to self-excitation and is **biased toward criticality**. A "
             "time-varying baseline would be the fix.\n")
    L.append("Step 2 also fits the parameters **on the whole day and then uses the intensities in-sample**. That is "
             "look-ahead in the parameters, though not in the recursion itself. A walk-forward fit (calibrate on "
             "a past window, apply to the next) is needed before any out-of-sample claim.\n")
    L.append("## 6. Exact continuous-time fit and time-rescaling goodness of fit\n")
    L.append(f"Event times come from `trades_labeled.parquet`. {cont['n_trades_collapsed_same_ts_sign']:,} trades that share "
             "a nanosecond timestamp and side with another trade were collapsed into one event (one aggressive "
             f"order sweeping several resting orders), leaving {cont['n_events']:,} events. The kernel is "
             "φ_ij(t) = Σ_k α_ij^k ω_k e^{−ω_k t} with decays shared across pairs. Fitted by exact MLE "
             "(L-BFGS-B on log-parameters). Ogata's time rescaling should turn inter-event compensator "
             "increments into i.i.d. Exp(1) if the model is right.\n")
    L.append("| kernel | loglik | AIC | decay half-lives (s) | ρ(branching) | KS buy (stat, p) | KS sell (stat, p) |")
    L.append("|---|---:|---:|---|---:|---|---|")
    c3 = cont["exp3"]
    for c, aic in ((c1, cont["aic_exp1"]), (c2, cont["aic_exp2"]), (c3, cont["aic_exp3"])):
        hl = ", ".join(f"{h:.3g}" for h in c["half_life_s"])
        L.append(f"| {c['K']} exponential(s) | {c['loglik']:.0f} | {aic:.0f} | {hl} | "
                 f"{c['spectral_radius_branching']:.3f} | {c['gof']['buy']['ks_stat']:.3f}, "
                 f"{c['gof']['buy']['ks_pvalue']:.1e} | {c['gof']['sell']['ks_stat']:.3f}, "
                 f"{c['gof']['sell']['ks_pvalue']:.1e} |")
    L.append("")
    best = min((c1, c2, c3), key=lambda c: 2 * c["n_free"] - 2 * c["loglik"])
    L.append(f"The discrete fit implies a decay of {cont['discrete_implied_decay_per_s']:.4f}/s "
             f"(half-life {np.log(2) / cont['discrete_implied_decay_per_s']:.1f} s) and a branching radius of "
             f"{cont['discrete_branching_spectral_radius']:.3f}. The AIC-best exact fit ({best['K']} exponentials) "
             f"has half-lives of {', '.join(f'{h:.3g}' for h in best['half_life_s'])} s and a radius of "
             f"{best['spectral_radius_branching']:.3f}.\n")
    L.append("**Finding:** the two estimators describe different phenomena. On event time, the excitation that "
             "dominates is at **millisecond** scale or below, invisible to 1-second buckets. The 12-second decay of "
             "the discrete fit is a separate, slower clustering, and the discrete model forces both into one β. "
             "Each extra exponential raises the likelihood a lot, which fits the multi-scale, slowly decaying "
             "kernels reported in the literature (Bacry, Jaisson & Muzy, 2016). Even so, with ~10⁵ events the KS "
             "test rejects every kernel tried. The Q–Q plot shows the misfit is in the tails (rescaled variance "
             "> 1: residual clustering is left). Treat the Hawkes intensities as a **descriptive feature "
             "generator**, not a correctly specified model. The 1-ns nudge used to break exact ties between buy "
             "and sell events, and the collapsing of same-timestamp trades, are modelling choices that affect "
             "the shortest time scale.\n")
    L.append("## Figures\n")
    for f, cap in (("a_bursts_vs_intensity.png", "(a) raw trade bursts vs fitted intensity"),
                   ("b_hawkes_ofi_vs_ewma_ofi.png", "(b) Hawkes OFI vs EWMA OFI"),
                   ("c_qq_time_rescaling.png", "(c) time-rescaling Q–Q"),
                   ("d_occupancy.png", "(d) occupancy"),
                   ("e_parameter_stability.png", "(e) parameter stability")):
        L.append(f"![{cap}](../output/hawkes_diagnostics/{f})\n")
    L.append("## Limitations\n")
    L.append("- One day, one symbol. Nothing here is evidence about other days.")
    L.append("- Trade signs come from Lee–Ready against 1-second quotes (see README, *Changes*, on the look-ahead "
             "fix to the labelling). Sign errors bias cross-excitation upward.")
    L.append("- The quasi-MLE ignores within-bucket timing, and the shared β forces one time scale on all four "
             "interactions.")
    L.append("- No parameter uncertainty is reported yet (bootstrap or observed information). Power-law kernels were not "
             "fitted; sums of up to three exponentials stand in for them.")
    L.append("- The parameters are estimated on the same day the engine trades. Hawkes features used in the engine "
             "are in-sample.")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
