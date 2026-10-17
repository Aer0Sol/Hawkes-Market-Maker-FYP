"""Step 4: multi-seed parameter sweep, Hawkes vs baseline with confidence intervals.

Changes vs the original sweep (see docs/changes.md):
- every trading configuration runs over SEEDS (the fill simulation is stochastic);
  tables report seed means, and the Hawkes-vs-baseline comparison is paired by
  (A, k, seed) with a t-based 95% confidence interval;
- markout_horizon only changes measurement, not trading, so the engine runs once
  per (mode, A, k, seed) and markouts at each horizon are computed here from the
  mid path;
- toxicity regimes use one proxy (|EWMA buy - sell intensity| from the feature
  table) for both modes, so regime cut points are identical across modes;
- the fill-model parameters calibrated by scripts/calibrate_fill_model.py are
  added as an extra configuration;
- the number of trials evaluated is written to every report;
- an ablation reruns Hawkes vs baseline with the OFI-driven logic switched off
  (no toxicity widening/pulls/haircuts, no OFI skew), so the remaining gap can
  only come from the pressure term of the fill model.
"""

import json
import subprocess
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_DIR = Path(".")
STEP3_BIN = BASE_DIR / "bin" / "step3_mm"
FEATURES_CSV = BASE_DIR / "output" / "step2_hawkes" / "feature_table_step2_1s.csv"
FILL_CALIB = BASE_DIR / "output" / "fill_calibration" / "fill_model_calibration.json"
OUT_ROOT = BASE_DIR / "output" / "step4_analysis"

BASE_ARGS = {
    "gamma": 0.10,
    "inventory_limit": 500,
    "max_order_size": 25,
    "tick_size": 0.01,
    "min_spread_ticks": 2,
    "ofi_skew_coef": 0.03,
    "microprice_skew_coef": 0.10,
    "latency_budget_us": 500,
    "maker_fee_bps": 0.20,
    "queue_haircut_base": 0.25,
    "toxicity_haircut_strength": 0.60,
    "toxic_ofi_threshold": 0.20,
    "toxic_side_widen_ticks": 3,
}

A_GRID = [0.05, 0.08, 0.12]
K_GRID = [4.0, 6.0, 8.0]
MARKOUT_GRID = [1, 3, 5]
USE_HAWKES_GRID = [True, False]
SEEDS = list(range(1, 11))
PLOT_SEED = 1  # per-run engine_results.csv is kept only for this seed (used by plot_graphs.py)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ak_grid() -> list[tuple[float, float, str]]:
    """(A, k, label) trading configurations: the original grid plus the calibrated point."""
    grid = [(A, k, "grid") for A, k in product(A_GRID, K_GRID)]
    if FILL_CALIB.exists():
        c = json.loads(FILL_CALIB.read_text())
        grid.append((round(c["A"], 4), round(c["k_per_dollar"], 2), "calibrated"))
    else:
        print(f"NOTE: {FILL_CALIB} missing; run scripts/calibrate_fill_model.py to add the calibrated config")
    return grid


# Switches off every use of the OFI signal in quoting (the pressure term stays).
SIGNAL_OFF_ARGS = ["--toxic-ofi-threshold", "10", "--toxicity-haircut-strength", "0", "--ofi-skew-coef", "0"]


def run_step3(outdir: Path, A: float, k: float, use_hawkes: bool, seed: int, extra: list | None = None):
    cmd = [
        str(STEP3_BIN),
        "--features", str(FEATURES_CSV),
        "--outdir", str(outdir),
        "--k", str(k),
        "--A", str(A),
        "--seed", str(seed),
        "--markout-horizon-steps", "1",
    ]
    for name, val in BASE_ARGS.items():
        cmd += ["--" + name.replace("_", "-"), str(val)]
    cmd += extra or []
    if use_hawkes:
        cmd.append("--use-hawkes")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def load_summary_and_results(run_dir: Path):
    with open(run_dir / "step3_summary.json", "r") as f:
        summary = json.load(f)
    df = pd.read_csv(run_dir / "engine_results.csv")
    return summary, df


def compute_drawdown(series: pd.Series):
    running_max = series.cummax()
    dd = running_max - series
    return float(dd.max())


def with_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Recompute markout columns of an engine run at another horizon (same sign convention as the engine)."""
    df = df.copy()
    mid = df["midprice"].to_numpy()
    idx = np.minimum(np.arange(len(df)) + max(1, horizon), len(df) - 1)
    fut = mid[idx]
    df["next_midprice"] = fut
    df["bid_markout"] = np.where(df["bid_fill"] == 1, fut - df["quoted_bid"], 0.0)
    df["ask_markout"] = np.where(df["ask_fill"] == 1, df["quoted_ask"] - fut, 0.0)
    df["adverse_selection_cost"] = (
        np.maximum(0.0, mid - fut) * df["bid_fill_qty"] + np.maximum(0.0, fut - mid) * df["ask_fill_qty"]
    )
    return df


def compute_extra_metrics(df: pd.DataFrame):
    out = {}

    out["final_pnl"] = float(df["mtm_pnl"].iloc[-1])
    out["max_drawdown"] = compute_drawdown(df["mtm_pnl"])
    out["inventory_mean_abs"] = float(df["inventory"].abs().mean())
    out["inventory_std"] = float(df["inventory"].std())
    out["latency_p99_empirical"] = float(df["latency_us"].quantile(0.99))
    out["fill_rate_bid_empirical"] = float(df["bid_fill"].mean())
    out["fill_rate_ask_empirical"] = float(df["ask_fill"].mean())
    out["avg_fee_per_step"] = float(df["fee_paid"].mean())
    out["avg_adverse_cost_per_step"] = float(df["adverse_selection_cost"].mean())
    out["avg_realized_spread_per_step"] = float(df["realized_spread_component"].mean())
    out["total_adverse_selection_cost_h"] = float(df["adverse_selection_cost"].sum())
    out["quote_pull_rate_bid"] = float(1.0 - df["quote_bid_active"].mean())
    out["quote_pull_rate_ask"] = float(1.0 - df["quote_ask_active"].mean())

    bid_markouts = df.loc[df["bid_fill"] == 1, "bid_markout"]
    ask_markouts = df.loc[df["ask_fill"] == 1, "ask_markout"]
    all_markouts = pd.concat([bid_markouts, ask_markouts], ignore_index=True)

    out["avg_bid_markout_per_fill_h"] = float(bid_markouts.mean()) if len(bid_markouts) else np.nan
    out["avg_ask_markout_per_fill_h"] = float(ask_markouts.mean()) if len(ask_markouts) else np.nan
    if len(all_markouts):
        out["avg_markout_all_fills"] = float(all_markouts.mean())
        out["median_markout_all_fills"] = float(all_markouts.median())
        out["p10_markout_all_fills"] = float(all_markouts.quantile(0.10))
        out["p90_markout_all_fills"] = float(all_markouts.quantile(0.90))
    else:
        out["avg_markout_all_fills"] = np.nan
        out["median_markout_all_fills"] = np.nan
        out["p10_markout_all_fills"] = np.nan
        out["p90_markout_all_fills"] = np.nan

    out["avg_abs_pressure_diff"] = float((df["buy_pressure"] - df["sell_pressure"]).abs().mean())
    return out


def toxicity_regimes() -> pd.Series:
    """Common regime labels for every run: tertiles of |EWMA buy - sell intensity|."""
    f = pd.read_csv(FEATURES_CSV, usecols=["lambda_buy_ewm", "lambda_sell_ewm"])
    proxy = (f["lambda_buy_ewm"] - f["lambda_sell_ewm"]).abs()
    q1, q2 = proxy.quantile(0.33), proxy.quantile(0.66)
    return pd.Series(np.where(proxy <= q1, "low_toxicity", np.where(proxy <= q2, "medium_toxicity", "high_toxicity")))


def regime_breakdown(df: pd.DataFrame, regimes: pd.Series):
    df = df.copy()
    df["toxicity_regime"] = regimes.to_numpy()

    rows = []
    for regime, g in df.groupby("toxicity_regime"):
        bid_markouts = g.loc[g["bid_fill"] == 1, "bid_markout"]
        ask_markouts = g.loc[g["ask_fill"] == 1, "ask_markout"]
        all_markouts = pd.concat([bid_markouts, ask_markouts], ignore_index=True)

        rows.append({
            "toxicity_regime": regime,
            "n_steps": int(len(g)),
            "mean_pnl": float(g["mtm_pnl"].mean()),
            "final_pnl_last_step": float(g["mtm_pnl"].iloc[-1]),
            "pnl_change_in_regime": float(df["mtm_pnl"].diff().fillna(df["mtm_pnl"]).loc[g.index].sum()),
            "fill_rate_bid": float(g["bid_fill"].mean()),
            "fill_rate_ask": float(g["ask_fill"].mean()),
            "avg_adverse_selection_cost": float(g["adverse_selection_cost"].mean()),
            "avg_realized_spread_component": float(g["realized_spread_component"].mean()),
            "avg_markout": float(all_markouts.mean()) if len(all_markouts) else np.nan,
            "median_markout": float(all_markouts.median()) if len(all_markouts) else np.nan,
            "mean_inventory_abs": float(g["inventory"].abs().mean()),
        })

    order = {"low_toxicity": 0, "medium_toxicity": 1, "high_toxicity": 2}
    regime_df = pd.DataFrame(rows)
    if len(regime_df):
        regime_df["sort_key"] = regime_df["toxicity_regime"].map(order)
        regime_df = regime_df.sort_values("sort_key").drop(columns="sort_key")
    return regime_df


def plot_top_pnl_paths(master_df: pd.DataFrame, results_map: dict, outdir: Path):
    plt.figure(figsize=(12, 7))
    top = master_df.head(6)
    for _, row in top.iterrows():
        rid = row["run_id"]
        label = f"{rid}"
        plt.plot(results_map[rid]["mtm_pnl"].values, label=label, linewidth=1.4)
    plt.title("Top 6 Runs by Risk-Adjusted Score")
    plt.xlabel("Step")
    plt.ylabel("MTM PnL")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "top6_pnl_paths.png", dpi=180)
    plt.close()

def plot_heatmap(master_df: pd.DataFrame, outdir: Path, use_hawkes: bool):
    sub = master_df[master_df["use_hawkes"] == use_hawkes].copy()
    if sub.empty:
        return
    pivot = sub.groupby(["A", "k"])["final_mtm_pnl"].mean().reset_index()
    heat = pivot.pivot(index="A", columns="k", values="final_mtm_pnl")

    plt.figure(figsize=(8, 6))
    plt.imshow(heat.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(heat.columns)), [str(c) for c in heat.columns])
    plt.yticks(range(len(heat.index)), [str(i) for i in heat.index])
    plt.xlabel("k")
    plt.ylabel("A")
    plt.title(f"Average Final MTM PnL by A and k ({'Hawkes' if use_hawkes else 'Baseline'})")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            plt.text(j, i, f"{heat.values[i, j]:.0f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(label="Final MTM PnL")
    plt.tight_layout()
    name = "pnl_heatmap_A_k_hawkes.png" if use_hawkes else "pnl_heatmap_A_k_baseline.png"
    plt.savefig(outdir / name, dpi=180)
    plt.close()

def plot_markout_by_horizon(master_df: pd.DataFrame, outdir: Path):
    agg = (
        master_df.groupby(["use_hawkes", "markout_horizon_steps"])["avg_markout_all_fills"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    for use_hawkes, g in agg.groupby("use_hawkes"):
        label = "Hawkes" if use_hawkes else "Baseline"
        plt.plot(g["markout_horizon_steps"], g["avg_markout_all_fills"], marker="o", label=label)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Markout Horizon Steps")
    plt.ylabel("Average Markout Across Fills")
    plt.title("Average Markout vs Horizon: Hawkes vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "avg_markout_by_horizon_hawkes_vs_baseline.png", dpi=180)
    plt.close()

def plot_hawkes_vs_baseline_box(master_df: pd.DataFrame, outdir: Path):
    hawkes = master_df.loc[master_df["use_hawkes"] == True, "final_mtm_pnl"].values
    baseline = master_df.loc[master_df["use_hawkes"] == False, "final_mtm_pnl"].values
    plt.figure(figsize=(7, 5))
    plt.boxplot([hawkes, baseline], tick_labels=["Hawkes", "Baseline"])
    plt.ylabel("Final MTM PnL")
    plt.title("Final MTM PnL Distribution: Hawkes vs Baseline")
    plt.tight_layout()
    plt.savefig(outdir / "hawkes_vs_baseline_pnl_boxplot.png", dpi=180)
    plt.close()

def plot_regime_comparison(regime_master: pd.DataFrame, outdir: Path):
    if regime_master.empty:
        return

    agg = (
        regime_master.groupby(["use_hawkes", "toxicity_regime"])
        .agg(
            avg_markout=("avg_markout", "mean"),
            avg_adverse_selection_cost=("avg_adverse_selection_cost", "mean"),
            mean_pnl=("mean_pnl", "mean"),
        )
        .reset_index()
    )

    regimes = ["low_toxicity", "medium_toxicity", "high_toxicity"]
    x = np.arange(len(regimes))
    width = 0.35

    for metric, filename, title in [
        ("avg_markout", "regime_avg_markout_hawkes_vs_baseline.png", "Average Markout by Toxicity Regime"),
        ("avg_adverse_selection_cost", "regime_adverse_cost_hawkes_vs_baseline.png", "Adverse Selection Cost by Toxicity Regime"),
        ("mean_pnl", "regime_mean_pnl_hawkes_vs_baseline.png", "Mean PnL by Toxicity Regime"),
    ]:
        hawkes_vals = []
        base_vals = []
        for r in regimes:
            h = agg[(agg["toxicity_regime"] == r) & (agg["use_hawkes"] == True)]
            b = agg[(agg["toxicity_regime"] == r) & (agg["use_hawkes"] == False)]
            hawkes_vals.append(h[metric].iloc[0] if len(h) else np.nan)
            base_vals.append(b[metric].iloc[0] if len(b) else np.nan)

        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, hawkes_vals, width, label="Hawkes")
        plt.bar(x + width/2, base_vals, width, label="Baseline")
        plt.xticks(x, regimes)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / filename, dpi=180)
        plt.close()

SUMMARY_KEYS = [
    "final_mtm_pnl", "mean_mtm_pnl", "std_mtm_pnl", "fill_count_total", "avg_bid_fill_prob", "avg_ask_fill_prob",
    "total_fees_paid", "total_realized_spread_component", "max_abs_inventory",
    "mean_latency_us", "p99_latency_us", "max_latency_us", "latency_budget_exceeded_count",
]


def run_id_for(use_hawkes: bool, A: float, k: float, horizon: int) -> str:
    return f"{'hawkes' if use_hawkes else 'baseline'}__A_{A:.2f}__k_{k:.1f}__h_{horizon}"


def build_seed_table():
    """Run every (mode, A, k, seed) once; expand to markout horizons. Returns per-seed rows."""
    ensure_dir(OUT_ROOT)
    runs_dir = OUT_ROOT / "runs"
    ensure_dir(runs_dir)
    regimes = toxicity_regimes()
    scratch = OUT_ROOT / "_scratch_run"

    seed_rows, regime_rows, plot_results = [], [], {}
    grid = ak_grid()
    total = len(grid) * len(USE_HAWKES_GRID) * len(SEEDS)
    done = 0
    for (A, k, cfg_kind), use_hawkes, seed in product(grid, USE_HAWKES_GRID, SEEDS):
        done += 1
        run_step3(scratch, A, k, use_hawkes, seed)
        summary, df_raw = load_summary_and_results(scratch)
        if done % 20 == 0 or done == total:
            print(f"[{done}/{total}] mode={'hawkes' if use_hawkes else 'baseline'} A={A} k={k} seed={seed}")

        for horizon in MARKOUT_GRID:
            run_id = run_id_for(use_hawkes, A, k, horizon)
            df = with_horizon(df_raw, horizon)
            extra = compute_extra_metrics(df)
            row = {
                "run_id": run_id, "use_hawkes": use_hawkes, "model_type": "hawkes" if use_hawkes else "baseline",
                "config_kind": cfg_kind, "A": A, "k": k, "markout_horizon_steps": horizon, "seed": seed,
                **{key: summary[key] for key in SUMMARY_KEYS},
                "total_adverse_selection_cost": extra.pop("total_adverse_selection_cost_h"),
                "avg_bid_markout_per_fill": extra.pop("avg_bid_markout_per_fill_h"),
                "avg_ask_markout_per_fill": extra.pop("avg_ask_markout_per_fill_h"),
                **extra,
            }
            seed_rows.append(row)
            reg = regime_breakdown(df, regimes)
            reg["run_id"], reg["use_hawkes"], reg["A"], reg["k"] = run_id, use_hawkes, A, k
            reg["markout_horizon_steps"], reg["seed"] = horizon, seed
            regime_rows.append(reg)
            if seed == PLOT_SEED:
                run_dir = runs_dir / run_id
                ensure_dir(run_dir)
                df.to_csv(run_dir / "engine_results.csv", index=False, float_format="%.8f")
                (run_dir / "step3_summary.json").write_text(json.dumps(
                    {**summary, "markout_horizon_steps_recomputed": horizon}, indent=2))
                reg.to_csv(run_dir / "regime_breakdown.csv", index=False)
                plot_results[run_id] = df

    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()
    seed_df = pd.DataFrame(seed_rows)
    seed_df["risk_adjusted_score"] = (
        seed_df["final_mtm_pnl"] - 0.5 * seed_df["max_drawdown"] - 50.0 * seed_df["avg_adverse_cost_per_step"]
    )
    seed_df.to_csv(OUT_ROOT / "step4_seed_runs.csv", index=False)
    regime_seed = pd.concat(regime_rows, ignore_index=True)
    return seed_df, regime_seed, plot_results


def mean_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan, np.nan
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    h = float(stats.t.ppf(0.5 + level / 2, len(x) - 1) * se)
    return m, se, m - h, m + h


def build_master_table(seed_df: pd.DataFrame):
    keys = ["run_id", "use_hawkes", "model_type", "config_kind", "A", "k", "markout_horizon_steps"]
    num = [c for c in seed_df.columns if c not in keys + ["seed"]]
    master_df = seed_df.groupby(keys, as_index=False)[num].mean()
    sd = seed_df.groupby(keys)["final_mtm_pnl"].std(ddof=1).rename("final_mtm_pnl_seed_std")
    master_df = master_df.merge(sd.reset_index(), on=keys)
    master_df["n_seeds"] = len(SEEDS)
    master_df = master_df.sort_values(["risk_adjusted_score", "final_mtm_pnl"], ascending=[False, False])
    master_df = master_df.reset_index(drop=True)
    master_df.to_csv(OUT_ROOT / "step4_master_summary.csv", index=False)
    master_df.head(10).to_csv(OUT_ROOT / "step4_top10_runs.csv", index=False)

    hawkes_vs_baseline = (
        master_df[master_df["config_kind"] == "grid"].groupby("model_type")
        .agg(
            mean_final_mtm_pnl=("final_mtm_pnl", "mean"),
            median_final_mtm_pnl=("final_mtm_pnl", "median"),
            mean_risk_adjusted_score=("risk_adjusted_score", "mean"),
            mean_avg_markout_all_fills=("avg_markout_all_fills", "mean"),
            mean_total_adverse_selection_cost=("total_adverse_selection_cost", "mean"),
            mean_fill_count_total=("fill_count_total", "mean"),
            mean_p99_latency_us=("p99_latency_us", "mean"),
        )
        .reset_index()
    )
    hawkes_vs_baseline.to_csv(OUT_ROOT / "step4_hawkes_vs_baseline_summary.csv", index=False)
    return master_df


def paired_comparison(seed_df: pd.DataFrame) -> pd.DataFrame:
    """Hawkes minus baseline final PnL, paired on (A, k, seed) at horizon 1 (PnL does not depend on horizon)."""
    d = seed_df[seed_df["markout_horizon_steps"] == MARKOUT_GRID[0]]
    piv = d.pivot_table(index=["config_kind", "A", "k", "seed"], columns="model_type", values="final_mtm_pnl")
    piv["diff"] = piv["hawkes"] - piv["baseline"]
    rows = []
    for (kind, A, k), g in piv.groupby(level=[0, 1, 2]):
        m, se, lo, hi = mean_ci(g["diff"].to_numpy())
        rows.append({"config_kind": kind, "A": A, "k": k, "n_seeds": len(g),
                     "mean_hawkes": g["hawkes"].mean(), "mean_baseline": g["baseline"].mean(),
                     "mean_diff": m, "se_diff": se, "ci95_lo": lo, "ci95_hi": hi,
                     "share_seeds_hawkes_better": float((g["diff"] > 0).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "step4_hawkes_vs_baseline_paired.csv", index=False)
    return out


def signal_off_ablation() -> pd.DataFrame:
    """Paired Hawkes - baseline final PnL with the OFI signal unused in quoting."""
    scratch = OUT_ROOT / "_scratch_ablation"
    configs = [(0.08, 6.0, "grid")] + [c for c in ak_grid() if c[2] == "calibrated"]
    rows = []
    for A, k, kind in configs:
        pnl = {}
        for use_hawkes, seed in product(USE_HAWKES_GRID, SEEDS):
            run_step3(scratch, A, k, use_hawkes, seed, SIGNAL_OFF_ARGS)
            pnl[(use_hawkes, seed)] = json.loads((scratch / "step3_summary.json").read_text())["final_mtm_pnl"]
        diff = np.array([pnl[(True, s)] - pnl[(False, s)] for s in SEEDS])
        m, se, lo, hi = mean_ci(diff)
        rows.append({"config_kind": kind, "A": A, "k": k, "variant": "ofi_signal_off", "n_seeds": len(SEEDS),
                     "mean_diff": m, "se_diff": se, "ci95_lo": lo, "ci95_hi": hi,
                     "share_seeds_hawkes_better": float((diff > 0).mean())})
    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ROOT / "step4_ablation_signal_off.csv", index=False)
    return out


def build_regime_master(regime_seed: pd.DataFrame):
    keys = ["run_id", "use_hawkes", "A", "k", "markout_horizon_steps", "toxicity_regime"]
    num = [c for c in regime_seed.columns if c not in keys + ["seed"]]
    regime_master = regime_seed.groupby(keys, as_index=False)[num].mean()
    regime_master.to_csv(OUT_ROOT / "step4_regime_master.csv", index=False)

    regime_agg = (
        regime_master.groupby(["use_hawkes", "toxicity_regime"])
        .agg(
            avg_markout=("avg_markout", "mean"),
            mean_pnl=("mean_pnl", "mean"),
            avg_adverse_selection_cost=("avg_adverse_selection_cost", "mean"),
            fill_rate_bid=("fill_rate_bid", "mean"),
            fill_rate_ask=("fill_rate_ask", "mean"),
            mean_inventory_abs=("mean_inventory_abs", "mean"),
        )
        .reset_index()
    )
    regime_agg["model_type"] = regime_agg["use_hawkes"].map({True: "hawkes", False: "baseline"})
    regime_agg.to_csv(OUT_ROOT / "step4_regime_aggregate.csv", index=False)

    plot_regime_comparison(regime_master, OUT_ROOT)
    return regime_master


def write_report(master_df: pd.DataFrame, seed_df: pd.DataFrame, paired: pd.DataFrame, ablation: pd.DataFrame):
    n_configs = seed_df[["model_type", "A", "k"]].drop_duplicates().shape[0] + 2 * len(ablation)
    n_trials = seed_df[["model_type", "A", "k", "seed"]].drop_duplicates().shape[0] + 2 * len(SEEDS) * len(ablation)
    best = master_df.iloc[0]
    best_hawkes = master_df[master_df["use_hawkes"] == True].iloc[0]
    best_baseline = master_df[master_df["use_hawkes"] == False].iloc[0]
    grid_master = master_df[master_df["config_kind"] == "grid"]

    lines = ["STEP 4 COMPLETE", ""]
    lines.append("Data: AAPL 2024-06-03, one session (23,400 one-second rows). In-sample; no holdout.")
    lines.append(f"Trials: {n_configs} trading configurations x {len(SEEDS)} seeds = {n_trials} engine runs, "
                 f"including the ablation (markout horizons {MARKOUT_GRID} re-measure the same runs).")
    lines.append(f"Seeds: {SEEDS}. All numbers below are seed means unless stated.")
    lines.append("")
    lines.append(f"Overall best config (by mean risk-adjusted score): {best['run_id']}")
    lines.append(f"  mean final MTM PnL {best['final_mtm_pnl']:.2f} (seed std {best['final_mtm_pnl_seed_std']:.2f})")
    lines.append(f"  median config final PnL across all configs: {master_df['final_mtm_pnl'].median():.2f}")
    lines.append("")
    for label, b in (("Best Hawkes config", best_hawkes), ("Best Baseline config", best_baseline)):
        lines.append(label)
        lines.append(f"  run_id: {b['run_id']}")
        lines.append(f"  final_mtm_pnl: {b['final_mtm_pnl']:.2f} (seed std {b['final_mtm_pnl_seed_std']:.2f})")
        lines.append(f"  risk_adjusted_score: {b['risk_adjusted_score']:.2f}")
        lines.append(f"  avg_markout_all_fills: {b['avg_markout_all_fills']:.6f} (positive = profitable)")
        lines.append("")

    lines.append("Hawkes minus baseline final PnL, paired on (A, k, seed), 95% t-interval over seeds:")
    for _, r in paired.iterrows():
        sig = "excludes 0" if (r["ci95_lo"] > 0 or r["ci95_hi"] < 0) else "includes 0"
        lines.append(f"  [{r['config_kind']:10s}] A={r['A']:<6} k={r['k']:<6} diff={r['mean_diff']:9.2f} "
                     f"CI=[{r['ci95_lo']:9.2f}, {r['ci95_hi']:9.2f}] ({sig}); "
                     f"hawkes better in {r['share_seeds_hawkes_better']:.0%} of seeds")
    lines.append("")
    lines.append("Ablation: same comparison with the OFI signal unused in quoting (no toxicity logic, no OFI skew):")
    for _, r in ablation.iterrows():
        sig = "excludes 0" if (r["ci95_lo"] > 0 or r["ci95_hi"] < 0) else "includes 0"
        lines.append(f"  [{r['config_kind']:10s}] A={r['A']:<6} k={r['k']:<6} diff={r['mean_diff']:9.2f} "
                     f"CI=[{r['ci95_lo']:9.2f}, {r['ci95_hi']:9.2f}] ({sig})")
    lines.append("  Whatever gap remains here comes only from the Hawkes intensity entering the simulated fill rate.")
    lines.append("")
    lines.append("Grid aggregate (original A/k grid only):")
    for mt, g in grid_master.groupby("model_type"):
        lines.append(f"  {mt}: mean_final_mtm_pnl={g['final_mtm_pnl'].mean():.2f}, "
                     f"median={g['final_mtm_pnl'].median():.2f}, "
                     f"mean_avg_markout_all_fills={g['avg_markout_all_fills'].mean():.6f}")
    lines.append("")
    lines.append("Caveats: fills are simulated by a parametric model that also consumes the Hawkes intensity, so")
    lines.append("the comparison is partly circular; one day of data; the CI covers fill-simulation noise only,")
    lines.append("not day-to-day market variation. See README 'Limitations'.")

    with open(OUT_ROOT / "step4_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if not STEP3_BIN.exists():
        raise FileNotFoundError(f"Missing Step 3 binary: {STEP3_BIN} (run `make build`)")
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing Step 2 features CSV: {FEATURES_CSV}")

    ensure_dir(OUT_ROOT)
    seed_df, regime_seed, plot_results = build_seed_table()
    master_df = build_master_table(seed_df)
    paired = paired_comparison(seed_df)
    build_regime_master(regime_seed)

    plot_top_pnl_paths(master_df, plot_results, OUT_ROOT)
    plot_heatmap(master_df[master_df["config_kind"] == "grid"], OUT_ROOT, use_hawkes=True)
    plot_heatmap(master_df[master_df["config_kind"] == "grid"], OUT_ROOT, use_hawkes=False)
    plot_markout_by_horizon(master_df, OUT_ROOT)
    plot_hawkes_vs_baseline_box(master_df[master_df["config_kind"] == "grid"], OUT_ROOT)
    ablation = signal_off_ablation()
    write_report(master_df, seed_df, paired, ablation)


if __name__ == "__main__":
    main()
