from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.style.use("seaborn-v0_8-whitegrid")

ROOT = Path(".")
STEP4_DIR = ROOT / "output" / "step4_analysis"
RUNS_DIR = STEP4_DIR / "runs"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SUMMARY = STEP4_DIR / "step4_master_summary.csv"
HAWKES_BASELINE_SUMMARY = STEP4_DIR / "step4_hawkes_vs_baseline_summary.csv"
REGIME_AGG = STEP4_DIR / "step4_regime_aggregate.csv"

master = pd.read_csv(MASTER_SUMMARY)
hb_summary = pd.read_csv(HAWKES_BASELINE_SUMMARY)
regime_agg = pd.read_csv(REGIME_AGG)

master["use_hawkes"] = master["use_hawkes"].astype(bool)

best_hawkes = master[master["use_hawkes"] == True].sort_values(
    "risk_adjusted_score", ascending=False
).iloc[0]

best_baseline = master[master["use_hawkes"] == False].sort_values(
    "risk_adjusted_score", ascending=False
).iloc[0]

best_hawkes_df = pd.read_csv(RUNS_DIR / best_hawkes["run_id"] / "engine_results.csv")
best_baseline_df = pd.read_csv(RUNS_DIR / best_baseline["run_id"] / "engine_results.csv")


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, dpi=220, bbox_inches="tight")
    plt.close()


def add_bar_labels(ax, bars, fmt="{:.3f}", fontsize=9):
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for bar in bars:
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        if h >= 0:
            y = h + 0.015 * yrange
            va = "bottom"
        else:
            y = h - 0.015 * yrange
            va = "top"
        ax.text(x, y, fmt.format(h), ha="center", va=va, fontsize=fontsize)


# 1) PnL path
plt.figure(figsize=(12, 6))
plt.plot(best_hawkes_df["mtm_pnl"], label=f"Hawkes: {best_hawkes['run_id']}", linewidth=2)
plt.plot(best_baseline_df["mtm_pnl"], label=f"Baseline: {best_baseline['run_id']}", linewidth=2)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Best-run MTM PnL: Hawkes vs Baseline")
plt.xlabel("Step")
plt.ylabel("MTM PnL")
plt.legend()
savefig("01_pnl_path_hawkes_vs_baseline.png")


# 2) Inventory path
plt.figure(figsize=(12, 6))
plt.plot(best_hawkes_df["inventory"], label="Hawkes inventory", linewidth=1.8)
plt.plot(best_baseline_df["inventory"], label="Baseline inventory", linewidth=1.8)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Inventory Path: Hawkes vs Baseline")
plt.xlabel("Step")
plt.ylabel("Inventory")
plt.legend()
savefig("02_inventory_path_hawkes_vs_baseline.png")


# 3) Markout by horizon
markout_horizon = (
    master.groupby(["model_type", "markout_horizon_steps"])["avg_markout_all_fills"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 6))
for model_type, g in markout_horizon.groupby("model_type"):
    g = g.sort_values("markout_horizon_steps")
    plt.plot(
        g["markout_horizon_steps"],
        g["avg_markout_all_fills"],
        marker="o",
        linewidth=2,
        label=model_type.capitalize(),
    )
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.title("Average Markout by Horizon")
plt.xlabel("Markout Horizon Steps")
plt.ylabel("Average Markout per Fill")
plt.legend()
savefig("03_markout_by_horizon.png")


# 4) PnL decomposition
decomp2 = (
    master.groupby("model_type")
    .agg(
        realized_spread=("total_realized_spread_component", "mean"),
        adverse_selection=("total_adverse_selection_cost", "mean"),
        fees=("total_fees_paid", "mean"),
        final_pnl=("final_mtm_pnl", "mean"),
    )
    .reset_index()
)

decomp2["adverse_selection_plot"] = -decomp2["adverse_selection"]
decomp2["fees_plot"] = -decomp2["fees"]

x = np.arange(len(decomp2))
w = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - 1.5*w, decomp2["realized_spread"], width=w, label="Realized spread")
bars2 = ax.bar(x - 0.5*w, decomp2["adverse_selection_plot"], width=w, label="Adverse selection cost")
bars3 = ax.bar(x + 0.5*w, decomp2["fees_plot"], width=w, label="Fees")
bars4 = ax.bar(x + 1.5*w, decomp2["final_pnl"], width=w, label="Final PnL")

ax.axhline(0, color="black", linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in decomp2["model_type"]])
ax.set_title("PnL Decomposition: Hawkes vs Baseline")
ax.set_ylabel("Average Value")
ax.legend()

add_bar_labels(ax, bars1, fmt="{:.0f}")
add_bar_labels(ax, bars2, fmt="{:.0f}")
add_bar_labels(ax, bars3, fmt="{:.0f}")
add_bar_labels(ax, bars4, fmt="{:.0f}")

savefig("04_pnl_decomposition.png")


# 5) Toxicity regime comparison
regime_agg["model_type"] = regime_agg["model_type"].str.capitalize()
regimes = ["low_toxicity", "medium_toxicity", "high_toxicity"]
metrics = [
    ("avg_markout", "Average Markout by Toxicity Regime", "Markout", "{:.3f}"),
    ("avg_adverse_selection_cost", "Adverse Selection Cost by Toxicity Regime", "Adverse Selection Cost", "{:.3f}"),
    ("mean_pnl", "Mean PnL by Toxicity Regime", "Mean PnL", "{:.0f}"),
]

for metric, title, ylabel, fmt in metrics:
    pivot = regime_agg.pivot(index="toxicity_regime", columns="model_type", values=metric).reindex(regimes)
    x = np.arange(len(pivot.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_h = ax.bar(x - width/2, pivot["Hawkes"], width=width, label="Hawkes")
    bars_b = ax.bar(x + width/2, pivot["Baseline"], width=width, label="Baseline")

    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_title(title)
    ax.set_xlabel("Toxicity Regime")
    ax.set_ylabel(ylabel)
    ax.legend()

    add_bar_labels(ax, bars_h, fmt=fmt)
    add_bar_labels(ax, bars_b, fmt=fmt)

    fname = f"05_{metric}_by_regime.png"
    savefig(fname)


# 6) Latency distribution (CDF) with zoomed inset
def ecdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

hx, hy = ecdf(best_hawkes_df["latency_us"].dropna())
bx, by = ecdf(best_baseline_df["latency_us"].dropna())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(hx, hy, label="Hawkes", linewidth=2)
ax.plot(bx, by, label="Baseline", linewidth=2)
ax.set_title("Latency CDF: Best Hawkes vs Best Baseline")
ax.set_xlabel("Latency (microseconds)")
ax.set_ylabel("CDF")
ax.legend()

x_zoom_max = max(
    np.quantile(best_hawkes_df["latency_us"].dropna(), 0.995),
    np.quantile(best_baseline_df["latency_us"].dropna(), 0.995),
)
x_zoom_max = min(x_zoom_max, 0.5)

axins = inset_axes(ax, width="45%", height="45%", loc="lower right", borderpad=2)
axins.plot(hx, hy, linewidth=2)
axins.plot(bx, by, linewidth=2)
axins.set_xlim(0, x_zoom_max)
axins.set_ylim(0.90, 1.001)
axins.set_title("Zoom", fontsize=10)
axins.grid(True, alpha=0.3)

savefig("06_latency_cdf.png")


# 7) Hawkes minus baseline PnL with paired 95% CIs, signal used vs unused
paired = pd.read_csv(STEP4_DIR / "step4_hawkes_vs_baseline_paired.csv")
ablation = pd.read_csv(STEP4_DIR / "step4_ablation_signal_off.csv")


def ci_row(df, kind, A=None, k=None):
    sel = df[df["config_kind"] == kind]
    if A is not None:
        sel = sel[np.isclose(sel["A"], A) & np.isclose(sel["k"], k)]
    return sel.iloc[0]


best = paired[paired["config_kind"] == "grid"].sort_values("mean_diff").iloc[-1]
panels = [
    ("Assumed fill model (k = 4–8 per USD)", [
        (f"Best grid (A={best.A:g}, k={best.k:g}), signal used", best, "used"),
        ("Default (A=0.08, k=6), signal used", ci_row(paired, "grid", 0.08, 6.0), "used"),
        ("Default (A=0.08, k=6), signal unused", ci_row(ablation, "grid", 0.08, 6.0), "unused"),
    ]),
    ("Calibrated fill model (k ≈ 58 per USD)", [
        ("Calibrated, signal used", ci_row(paired, "calibrated"), "used"),
        ("Calibrated, signal unused", ci_row(ablation, "calibrated"), "unused"),
    ]),
]
colors = {"used": "#2f6f9f", "unused": "#9a9a9a"}

fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), gridspec_kw={"height_ratios": [3, 2]})
for ax, (title, rows) in zip(axes, panels):
    for i, (label, r, variant) in enumerate(rows[::-1]):
        lo, hi = r["ci95_lo"], r["ci95_hi"]
        ax.plot([lo, hi], [i, i], color=colors[variant], linewidth=6, alpha=0.35, solid_capstyle="round")
        ax.plot(r["mean_diff"], i, "o", color=colors[variant], markersize=8)
        ax.annotate(f"{r['mean_diff']:+.0f}  [{lo:.0f}, {hi:.0f}]", (hi, i), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), [label for label, _, _ in rows[::-1]])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(title, loc="left", fontsize=11)
    lo_all = min(r["ci95_lo"] for _, r, _ in rows)
    hi_all = max(r["ci95_hi"] for _, r, _ in rows)
    pad = 0.35 * (hi_all - min(lo_all, 0))
    ax.set_xlim(min(lo_all, 0) - 0.08 * pad, hi_all + pad)
    ax.set_ylim(-0.6, len(rows) - 0.4)
axes[-1].set_xlabel("Hawkes minus baseline final PnL (USD), mean over 10 paired seeds, 95% CI")
fig.suptitle("Hawkes beats the baseline only when the order-flow signal drives quoting",
             x=0.01, ha="left", fontsize=13)
savefig("07_hawkes_vs_baseline_ci.png")


summary_table = pd.DataFrame({
    "best_hawkes_run": [best_hawkes["run_id"]],
    "best_baseline_run": [best_baseline["run_id"]],
    "best_hawkes_final_pnl": [best_hawkes["final_mtm_pnl"]],
    "best_baseline_final_pnl": [best_baseline["final_mtm_pnl"]],
    "best_hawkes_risk_adjusted_score": [best_hawkes["risk_adjusted_score"]],
    "best_baseline_risk_adjusted_score": [best_baseline["risk_adjusted_score"]],
})
summary_table.to_csv(PLOTS_DIR / "plot_summary_table.csv", index=False)

print(f"Saved plots to: {PLOTS_DIR.resolve()}")
for p in sorted(PLOTS_DIR.glob('*.png')):
    print(" -", p.name)