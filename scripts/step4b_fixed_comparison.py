"""Step 4b: the Hawkes-vs-baseline engine comparison with the step 4 confounds removed.

Step 4 (kept as-is for reference) has three problems that stop it answering "does the Hawkes
signal help the market maker?":
  O1  the engine reads step 2's stale lambda_t, fitted on the whole day (look-ahead);
  O2  one +-0.2 threshold and one 0.03 $ skew coefficient serve two signals of different scale;
  O3  in Hawkes runs the Hawkes intensity also sets the simulated fill rate (circular).

Here:
  - features come from step 2b (walk-forward, one-step-ahead Hawkes; corrected trade labels);
  - the engine standardises whichever OFI it uses causally (--ofi-standardize), so a threshold
    means the same number of standard deviations for both signals;
  - fills are replayed from the next second's real trades with FIFO queue tracking at the touch
    (--fill-model replay): the signal cannot influence its own fills, and there is no RNG.
    As a robustness check the parametric model is also run with --pressure-source ewma (the same
    pressure in both modes) over 10 seeds.

Replay runs are deterministic, so uncertainty comes from the market path instead of the RNG: the
Hawkes-minus-baseline difference in per-second PnL increments gets a Newey-West standard error.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(".")
BIN = ROOT / "bin" / "step3_mm"
CAUSAL = ROOT / "output" / "step2b_causal"
FEATURES = CAUSAL / "feature_table_causal_1s.csv"
TRADES = CAUSAL / "trades_by_bucket.csv"
FILL_CALIB = ROOT / "output" / "fill_calibration" / "fill_model_calibration.json"
OUT = ROOT / "output" / "step4b_fixed"

COMMON = ["--gamma", "0.10", "--inventory-limit", "500", "--max-order-size", "25", "--tick-size", "0.01",
          "--ofi-skew-coef", "0.03", "--microprice-skew-coef", "0.10", "--maker-fee-bps", "0.20",
          "--toxic-ofi-threshold", "0.20", "--toxic-side-widen-ticks", "3", "--toxic-pull-threshold", "0.45",
          "--ofi-standardize", "--ofi-target-sd", "0.20", "--ofi-warmup-rows", "300"]
SIGNAL_OFF = ["--toxic-ofi-threshold", "10", "--toxic-pull-threshold", "10", "--toxicity-haircut-strength", "0",
              "--ofi-skew-coef", "0"]
SEEDS = list(range(1, 11))
HAC_LAG = 300  # inventory carries across seconds, so PnL increments are autocorrelated


def calibrated_ak() -> tuple[float, float]:
    c = json.loads(FILL_CALIB.read_text())
    return round(c["A"], 4), round(c["k_per_dollar"], 2)


def replay_configs() -> list[dict]:
    A, k = calibrated_ak()
    return [
        {"config": "calibrated (k=58)", "A": A, "k": k, "min_spread_ticks": 2},
        {"config": "at the touch (k=200, 1-tick floor)", "A": A, "k": 200.0, "min_spread_ticks": 1},
        {"config": "legacy grid default (k=6)", "A": 0.08, "k": 6.0, "min_spread_ticks": 2},
    ]


def run(outdir: Path, cfg: dict, use_hawkes: bool, signal_on: bool, fill: str, seed: int = 42,
        pressure: str = "ewma") -> tuple[dict, pd.DataFrame]:
    cmd = [str(BIN), "--features", str(FEATURES), "--outdir", str(outdir), "--A", str(cfg["A"]), "--k", str(cfg["k"]),
           "--min-spread-ticks", str(cfg["min_spread_ticks"]), "--seed", str(seed), "--fill-model", fill,
           "--pressure-source", pressure, *COMMON]
    if fill == "replay":
        cmd += ["--trades", str(TRADES)]
    if use_hawkes:
        cmd.append("--use-hawkes")
    if not signal_on:
        cmd += SIGNAL_OFF
    subprocess.run(cmd, check=True, capture_output=True)
    summary = json.loads((outdir / "step3_summary.json").read_text())
    res = pd.read_csv(outdir / "engine_results.csv", usecols=["mtm_pnl", "inventory", "bid_fill_qty", "ask_fill_qty",
                                                              "bid_markout", "ask_markout", "bid_fill", "ask_fill"])
    return summary, res


def newey_west_total(d: np.ndarray, lag: int) -> tuple[float, float, float, float]:
    """Sum of d with a Newey-West standard error and 95% normal CI; returns (sum, se, lo, hi)."""
    n = len(d)
    dc = d - d.mean()
    lrv = dc @ dc / n
    for k in range(1, lag + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    se = float(np.sqrt(max(lrv, 0.0) * n))
    total = float(d.sum())
    return total, se, total - 1.96 * se, total + 1.96 * se


def replay_comparison(scratch: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, paired = [], []
    for cfg in replay_configs():
        pnl_paths = {}
        for signal_on in (True, False):
            for use_hawkes in (True, False):
                s, r = run(scratch, cfg, use_hawkes, signal_on, "replay")
                fills = r["bid_fill"].sum() + r["ask_fill"].sum()
                mk = pd.concat([r.loc[r.bid_fill == 1, "bid_markout"], r.loc[r.ask_fill == 1, "ask_markout"]])
                rows.append({"config": cfg["config"], "k": cfg["k"], "signal": "on" if signal_on else "off",
                             "model": "hawkes" if use_hawkes else "baseline", "final_pnl": s["final_mtm_pnl"],
                             "fills": int(fills), "shares_traded": int(r.bid_fill_qty.sum() + r.ask_fill_qty.sum()),
                             "avg_markout_per_share_1s": float(mk.mean()) if len(mk) else np.nan,
                             "max_abs_inventory": s["max_abs_inventory"],
                             "adverse_selection": s["total_adverse_selection_cost"]})
                pnl_paths[(signal_on, use_hawkes)] = r["mtm_pnl"].to_numpy()
        for label, a, b in [
            ("hawkes - baseline, signal on", (True, True), (True, False)),
            ("hawkes - baseline, signal off", (False, True), (False, False)),
            ("signal on - off, hawkes", (True, True), (False, True)),
            ("signal on - off, baseline", (True, False), (False, False)),
        ]:
            d = np.diff(pnl_paths[a], prepend=0.0) - np.diff(pnl_paths[b], prepend=0.0)
            total, se, lo, hi = newey_west_total(d, HAC_LAG)
            paired.append({"config": cfg["config"], "comparison": label, "diff": total, "se_nw": se,
                           "ci95_lo": lo, "ci95_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows), pd.DataFrame(paired)


def parametric_robustness(scratch: Path) -> pd.DataFrame:
    A, k = calibrated_ak()
    cfgs = [{"config": "calibrated (k=58)", "A": A, "k": k, "min_spread_ticks": 2},
            {"config": "legacy grid default (k=6)", "A": 0.08, "k": 6.0, "min_spread_ticks": 2}]
    out = []
    for cfg in cfgs:
        for signal_on in (True, False):
            diffs = []
            for seed in SEEDS:
                h, _ = run(scratch, cfg, True, signal_on, "parametric", seed, "ewma")
                b, _ = run(scratch, cfg, False, signal_on, "parametric", seed, "ewma")
                diffs.append(h["final_mtm_pnl"] - b["final_mtm_pnl"])
            d = np.array(diffs)
            se = d.std(ddof=1) / np.sqrt(len(d))
            h95 = stats.t.ppf(0.975, len(d) - 1) * se
            out.append({"config": cfg["config"], "signal": "on" if signal_on else "off", "n_seeds": len(d),
                        "mean_diff": d.mean(), "ci95_lo": d.mean() - h95, "ci95_hi": d.mean() + h95})
    return pd.DataFrame(out)


def write_report(runs: pd.DataFrame, paired: pd.DataFrame, param: pd.DataFrame):
    L = ["STEP 4b: Hawkes vs baseline with the step 4 confounds removed", "",
         "Data: AAPL 2024-06-03 (23,400 rows). Features: output/step2b_causal (walk-forward one-step-ahead",
         "Hawkes; first 30 min have no Hawkes fit, so both modes use the EWMA there). OFI standardised causally",
         "to sd 0.20 in both modes. Fills: replay of the next second's real trades, FIFO at the touch, no RNG.",
         f"Uncertainty: Newey-West (lag {HAC_LAG} s) on per-second PnL increment differences.", "",
         "Replay runs:"]
    L.append(runs.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    L += ["", "Paired differences (final PnL, $):"]
    for _, r in paired.iterrows():
        flag = "excludes 0" if r["excludes_zero"] else "includes 0"
        L.append(f"  [{r['config']}] {r['comparison']:32s} {r['diff']:9.2f}  95% CI [{r['ci95_lo']:9.2f}, "
                 f"{r['ci95_hi']:9.2f}]  ({flag})")
    L += ["", "Robustness: parametric fills with the same EWMA pressure in both modes (10 seeds, t-interval):"]
    for _, r in param.iterrows():
        L.append(f"  [{r['config']}] signal {r['signal']:3s} hawkes - baseline {r['mean_diff']:9.2f}  "
                 f"95% CI [{r['ci95_lo']:9.2f}, {r['ci95_hi']:9.2f}]")
    L += ["", "Caveats: one day; queue ahead of us is never reduced by cancellations (fills are late, i.e.",
          "conservative); our quote is assumed live for the whole next second (no latency); behind the touch",
          "only trades printed through our price fill us; MTM at mid with no liquidation cost."]
    (OUT / "step4b_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    for p in (BIN, FEATURES, TRADES):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: run `make build` and scripts/step2b_causal_features.py")
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / "_scratch"
    scratch.mkdir(exist_ok=True)
    runs, paired = replay_comparison(scratch)
    param = parametric_robustness(scratch)
    for f in scratch.glob("*"):
        f.unlink()
    scratch.rmdir()
    runs.to_csv(OUT / "step4b_replay_runs.csv", index=False)
    paired.to_csv(OUT / "step4b_paired.csv", index=False)
    param.to_csv(OUT / "step4b_parametric_robustness.csv", index=False)
    write_report(runs, paired, param)


if __name__ == "__main__":
    main()
