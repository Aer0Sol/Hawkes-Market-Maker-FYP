import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
BIN = ROOT / "bin" / "step3_mm"  # build with `make build`
FEATURES = ROOT / "output" / "step2_hawkes" / "feature_table_step2_1s.csv"
BENCH_DIR = ROOT / "benchmark_results"
RUNS_DIR = BENCH_DIR / "runs"

BENCH_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

N_WARMUP = 3
N_RUNS = 20

ENGINE_ARGS = {
    "gamma": 0.10,
    "k": 6.0,
    "A": 0.08,
    "inventory_limit": 500,
    "max_order_size": 25,
    "tick_size": 0.01,
    "min_spread_ticks": 2,
    "use_hawkes": True,
    "ofi_skew_coef": 0.03,
    "microprice_skew_coef": 0.10,
    "latency_budget_us": 500.0,
    "maker_fee_bps": 0.20,
    "queue_haircut_base": 0.25,
    "toxicity_haircut_strength": 0.60,
    "toxic_ofi_threshold": 0.20,
    "toxic_side_widen_ticks": 3.0,
    "markout_horizon_steps": 5,
}


def percentile(vals, q):
    if not vals:
        return math.nan
    vals = sorted(vals)
    idx = (len(vals) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return vals[int(idx)]
    frac = idx - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (math.isnan(val) or math.isinf(val)) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [make_json_safe(x) for x in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if pd.isna(obj) and not isinstance(obj, str):
        return None
    return obj


def build_cmd(outdir: Path):
    cmd = [
        str(BIN),
        "--features", str(FEATURES),
        "--outdir", str(outdir),
        "--gamma", str(ENGINE_ARGS["gamma"]),
        "--k", str(ENGINE_ARGS["k"]),
        "--A", str(ENGINE_ARGS["A"]),
        "--inventory-limit", str(ENGINE_ARGS["inventory_limit"]),
        "--max-order-size", str(ENGINE_ARGS["max_order_size"]),
        "--tick-size", str(ENGINE_ARGS["tick_size"]),
        "--min-spread-ticks", str(ENGINE_ARGS["min_spread_ticks"]),
        "--ofi-skew-coef", str(ENGINE_ARGS["ofi_skew_coef"]),
        "--microprice-skew-coef", str(ENGINE_ARGS["microprice_skew_coef"]),
        "--latency-budget-us", str(ENGINE_ARGS["latency_budget_us"]),
        "--maker-fee-bps", str(ENGINE_ARGS["maker_fee_bps"]),
        "--queue-haircut-base", str(ENGINE_ARGS["queue_haircut_base"]),
        "--toxicity-haircut-strength", str(ENGINE_ARGS["toxicity_haircut_strength"]),
        "--toxic-ofi-threshold", str(ENGINE_ARGS["toxic_ofi_threshold"]),
        "--toxic-side-widen-ticks", str(ENGINE_ARGS["toxic_side_widen_ticks"]),
        "--markout-horizon-steps", str(ENGINE_ARGS["markout_horizon_steps"]),
    ]
    if ENGINE_ARGS["use_hawkes"]:
        cmd.append("--use-hawkes")
    return cmd


def get_peak_rss_kb_from_time(stderr_text: str):
    for line in stderr_text.splitlines():
        if "Maximum resident set size" in line:
            try:
                return int(line.split(":")[-1].strip())
            except Exception:
                return math.nan
    return math.nan


def run_once(run_idx: int, warmup: bool = False):
    outdir = RUNS_DIR / f"run_{run_idx:03d}"
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = build_cmd(outdir)
    # Peak RSS needs GNU time; without it RSS is reported as missing (getrusage on
    # children would count the forked Python parent before exec, which is misleading).
    have_gnu_time = os.access("/usr/bin/time", os.X_OK)
    timed_cmd = (["/usr/bin/time", "-v"] + cmd) if have_gnu_time else cmd

    t0 = time.perf_counter_ns()
    proc = subprocess.run(
        timed_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    t1 = time.perf_counter_ns()

    wall_ms = (t1 - t0) / 1_000_000.0
    peak_rss_kb = get_peak_rss_kb_from_time(proc.stderr) if have_gnu_time else math.nan

    summary_path = outdir / "step3_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    n_steps = int(summary["n_steps"])
    throughput_eps = n_steps / (wall_ms / 1000.0)

    row = {
        "run_idx": int(run_idx),
        "warmup": bool(warmup),
        "wall_ms": float(wall_ms),
        "throughput_events_per_sec": float(throughput_eps),
        "peak_rss_kb": None if pd.isna(peak_rss_kb) else int(peak_rss_kb),
        "n_steps": n_steps,
        "final_mtm_pnl": float(summary["final_mtm_pnl"]),
        "mean_mtm_pnl": float(summary["mean_mtm_pnl"]),
        "std_mtm_pnl": float(summary["std_mtm_pnl"]),
        "fill_count_total": int(summary["fill_count_total"]),
        "avg_bid_fill_prob": float(summary["avg_bid_fill_prob"]),
        "avg_ask_fill_prob": float(summary["avg_ask_fill_prob"]),
        "mean_latency_us_engine": float(summary["mean_latency_us"]),
        "p50_latency_us_engine": float(summary["p50_latency_us"]),
        "p95_latency_us_engine": float(summary["p95_latency_us"]),
        "p99_latency_us_engine": float(summary["p99_latency_us"]),
        "max_latency_us_engine": float(summary["max_latency_us"]),
        "latency_budget_exceeded_count": int(summary["latency_budget_exceeded_count"]),
    }
    return row


def main():
    if not BIN.exists():
        raise FileNotFoundError(f"Missing binary: {BIN}")
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing features CSV: {FEATURES}")

    rows = []

    print(f"Warmup runs: {N_WARMUP}")
    for i in range(N_WARMUP):
        row = run_once(i, warmup=True)
        rows.append(row)
        print(f"[warmup {i+1}/{N_WARMUP}] wall_ms={row['wall_ms']:.3f}, throughput={row['throughput_events_per_sec']:.1f} ev/s")

    print(f"\nMeasured runs: {N_RUNS}")
    for i in range(N_RUNS):
        row = run_once(i + N_WARMUP, warmup=False)
        rows.append(row)
        print(
            f"[run {i+1}/{N_RUNS}] wall_ms={row['wall_ms']:.3f}, "
            f"throughput={row['throughput_events_per_sec']:.1f} ev/s, "
            f"p99_engine={row['p99_latency_us_engine']:.6f} us"
        )

    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "benchmark_raw_runs.csv", index=False)

    measured = df[df["warmup"] == False].copy()

    summary = {
        "n_measured_runs": int(len(measured)),
        "wall_ms_mean": float(measured["wall_ms"].mean()),
        "wall_ms_median": float(measured["wall_ms"].median()),
        "wall_ms_p95": float(percentile(measured["wall_ms"].tolist(), 0.95)),
        "wall_ms_p99": float(percentile(measured["wall_ms"].tolist(), 0.99)),
        "throughput_eps_mean": float(measured["throughput_events_per_sec"].mean()),
        "throughput_eps_median": float(measured["throughput_events_per_sec"].median()),
        "throughput_eps_p05": float(percentile(measured["throughput_events_per_sec"].tolist(), 0.05)),
        "throughput_eps_p95": float(percentile(measured["throughput_events_per_sec"].tolist(), 0.95)),
        "peak_rss_kb_mean": float(measured["peak_rss_kb"].mean()),
        "peak_rss_kb_max": float(measured["peak_rss_kb"].max()),
        "engine_p50_us_median": float(measured["p50_latency_us_engine"].median()),
        "engine_p95_us_median": float(measured["p95_latency_us_engine"].median()),
        "engine_p99_us_median": float(measured["p99_latency_us_engine"].median()),
        "engine_max_us_max": float(measured["max_latency_us_engine"].max()),
        "latency_budget_exceeded_total": int(measured["latency_budget_exceeded_count"].sum()),
    }

    safe_summary = make_json_safe(summary)

    with open(BENCH_DIR / "benchmark_summary.json", "w") as f:
        json.dump(safe_summary, f, indent=2)

    md = []
    md.append("# Step 3 Engine Benchmark")
    md.append("")
    md.append(f"- Measured runs: {safe_summary['n_measured_runs']}")
    md.append(f"- Mean wall time: {safe_summary['wall_ms_mean']:.3f} ms")
    md.append(f"- Median wall time: {safe_summary['wall_ms_median']:.3f} ms")
    md.append(f"- p95 wall time: {safe_summary['wall_ms_p95']:.3f} ms")
    md.append(f"- Mean throughput: {safe_summary['throughput_eps_mean']:.2f} events/sec")
    md.append(f"- Median throughput: {safe_summary['throughput_eps_median']:.2f} events/sec")
    rss = safe_summary["peak_rss_kb_mean"]
    md.append(f"- Mean peak RSS: {rss:.2f} KB" if rss is not None else "- Mean peak RSS: n/a (GNU time not installed)")
    md.append(f"- Median engine p50 latency: {safe_summary['engine_p50_us_median']:.6f} us")
    md.append(f"- Median engine p95 latency: {safe_summary['engine_p95_us_median']:.6f} us")
    md.append(f"- Median engine p99 latency: {safe_summary['engine_p99_us_median']:.6f} us")
    md.append(f"- Worst observed engine max latency: {safe_summary['engine_max_us_max']:.6f} us")
    md.append(f"- Total latency budget exceedances: {int(safe_summary['latency_budget_exceeded_total'])}")

    (BENCH_DIR / "benchmark_report.md").write_text("\n".join(md))

    print("\nBenchmark complete.")
    print(json.dumps(safe_summary, indent=2))

if __name__ == "__main__":
    main()