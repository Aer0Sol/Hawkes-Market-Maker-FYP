"""Calibrate the engine's parametric fill model  lambda(delta) = A * pressure * exp(-k * delta).

For each second and distance delta, check whether next-second aggressive trades
reached mid + delta (asks) or mid - delta (bids), then fit

    Lambda(delta) = -ln(1 - P_touch(delta)),   ln Lambda = ln Lambda0 - k * delta,
    A = Lambda0 / mean pressure.

Touched is an upper bound on filled (queue position is ignored).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRADES = ROOT / "output" / "step2_hawkes" / "trades_labeled.parquet"
FEATURES = ROOT / "output" / "step2_hawkes" / "feature_table_step2_1s.parquet"
OUTDIR = ROOT / "output" / "fill_calibration"
TICK = 0.01
DELTAS = np.round(np.arange(0.005, 0.105, 0.005), 4)  # 0.5 to 10 ticks from mid


def touch_curve(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Per-second touch probabilities at each distance for both sides."""
    f = features[["ts", "midprice"]].copy()
    f["bucket"] = f["ts"]
    t = trades[["ts", "price", "trade_sign"]].copy()
    # Trade in [T+1, T+2) is referenced to the mid at the end of bucket T.
    t["bucket"] = t["ts"].dt.floor("1s") - pd.Timedelta("1s")
    t = t.merge(f[["bucket", "midprice"]], on="bucket", how="inner")
    t["dist"] = np.where(t["trade_sign"] == 1, t["price"] - t["midprice"], t["midprice"] - t["price"])

    n_buckets = len(f) - 1  # last bucket has no next second
    max_buy = t[t["trade_sign"] == 1].groupby("bucket")["dist"].max()
    max_sell = t[t["trade_sign"] == -1].groupby("bucket")["dist"].max()
    rows = []
    for d in DELTAS:
        p_ask = (max_buy >= d - 1e-9).sum() / n_buckets   # ask at mid+d touched by buyers
        p_bid = (max_sell >= d - 1e-9).sum() / n_buckets  # bid at mid-d touched by sellers
        rows.append({"delta": d, "p_touch_ask": p_ask, "p_touch_bid": p_bid})
    out = pd.DataFrame(rows)
    for side in ("ask", "bid"):
        p = out[f"p_touch_{side}"].clip(upper=1 - 1e-9)
        out[f"lambda_{side}"] = -np.log1p(-p)
    return out


def fit_exponential(curve: pd.DataFrame, side: str, min_events: float) -> tuple[float, float]:
    """Fit ln Lambda = ln Lambda0 - k delta on points with enough support."""
    c = curve[curve[f"p_touch_{side}"] * min_events >= 10]  # at least ~10 touched seconds
    slope, intercept = np.polyfit(c["delta"], np.log(c[f"lambda_{side}"]), 1)
    return float(np.exp(intercept)), float(-slope)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(TRADES)
    features = pd.read_parquet(FEATURES)
    curve = touch_curve(trades, features)
    n = len(features) - 1

    fits = {}
    for side, pressure_col in (("ask", "lambda_buy_ewm"), ("bid", "lambda_sell_ewm")):
        lam0, k = fit_exponential(curve, side, n)
        fits[side] = {"Lambda0_per_s": lam0, "k_per_dollar": k,
                      "mean_pressure_ewm": float(features[pressure_col].mean()),
                      "A_ewm": lam0 / float(features[pressure_col].mean())}
    hawkes_mean = float(features[["lambda_buy_hawkes", "lambda_sell_hawkes"]].mean().mean())
    k = float(np.mean([fits[s]["k_per_dollar"] for s in fits]))
    lam0 = float(np.mean([fits[s]["Lambda0_per_s"] for s in fits]))
    ewm_mean = float(np.mean([fits[s]["mean_pressure_ewm"] for s in fits]))
    result = {
        "method": "touch-probability curve, next-second trades vs end-of-bucket mid; see script docstring",
        "n_buckets": n,
        "delta_grid_dollars": DELTAS.tolist(),
        "per_side": fits,
        "k_per_dollar": k,
        "A": lam0 / ewm_mean,
        "A_hawkes_pressure": lam0 / hawkes_mean,
        "implied_half_spread_gamma_0.1": (1 / 0.1) * np.log1p(0.1 / k),
        "caveats": [
            "touch is an upper bound on fill: queue position is ignored (engine queue haircut applies on top)",
            "one day, one symbol; exponential shape is an approximation, see the plot",
            "A is scaled by the mean EWMA pressure; the Hawkes-pressure equivalent is A_hawkes_pressure",
        ],
    }
    (OUTDIR / "fill_model_calibration.json").write_text(json.dumps(result, indent=2) + "\n")
    curve.to_csv(OUTDIR / "touch_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for side, mk in (("ask", "o"), ("bid", "s")):
        ax.semilogy(curve["delta"] / TICK, curve[f"lambda_{side}"], mk, label=f"{side} side (empirical)")
        f = fits[side]
        ax.semilogy(curve["delta"] / TICK, f["Lambda0_per_s"] * np.exp(-f["k_per_dollar"] * curve["delta"]),
                    "-", label=f"{side} fit: k={f['k_per_dollar']:.0f}/$")
    ax.axvline(16.5, color="grey", ls="--", lw=1)
    ax.text(16.6, ax.get_ylim()[0] * 3, "legacy quote\ndistance", fontsize=8, color="grey")
    ax.set_xlabel("distance from mid (ticks)")
    ax.set_ylabel("touch intensity per second")
    ax.set_title("Fill-model calibration: touch intensity vs quote distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTDIR / "touch_curve.png", dpi=160)
    print(json.dumps({k_: result[k_] for k_ in ("k_per_dollar", "A", "A_hawkes_pressure",
                                                 "implied_half_spread_gamma_0.1")}, indent=2))


if __name__ == "__main__":
    main()
