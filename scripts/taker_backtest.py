"""Paper P&L of trading the 1 s forecasts directly (option 1), out of sample on BTCUSDT.

For each pair of consecutive days: fit the models on day d-1 (as step 6's cross-day test), predict day d,
hold +notional / -notional by the sign of each forecast for one second, trade only when the sign
flips, and charge a fee per dollar traded. Reported at 0, 1 and 10 bp per side (10 bp is Binance's
standard spot taker fee). The spread is ignored: one cent on ~$65k is ~0.0015 bp.

Writes output/taker_backtest/ and prints a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_binance as dl  # noqa: E402
import hawkes_diagnostics as hd  # noqa: E402
import step5_alpha_signal as s5  # noqa: E402
import step6_binance_validation as s6  # noqa: E402

OUT = Path("output/taker_backtest")
NOTIONAL = 10_000.0
FEES_BPS = (0.0, 1.0, 10.0)
TARGET = "future_ret_1s"
MODELS = {"baseline_ewma": s6.BASELINE_FEATURES, **s6.MODELS}


def strategy_pnl(pred: np.ndarray, ret: np.ndarray) -> dict:
    pos = np.sign(pred)
    gross = float(np.sum(pos * NOTIONAL * np.expm1(ret)))
    traded = float(NOTIONAL * (np.abs(pos[0]) + np.abs(np.diff(pos)).sum()))
    return {"gross": gross, "traded": traded, "flips": int((np.diff(pos) != 0).sum()),
            **{f"net_{f:g}bp": gross - traded * f / 1e4 for f in FEES_BPS}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dates = s6.DEFAULT_DATES
    days = {}
    for d in dates:
        print(f"[{d}] features ...", flush=True)
        days[d], _, _ = s6.build_feature_table(s6.load_trades(dl.OUTDIR / f"BTCUSDT-aggTrades-{d}.csv"))

    rows = []
    for prev, nxt in zip(dates[:-1], dates[1:]):
        p_df, n_df = days[prev], days[nxt]
        fit = hd.fit_discrete(p_df["buy_events"].to_numpy(float), p_df["sell_events"].to_numpy(float))["x"]
        for name, feats in MODELS.items():
            hp = fit if name == "with_hawkes" else None
            x_tr, y_tr = s6.day_design(p_df, feats, TARGET, hp)
            x_te, y_te = s6.day_design(n_df, feats, TARGET, hp)
            pred = s5.ridge_fit_predict(x_tr, y_tr, x_te, s5.RIDGE_ALPHA)
            rows.append({"test_day": nxt, "model": name, "seconds": len(y_te), **strategy_pnl(pred, y_te),
                         "max_abs_pred_bp": float(np.abs(pred).max() * 1e4)})
        # Context: holding a long position all day.
        _, y = s6.day_design(n_df, s6.BASELINE_FEATURES, TARGET, None)
        rows.append({"test_day": nxt, "model": "always_long", "seconds": len(y),
                     **strategy_pnl(np.ones_like(y), y), "max_abs_pred_bp": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "taker_backtest_by_day.csv", index=False)
    cols = ["gross", "traded", "flips"] + [f"net_{f:g}bp" for f in FEES_BPS]
    total = df.groupby("model", sort=False)[cols].sum()
    total["days_gross_positive"] = df.assign(pos=df["gross"] > 0).groupby("model", sort=False)["pos"].sum()
    total["max_abs_forecast_bp"] = df.groupby("model", sort=False)["max_abs_pred_bp"].max()
    total.to_csv(OUT / "taker_backtest_summary.csv")
    pd.set_option("display.width", 200)
    print(f"\nTrade the 1 s forecast, ${NOTIONAL:,.0f} long/short, 6 out-of-sample BTCUSDT days ({dates[1]} to {dates[-1]}):")
    print(total.to_string(float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
