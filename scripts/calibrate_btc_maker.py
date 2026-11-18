"""Calibrate the paper market maker's Avellaneda-Stoikov fill decay k for BTCUSDT.

Same estimator as scripts/calibrate_fill_model.py (per-second touch probability at each distance from the
end-of-second price, fitted as ln Lambda = ln Lambda0 - k delta), on a BTCUSDT day, with a dollar-scale grid.
Writes output/step6_binance_validation/maker_calibration.json and adds a "maker" block to live_model.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_fill_model as cf  # noqa: E402
import download_binance as dl  # noqa: E402
import step6_binance_validation as s6  # noqa: E402

OUT = Path("output/step6_binance_validation")
DAY = s6.DEFAULT_DATES[-1]
DELTAS = np.round(np.concatenate([np.arange(0.01, 1.0, 0.05), np.arange(1.0, 12.0, 0.5)]), 2)
GAMMA = 0.01     # per $: inventory skew of ~$1.5 at the $50k cap with tau = 10 s
TAU_S = 10.0     # risk horizon in seconds; the maker re-quotes every second


def main():
    tr = s6.load_trades(dl.OUTDIR / f"BTCUSDT-aggTrades-{DAY}.csv")
    df, _, _ = s6.build_feature_table(tr)
    cf.DELTAS = DELTAS
    curve = cf.touch_curve(tr[["ts", "price", "trade_sign"]], df[["ts", "mid"]].rename(columns={"mid": "midprice"}))
    fits = {side: cf.fit_exponential(curve, side, len(df) - 1) for side in ("ask", "bid")}
    k = float(np.mean([fits[s][1] for s in fits]))
    cal = {"calibrated_on": DAY, "k_per_dollar": k, "Lambda0_per_s": float(np.mean([fits[s][0] for s in fits])),
           "by_side": {s: {"Lambda0": a, "k": kk} for s, (a, kk) in fits.items()},
           "p_touch_at_1_cent": float(curve.loc[0, ["p_touch_ask", "p_touch_bid"]].mean()),
           "gamma_per_dollar": GAMMA, "tau_s": TAU_S,
           "note": "Two regimes: ~38% of seconds reach 1 cent from the price (the bid-ask bounce), ~2% reach 25 cents; "
                   "k describes the tail beyond the first few cents."}
    curve.to_csv(OUT / "maker_touch_curve.csv", index=False)
    (OUT / "maker_calibration.json").write_text(json.dumps(cal, indent=2))
    live = json.loads((OUT / "live_model.json").read_text())
    live["maker"] = {"k": k, "gamma": GAMMA, "tau_s": TAU_S, "calibrated_on": DAY}
    (OUT / "live_model.json").write_text(json.dumps(live, indent=2))
    print(json.dumps(cal, indent=2))


if __name__ == "__main__":
    main()
