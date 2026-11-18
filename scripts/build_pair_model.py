"""Build a live-monitor model file for another Binance pair, for the model-free books.

The live monitor needs, for any symbol: the 1 s Hawkes parameters and EWMA span (to compute and chart order-flow
features), the market maker's Avellaneda-Stoikov fill decay k and risk aversion gamma, the tick size and the bounce
filter. This fits those on one day of the pair's aggTrades and writes output/pairs/<SYMBOL>/live_model.json with no
forecasting models, so the monitor runs only the touch and A-S market makers and passive execution without a signal.
(scripts/train_pair.py adds forecasting models validated across several days of the pair.)

  uv run python scripts/download_binance.py --symbol NEARUSDT --date 2026-11-16
  uv run python scripts/build_pair_model.py --symbol NEARUSDT --date 2026-11-16
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_fill_model as cf  # noqa: E402
import download_binance as dl  # noqa: E402
import hawkes_diagnostics as hd  # noqa: E402
import step5_alpha_signal as s5  # noqa: E402
import step6_binance_validation as s6  # noqa: E402

BTC_GAMMA, BTC_PRICE = 0.01, 65_000.0   # keep gamma * price, the risk aversion relative to price, as for BTCUSDT


def tick_size(symbol: str) -> float:
    info = json.load(urllib.request.urlopen(f"https://api.binance.com/api/v3/exchangeInfo?symbol={symbol}", timeout=20))
    f = {x["filterType"]: x for x in info["symbols"][0]["filters"]}
    return float(f["PRICE_FILTER"]["tickSize"])


def build(sym: str, date: str, tick: float, trades=None, table=None):
    """Model-free live model for `sym` (Hawkes features, maker calibration, tick) fitted on one day.
    `trades` and `table` (the output of s6.build_feature_table) are recomputed when not given."""
    if trades is None:
        trades = s6.load_trades(dl.OUTDIR / f"{sym}-aggTrades-{date}.csv")
    df, hawkes_fit, _ = table if table is not None else s6.build_feature_table(trades)
    price = float(df["mid"].median())
    p = hawkes_fit["params"]
    st = hd.stability(np.array([p[k] for k in hd.PARAM_NAMES]))

    cf.DELTAS = np.round(tick * np.arange(1, 61), 10)
    curve = cf.touch_curve(trades[["ts", "price", "trade_sign"]], df[["ts", "mid"]].rename(columns={"mid": "midprice"}))
    fits = {side: cf.fit_exponential(curve, side, len(df) - 1) for side in ("ask", "bid")}
    k = float(np.mean([fits[s][1] for s in fits]))
    model = {
        "symbol": sym, "trained_on": date, "tick": tick, "hit_min_abs_move": 1.5 * tick,
        "target": "ln(last price at t+1 s) - ln(last price at t)",
        "hawkes_params": p, "ewma_span": 60, "rv_window": 60, "event_half_lives_s": list(s5.EVENT_HALF_LIVES_S),
        "maker": {"k": k, "gamma": BTC_GAMMA * BTC_PRICE / price, "tau_s": 10.0, "calibrated_on": date},
        "models": {},
        "notes": {"median_price": price, "tick_bp": 1e4 * tick / price,
                  "events": int(df[["buy_events", "sell_events"]].sum().sum()),
                  "hawkes_branching": st["spectral_radius_branching"], "hawkes_half_life_s": st["half_life_buckets"],
                  "p_touch_1_tick": float(curve.loc[0, ["p_touch_ask", "p_touch_bid"]].mean())},
    }
    return model, curve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--tick", type=float, help="tick size; fetched from Binance if omitted")
    args = ap.parse_args()
    sym = args.symbol.upper()
    model, curve = build(sym, args.date, args.tick or tick_size(sym))
    out_dir = Path("output/pairs") / sym
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_model.json").write_text(json.dumps(model, indent=2))
    curve.to_csv(out_dir / "touch_curve.csv", index=False)
    print(json.dumps({k_: model[k_] for k_ in ("symbol", "tick", "maker", "notes")}, indent=2))


if __name__ == "__main__":
    main()
