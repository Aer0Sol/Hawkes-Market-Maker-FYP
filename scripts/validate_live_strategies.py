"""Out-of-sample validation of every live paper strategy, by replaying downloaded trades through the live code.

For each test day d of a pair, the live model file is rebuilt from day d-1 only (Hawkes fits, ridge models,
maker calibration: scripts/train_pair.py's code), and day d's aggTrades are fed trade by trade through
scripts/live_monitor.py's Engine, so the market makers (one-second and event-driven), the forecast takers and
the execution books (slow and fast) run exactly as on the dashboard.

Historical aggTrades carry no order book. The best bid / ask is rebuilt from the trades themselves (last
sell-initiated price = bid, last buy-initiated price = ask, at least one tick apart), with unknown displayed
size, so resting orders fill only when a trade prints through their price (conservative: no at-price fills).
Book updates take the trade's timestamp, so event-driven books see them with their 50 ms latency.

  uv run --group live python scripts/validate_live_strategies.py --symbol ETHUSDT --dates 2026-11-10 ... 2026-11-16
writes output/validation/<SYMBOL>/days.csv (one row per test day and strategy) and orders.csv (execution orders).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pair_model as bp  # noqa: E402
import download_binance as dl  # noqa: E402
import hawkes_diagnostics as hd  # noqa: E402
import live_monitor as lm  # noqa: E402
import step6_binance_validation as s6  # noqa: E402
import train_pair as tp  # noqa: E402

OUT_ROOT = Path("output/validation")
BIG = 1e18   # displayed size when unknown: at-price trades never reach us, only through-trades fill


def model_for(sym: str, day: str, tick: float) -> dict:
    tr, table = tp.load_day(sym, day)
    model, _ = bp.build(sym, day, tick, trades=tr, table=table)
    hp = np.array([model["hawkes_params"][k] for k in hd.PARAM_NAMES])
    model["models"], model["exact_hawkes"] = tp.live_models(table[0], tr, hp)
    return model


def replay_day(model: dict, path: Path, tick: float) -> tuple[dict, list]:
    eng = lm.Engine(model, paper=lm.PaperTaker(10_000.0, 10.0),
                    maker_kwargs={"notional": 10_000.0, "maker_fee_bps": 10.0, "inventory_limit": 50_000.0,
                                  "latency_ms": 50.0},
                    exec_kwargs={"notional": 10_000.0, "taker_fee_bps": 10.0, "maker_fee_bps": 10.0, "fast_exec": True},
                    models=lm.MODEL_NAMES)   # every model in every family: this is where the pruning is decided
    eng.executor.order_log = []
    bid = ask = float("nan")
    with open(path, newline="") as f:
        rows = csv.reader(f)
        first = next(rows)
        scale = 1000 if int(first[5]) > 1e14 else 1          # microsecond files (2025+)
        for row in itertools.chain([first], rows):
            t_ms = int(row[5]) // scale
            if eng.bucket_ms is not None:
                while t_ms >= eng.bucket_ms + 1000:
                    eng.close_bucket()
            price, qty, buyer_is_maker = float(row[1]), float(row[2]), row[6] == "True"
            eng.on_trade(t_ms, price, qty, buyer_is_maker)
            if buyer_is_maker:            # sell-initiated: it traded at the bid
                bid = price
                if not ask > bid:
                    ask = bid + tick
            else:
                ask = price
                if not bid < ask:
                    bid = ask - tick
            if bid == bid and ask == ask:
                eng.on_book(bid, BIG, ask, BIG, t_ms)
    return eng.scoreboard(), eng.executor.order_log


def flatten(sc: dict, day: str) -> list[dict]:
    out = []
    for k, v in sc["maker"].items():
        if isinstance(v, dict):
            out.append({"day": day, "family": "maker", "strategy": k, "fills": v["fills"], "volume": v["volume"],
                        "pnl_zero_fee": v["pnl_zero_fee"], "pnl_net": v["pnl_net"],
                        "edge_0fee_bp": 1e4 * v["pnl_zero_fee"] / v["volume"] if v["volume"] else np.nan,
                        "markout_10s_bp": v["markout_10s_bp"]})
    for k, v in sc["paper"].items():
        if isinstance(v, dict):
            out.append({"day": day, "family": "taker", "strategy": k, "pnl_zero_fee": v["gross"], "pnl_net": v["net"],
                        "volume": v["traded"],
                        "edge_0fee_bp": 1e4 * v["gross"] / v["traded"] if v["traded"] else np.nan})
    for k, v in sc["execution"].items():
        if isinstance(v, dict):
            out.append({"day": day, "family": "execution", "strategy": k, "orders": v["orders"],
                        "pnl_net": v["saved"], "saving_bp": v["saving_bp"], "maker_share": v["maker_share"]})
    for k, v in sc.items():
        if isinstance(v, dict) and "hit" in v:
            out.append({"day": day, "family": "forecast", "strategy": k, "hit": v["hit"], "r2": v["r2"],
                        "mse_gain": v.get("mse_gain")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--tick", type=float)
    args = ap.parse_args()
    sym = args.symbol.upper()
    for d in args.dates:
        dl.check_not_holdout(sym, d, False)
    tick = args.tick or bp.tick_size(sym)
    s6.TICK = tick
    out = OUT_ROOT / sym
    out.mkdir(parents=True, exist_ok=True)
    rows, orders = [], []
    for prev, nxt in zip(args.dates[:-1], args.dates[1:]):
        t0 = time.time()
        model = model_for(sym, prev, tick)
        t1 = time.time()
        sc, log = replay_day(model, dl.OUTDIR / f"{sym}-aggTrades-{nxt}.csv", tick)
        rows += flatten(sc, nxt)
        orders += [{"day": nxt, "arrival_ms": a, "strategy": b, "saving_bp": s, "maker_share": m} for a, b, s, m in log]
        print(f"[{sym} {prev} -> {nxt}] model {t1 - t0:.0f}s, replay {time.time() - t1:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "days.csv", index=False)
        pd.DataFrame(orders).to_csv(out / "orders.csv", index=False)
    print(f"Wrote {out}/days.csv, orders.csv")


if __name__ == "__main__":
    main()
