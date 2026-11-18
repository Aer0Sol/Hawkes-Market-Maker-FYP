"""Reconcile the live paper market maker against an offline replay of the same recorded session.

The live maker (scripts/live_monitor.py) fills its quotes trade by trade, updating the queue ahead from every
book update. The C++ engine's replay venue (--fill-model replay) works per second: a quote placed at the end of
second t meets the trades of second t+1, with the queue ahead taken from the book at the end of second t. This
script replays the recorded quotes (data/live/<date>/maker_quotes.csv) through that per-second rule, implemented
in replay_fill below exactly as in engine/event_driven_mm_engine.cpp, and compares fills, volume and P&L with
the live fills (maker_fills.csv). The difference measures what the 1 s replay simplification costs.

Read the per-fill markouts, not the session P&L. The replay has no inventory control of its own: it replays quotes
that were skewed and capped by the live book's inventory, so once its fills differ its position random-walks
(about sqrt(fills) x order size) and its session P&L is mostly that unhedged drift. The 10 s markout (mid 10 s
after the fill vs the fill price, bp of volume) measures each fill's quality whatever the inventory.

Usage: uv run python scripts/reconcile_live_maker.py --date 2026-11-17
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LIVE = Path("data/live")
OUT = Path("output/live_reconciliation")


@dataclass
class Resting:
    live: bool = False
    price: float = float("nan")
    queue_ahead: float = 0.0


def replay_fill(order: Resting, active: bool, quote_px: float, best_px: float, displayed: float, is_bid: bool,
                size: float, tick: float, trades: np.ndarray | None) -> float:
    """Port of replay_fill() in engine/event_driven_mm_engine.cpp. trades: rows of (price, size, sign)."""
    if not active:
        order.live = False
        return 0.0
    eps = tick / 2
    if not order.live or abs(order.price - quote_px) > eps:
        at_touch = abs(quote_px - best_px) < eps
        order.live, order.price = True, quote_px
        order.queue_ahead = (0.0 if math.isnan(displayed) else displayed) if at_touch else math.inf
    if trades is None or len(trades) == 0:
        return 0.0
    hitting = -1 if is_bid else 1
    t = trades[trades[:, 2] == hitting]
    gap = (quote_px - t[:, 0]) if is_bid else (t[:, 0] - quote_px)
    through = t[gap > eps, 1].sum()
    at_price = t[np.abs(gap) <= eps, 1].sum()
    if through > 0:
        filled = size
    elif math.isfinite(order.queue_ahead):
        filled = max(0.0, min(size, at_price - order.queue_ahead))
    else:
        filled = 0.0
    if math.isfinite(order.queue_ahead):
        order.queue_ahead = max(0.0, order.queue_ahead - at_price)
    if filled > 0:
        order.live = False
    return filled


def load_session(date: str):
    d = LIVE / date
    trades = pd.read_csv(d / "aggTrades.csv", header=None,
                         names=["a", "price", "qty", "f", "l", "T", "m", "M"])
    trades["sign"] = np.where(trades["m"].astype(str) == "True", -1, 1)
    trades["sec"] = trades["T"] // 1000 * 1000
    book = pd.read_csv(d / "buckets_1s.csv")
    book["sec"] = pd.to_datetime(book["ts_utc"], format="ISO8601").astype("int64") // 10**6
    quotes = pd.read_csv(d / "maker_quotes.csv")
    fills = pd.read_csv(d / "maker_fills.csv")
    return trades, book.set_index("sec"), quotes, fills


def replay_session(trades, book, quotes, notional: float, tick: float = 0.01) -> pd.DataFrame:
    by_sec = {s: g[["price", "qty", "sign"]].to_numpy(float) for s, g in trades.groupby("sec")}
    rows = []
    for name, q in quotes.sort_values("ts_ms").groupby("book"):
        bid_o, ask_o = Resting(), Resting()
        for r in q.itertuples(index=False):
            sec_of_book = r.ts_ms - 1000          # the book snapshot at the moment of quoting
            if sec_of_book not in book.index:
                continue
            bk = book.loc[sec_of_book]
            size = notional / r.mid
            nxt = by_sec.get(r.ts_ms)             # trades in the second after the quote
            for side, o, px, best, disp in ((1, bid_o, r.bid, bk["bid"], bk["bid_qty"]),
                                            (-1, ask_o, r.ask, bk["ask"], bk["ask_qty"])):
                active = isinstance(px, float) and not math.isnan(px)
                qty = replay_fill(o, active, px if active else 0.0, best, disp, side == 1, size, tick, nxt)
                if qty > 0:
                    rows.append({"ts_ms": r.ts_ms, "book": name, "side": "buy" if side == 1 else "sell",
                                 "price": px, "qty_btc": qty})
    return pd.DataFrame(rows, columns=["ts_ms", "book", "side", "price", "qty_btc"])


def summarise(fills: pd.DataFrame, final_mid: float, mid_by_sec: pd.Series) -> pd.DataFrame:
    f = fills.assign(sgn=np.where(fills["side"] == "buy", 1, -1))
    f["usd"] = f["price"] * f["qty_btc"]
    later = mid_by_sec.reindex((f["ts_ms"] // 1000 * 1000 + 10_000).to_numpy()).to_numpy()
    f["mk10_usd"] = f["sgn"] * (later - f["price"]) * f["qty_btc"]
    ok = np.isfinite(later)
    g = f.groupby("book")
    out = pd.DataFrame({"fills": g.size(), "volume_usd": g["usd"].sum(),
                        "net_btc": g.apply(lambda x: (x["sgn"] * x["qty_btc"]).sum(), include_groups=False),
                        "cash": g.apply(lambda x: -(x["sgn"] * x["usd"]).sum(), include_groups=False)})
    out["pnl_zero_fee"] = out["cash"] + out["net_btc"] * final_mid
    m = f[ok].groupby("book")
    out["markout_10s_bp"] = 1e4 * m["mk10_usd"].sum() / m["usd"].sum()
    return out.drop(columns="cash")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--notional", type=float, default=10_000.0)
    args = ap.parse_args()
    trades, book, quotes, live = load_session(args.date)
    if live.empty or quotes.empty:
        raise SystemExit(f"No recorded maker quotes/fills under {LIVE / args.date}: run `make live` first.")
    start = quotes["ts_ms"].min()
    live = live[live["ts_ms"] >= start]
    replay = replay_session(trades, book, quotes, args.notional)
    final_mid = float(book["last"].dropna().iloc[-1])
    mid = ((book["bid"] + book["ask"]) / 2).fillna(book["last"])
    cmp = summarise(live, final_mid, mid).join(summarise(replay, final_mid, mid), lsuffix="_live", rsuffix="_replay",
                                               how="outer")
    cmp["fill_ratio_replay_vs_live"] = cmp["volume_usd_replay"] / cmp["volume_usd_live"]
    OUT.mkdir(parents=True, exist_ok=True)
    cmp.to_csv(OUT / f"reconciliation_{args.date}.csv")
    replay.to_csv(OUT / f"replay_fills_{args.date}.csv", index=False)
    span = (quotes["ts_ms"].max() - start) / 60_000
    pd.set_option("display.width", 200)
    print(f"Session {args.date}: {span:.0f} min of recorded quotes, {len(live)} live fills, {len(replay)} replay fills")
    cols = ["fills_live", "fills_replay", "fill_ratio_replay_vs_live", "markout_10s_bp_live", "markout_10s_bp_replay",
            "net_btc_live", "net_btc_replay"]
    print(cmp[cols].round(2).to_string())
    print("Compare markouts (fill quality); session P&L and net BTC of the replay are inventory drift (see docstring).")


if __name__ == "__main__":
    main()
