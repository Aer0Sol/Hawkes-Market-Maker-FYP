"""Live order-flow monitor for one Binance pair: the project's Hawkes models on the real-time feed, paper only.

One process does four things:
  1. collects Binance's public aggTrade and bookTicker streams (no key, no orders),
  2. rebuilds, every second, the exact features the models were trained on (EWMA and one-step-ahead Hawkes
     intensities, event-time decayed counts, the exact Hawkes intensity, realized vol, trade imbalance), predicts
     the next second's return with every model in the model file and scores each prediction when its second
     closes: a live, genuinely out-of-sample test;
  3. runs the paper strategies on the live book: forecast takers (PaperTaker), market makers (PaperMaker:
     one-second and event-driven) and execution books (Executor);
  4. serves a dashboard at http://localhost:<port>/ and streams updates over a WebSocket.

It records what it sees under --record/<UTC date>/: aggTrades.csv in Binance's historical format, buckets_1s.csv
with the best bid/ask each second (which the historical files lack), and the maker's quotes and fills.
Several pairs run side by side through scripts/live_hub.py (config/live_pairs.json).

Run:   uv run --group live python scripts/live_monitor.py
Replay (no network): uv run --group live python scripts/live_monitor.py \
           --replay data/binance/BTCUSDT-aggTrades-2024-06-09.csv --speed 20
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

import websockets
from websockets.asyncio.server import serve

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "output" / "step6_binance_validation" / "live_model.json"
DASHBOARD = Path(__file__).resolve().parent / "live_dashboard.html"
STREAM_URL = "wss://stream.binance.com:9443/stream?streams={s}@aggTrade/{s}@bookTicker"
GRACE_MS = 300          # wait this long past a second's end for late trades before closing it
HISTORY = 900           # seconds kept for the dashboard
MODEL_NAMES = ("baseline_ewma", "with_hawkes", "event_time", "hawkes_ensemble")  # those present in live_model.json run
# Which models drive which strategy family (scripts/validate_live_strategies.py replays each out of sample);
# execution also marks the Hawkes model that validated best per pair (display.exec_model).
# Execution drops +Event-time: in the replay it never beat the ensemble's execution, whose signal contains it.
FAMILY_MODELS = {"taker": MODEL_NAMES, "maker": MODEL_NAMES,
                 "execution": ("baseline_ewma", "with_hawkes", "hawkes_ensemble")}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


@dataclass
class Score:
    n: int = 0
    n_nonzero: int = 0
    hits: int = 0
    sse: float = 0.0
    sst: float = 0.0


@dataclass
class PaperTaker:
    """Forecast taker: trade each model's forecast directly. Hold +notional or -notional according to the sign of
    the 1 s forecast, trade only when the sign flips, pay the taker fee on every dollar traded. Paper only."""

    notional: float = 10_000.0
    fee_bps: float = 10.0
    pos: dict = field(default_factory=lambda: {m: 0 for m in MODEL_NAMES})
    gross: dict = field(default_factory=lambda: {m: 0.0 for m in MODEL_NAMES})
    fees: dict = field(default_factory=lambda: {m: 0.0 for m in MODEL_NAMES})
    traded: dict = field(default_factory=lambda: {m: 0.0 for m in MODEL_NAMES})

    def use(self, names):
        self.pos = {m: 0 for m in names}
        self.gross = {m: 0.0 for m in names}
        self.fees = {m: 0.0 for m in names}
        self.traded = {m: 0.0 for m in names}

    def mark(self, ret: float):
        """P&L of the positions held over the second that just closed (log return -> simple return)."""
        move = math.expm1(ret)
        for m, q in self.pos.items():
            self.gross[m] += q * self.notional * move

    def rebalance(self, preds: dict):
        for m, pred in preds.items():
            if m not in self.pos:
                continue                  # model not traded by the taker (FAMILY_MODELS)
            new = (pred > 0) - (pred < 0)
            dq = abs(new - self.pos[m])
            if dq:
                self.traded[m] += dq * self.notional
                self.fees[m] += dq * self.notional * self.fee_bps / 1e4
                self.pos[m] = new

    def snapshot(self) -> dict:
        return {m: {"pos": self.pos[m], "gross": self.gross[m], "fees": self.fees[m],
                    "net": self.gross[m] - self.fees[m], "traded": self.traded[m]} for m in self.pos}

@dataclass
class _Order:
    live: bool = False
    price: float = float("nan")
    queue_ahead: float = 0.0   # base-asset quantity ahead of us at our price (inf when behind the touch)
    remaining: float = 0.0     # base-asset quantity still to fill


@dataclass
class _MakerBook:
    mode: str                  # "touch" (control) or "as" (Avellaneda-Stoikov)
    signal: str | None         # model whose forecast skews and protects the quotes, or None
    inv: float = 0.0           # base asset
    cash: float = 0.0          # quote asset (USDT)
    fees: float = 0.0
    volume: float = 0.0        # USDT filled
    fills: int = 0
    spread_captured: float = 0.0   # sum of side * (mid at fill - fill price) * qty
    markout_1s: float = 0.0        # sum of side * (mid 1 s later - fill price) * qty
    markout_10s: float = 0.0
    marked_1s: float = 0.0         # USDT volume whose 1 s markout has resolved
    marked_10s: float = 0.0
    bid: _Order = field(default_factory=_Order)
    ask: _Order = field(default_factory=_Order)
    z_n: int = 0                   # causal running mean/variance of the forecast, for standardisation
    z_mean: float = 0.0
    z_m2: float = 0.0
    pulls: int = 0
    widens: int = 0
    last_quote: tuple = (float("nan"), float("nan"))
    hawkes_pull: bool = False      # event-driven book reacting to the per-trade exact Hawkes intensity
    last_pred: float | None = None # latest per-second forecast and its standardised score (event-driven books)
    last_z: float = 0.0
    burst_pulls: int = 0


class PaperMaker:
    """Paper market makers on the pair's live book. Paper only: nothing is sent anywhere.

    Books:
      touch         control: $notional on both sides at the best bid and ask, no model.
      as_none       Avellaneda-Stoikov quotes from the C++ engine, no signal.
      as_<model>    the same, plus the model's forecast: the reservation price moves by the expected price change
                    (mid * forecast), and the forecast, causally standardised and scaled to sd 0.2 as the engine's
                    --ofi-standardize does, drives the engine's toxicity logic: widen the threatened side above 0.2
                    (1 sd) and pull it above 0.45 (2.25 sd).
    A-S, as in the engine: r = mid - q*gamma*sigma^2*tau (+ mid*forecast), half-spread
    max((1/gamma) ln(1 + gamma/k) + gamma*sigma^2*tau/2, touch half-spread), rounded outward to the tick and
    clamped to never improve the best bid/ask. sigma is the live 60 s realised volatility in $/sqrt(s); k is
    calibrated on the pair's historical touches (scripts/calibrate_btc_maker.py, scripts/build_pair_model.py).

    Event-driven books (when the model has the exact Hawkes fit):
      ev_none       the same A-S quotes as as_none, but re-priced on every best bid/ask change and every trade
                    instead of once a second. Measures what speed alone is worth.
      ev_hawkes     ev_none plus the Hawkes ensemble's per-second skew/widen and, on every trade, a Hawkes burst
                    detector: log(lambda_buy / lambda_sell) of the exact Hawkes model, using only the kernels whose
                    half-life is at least the latency (faster ones decay before an order could act on them), in the
                    top / bottom BURST_Q of its own recent history (causal) pulls the threatened side. Measures what Hawkes adds once
                    the maker is fast enough to act on millisecond self-excitation.
      Every event-driven decision is made with the information at time t and takes effect at t + latency_ms
      (default 50 ms), so a book can never dodge the trade it is reacting to.

    Fills follow the engine's replay rule, trade by trade: a trade printed through our price fills us in full;
    trades at our price first consume the queue ahead. The queue starts at the displayed size when we join at
    the touch (unknown, so only through-trades fill us, behind it), shrinks with trades at our price, and can
    never exceed the displayed size. Re-quoting at an unchanged price keeps our place.
    """

    TICK = 0.01
    WIDEN_Z, PULL_Z = 0.20, 0.45   # engine thresholds on the standardised (sd 0.2) signal
    WARMUP = 60
    BURST_Q, BURST_WARMUP, BURST_WINDOW, BURST_REFRESH = 0.02, 2000, 20_000, 500

    def __init__(self, names, notional: float = 10_000.0, maker_fee_bps: float = 10.0,
                 inventory_limit: float = 50_000.0, k: float = 0.26, gamma: float = 0.01, tau_s: float = 10.0,
                 tick: float = 0.01, flow_fn=None, ev_signal: str | None = None, latency_ms: float = 50.0):
        self.notional, self.fee_bps, self.limit = notional, maker_fee_bps, inventory_limit
        self.TICK = tick
        self.tick_dp = max(0, -math.floor(math.log10(tick) + 1e-9))
        self.k, self.gamma, self.tau = k, gamma, tau_s
        self.books = {"touch": _MakerBook("touch", None), "as_none": _MakerBook("as", None)}
        for m in names:
            self.books[f"as_{m}"] = _MakerBook("as", m)
        # Event-driven books need the per-trade exact Hawkes intensity (flow_fn(t_ms) -> (lambda_buy, lambda_sell)).
        self.flow_fn, self.latency = flow_fn, latency_ms
        if flow_fn is not None:
            self.books["ev_none"] = _MakerBook("ev", None)
            self.books["ev_hawkes"] = _MakerBook("ev", ev_signal, hawkes_pull=True)
        self.ev_pending: deque = deque()   # (effective_ms, book name, bid px | None, ask px | None)
        self.sigma_usd = float("nan")
        self.burst_hist: deque = deque(maxlen=self.BURST_WINDOW)   # recent log intensity ratios (causal)
        self.burst_lo, self.burst_hi, self.burst_n = -math.inf, math.inf, 0
        self.now_ms: int | None = None
        self.mid = float("nan")
        self.best = (float("nan"), float("nan"), float("nan"), float("nan"))
        self.pending_marks: list = []
        self.quote_log: list = []
        self.fill_log: list = []

    # -- market data ----------------------------------------------------------------------
    def on_book(self, bid: float, bid_qty: float, ask: float, ask_qty: float, t_ms: int | None = None):
        if t_ms is not None:
            self.apply_pending(t_ms)
        moved = (bid, ask) != (self.best[0], self.best[2])
        self.best = (bid, bid_qty, ask, ask_qty)
        self.mid = (bid + ask) / 2
        for b in self.books.values():
            if b.bid.live and abs(b.bid.price - bid) < 1e-9:
                b.bid.queue_ahead = min(b.bid.queue_ahead, bid_qty)
            if b.ask.live and abs(b.ask.price - ask) < 1e-9:
                b.ask.queue_ahead = min(b.ask.queue_ahead, ask_qty)
        if moved and t_ms is not None:
            self.ev_decide(t_ms)

    def on_trade(self, t_ms: int, price: float, qty: float, buyer_is_maker: bool):
        self.apply_pending(t_ms)          # decisions that reached the exchange before this trade
        for name, b in self.books.items():
            order, side = (b.bid, 1) if buyer_is_maker else (b.ask, -1)   # sells hit bids, buys lift asks
            if not order.live or order.remaining <= 0:
                continue
            gap = (order.price - price) if side == 1 else (price - order.price)
            if gap > 1e-9:
                filled = order.remaining
            elif abs(gap) <= 1e-9:
                filled = min(order.remaining, max(0.0, qty - order.queue_ahead))
                order.queue_ahead = max(0.0, order.queue_ahead - qty)
            else:
                continue
            if filled <= 0:
                continue
            usd = filled * order.price
            b.fees += usd * self.fee_bps / 1e4
            b.volume += usd
            b.fills += 1
            b.inv += side * filled
            b.cash -= side * usd
            if not math.isnan(self.mid):
                b.spread_captured += side * (self.mid - order.price) * filled
                self.pending_marks.append([t_ms, name, side, order.price, filled, False, False])
            self.fill_log.append([t_ms, name, "buy" if side == 1 else "sell", order.price, filled, self.mid])
            order.remaining -= filled
            if order.remaining <= 1e-12:
                order.live = False   # a fresh order rejoins at the back next second

    # -- once a second ---------------------------------------------------------------------
    def requote(self, t_ms: int, preds: dict | None, sigma_usd: float):
        self.apply_pending(t_ms)
        self._resolve_marks(t_ms)
        bid, bid_qty, ask, ask_qty = self.best
        if math.isnan(bid) or math.isnan(ask) or math.isnan(bid_qty) or ask <= bid:
            return  # no live book (replay mode): the maker cannot quote
        size = self.notional / self.mid
        self.sigma_usd = sigma_usd
        for name, b in self.books.items():
            if b.mode == "ev":
                b.last_pred = preds[b.signal] if b.signal is not None and preds is not None else None
                b.last_z = self._standardise(b, b.last_pred) if b.last_pred is not None else 0.0
                continue
            if b.mode == "touch":
                bpx, apx = bid, ask
            else:
                if math.isnan(sigma_usd):
                    self._place(b.bid, False, 0, 0, 0)
                    self._place(b.ask, False, 0, 0, 0)
                    continue
                pred = preds[b.signal] if b.signal is not None and preds is not None else None
                z = self._standardise(b, pred) if pred is not None else 0.0
                bpx, apx, widened, pulled = self._as_prices(b, bid, ask, sigma_usd, pred, z)
                b.widens += int(widened)
                b.pulls += int(pulled)
            bpx, apx = self._cap(b, bpx, apx)
            self._place(b.bid, bpx is not None, bpx, bid_qty if bpx is not None and abs(bpx - bid) < 1e-9 else math.inf, size)
            self._place(b.ask, apx is not None, apx, ask_qty if apx is not None and abs(apx - ask) < 1e-9 else math.inf, size)
            b.last_quote = (bpx, apx)
            self.quote_log.append([t_ms, name, bpx, apx, round(b.inv, 8), self.mid])
        self.ev_decide(t_ms)

    def _as_prices(self, b: _MakerBook, bid: float, ask: float, sigma_usd: float, pred: float | None, z: float):
        """Avellaneda-Stoikov bid/ask (None = pulled) as in the engine, with the forecast's skew and toxicity rules."""
        var_tau = sigma_usd ** 2 * self.tau
        r = self.mid - b.inv * self.gamma * var_tau
        half = max((1 / self.gamma) * math.log(1 + self.gamma / self.k) + 0.5 * self.gamma * var_tau, (ask - bid) / 2)
        if pred is not None:
            r += self.mid * pred
        raw_bid, raw_ask = r - half, r + half
        pull_bid = pull_ask = widened = False
        if z > self.WIDEN_Z:          # predicted rise threatens the ask
            raw_ask += 0.5 * half
            widened, pull_ask = True, z > self.PULL_Z
        elif z < -self.WIDEN_Z:
            raw_bid -= 0.5 * half
            widened, pull_bid = True, z < -self.PULL_Z
        bpx = None if pull_bid else min(round(math.floor(raw_bid / self.TICK + 1e-9) * self.TICK, self.tick_dp), bid)
        apx = None if pull_ask else max(round(math.ceil(raw_ask / self.TICK - 1e-9) * self.TICK, self.tick_dp), ask)
        return bpx, apx, widened, pull_bid or pull_ask

    def _cap(self, b: _MakerBook, bpx, apx):
        inv_usd = b.inv * self.mid
        return (None if inv_usd >= self.limit else bpx), (None if inv_usd <= -self.limit else apx)

    # -- event-driven books ------------------------------------------------------------------
    def _log_ratio(self, t_ms: float) -> float:
        lb, ls = self.flow_fn(t_ms)
        return math.log(lb / ls)

    def on_flow(self, t_ms: int):
        """A trade updated the exact Hawkes state: learn the burst distribution (causal), then re-decide."""
        if self.flow_fn is None:
            return
        self.burst_hist.append(self._log_ratio(t_ms + self.latency))
        self.burst_n += 1
        if self.burst_n >= self.BURST_WARMUP and self.burst_n % self.BURST_REFRESH == 0:
            h = sorted(self.burst_hist)
            self.burst_lo = h[int(self.BURST_Q * (len(h) - 1))]
            self.burst_hi = h[int((1 - self.BURST_Q) * (len(h) - 1))]
        self.ev_decide(t_ms)

    def burst(self, t_ms: float) -> int:
        """+1 buy self-excitation burst, -1 sell burst, 0 neither, judged at the time an order would arrive."""
        if self.flow_fn is None or self.burst_n < self.BURST_WARMUP:
            return 0
        x = self._log_ratio(t_ms + self.latency)
        return 1 if x > self.burst_hi else -1 if x < self.burst_lo else 0

    def ev_decide(self, t_ms: int):
        """Price every event-driven book with the information at t_ms; the order reaches the market at t + latency."""
        if self.flow_fn is None:
            return
        self.now_ms = t_ms if self.now_ms is None else max(self.now_ms, t_ms)
        bid, bid_qty, ask, ask_qty = self.best
        if math.isnan(bid) or math.isnan(ask) or math.isnan(bid_qty) or ask <= bid or math.isnan(self.sigma_usd):
            return
        bz = None
        for name, b in self.books.items():
            if b.mode != "ev":
                continue
            bpx, apx, _, _ = self._as_prices(b, bid, ask, self.sigma_usd, b.last_pred, b.last_z)
            if b.hawkes_pull:
                bz = self.burst(t_ms) if bz is None else bz
                if bz > 0 and apx is not None:      # buy self-excitation: the ask is about to be lifted
                    apx = None
                    b.burst_pulls += 1
                elif bz < 0 and bpx is not None:
                    bpx = None
                    b.burst_pulls += 1
            self.ev_pending.append((t_ms + self.latency, name, bpx, apx))

    def apply_pending(self, now_ms: float):
        bid, bid_qty, ask, ask_qty = self.best
        while self.ev_pending and self.ev_pending[0][0] <= now_ms:
            _, name, bpx, apx = self.ev_pending.popleft()
            b = self.books[name]
            if math.isnan(self.mid):
                continue
            bpx, apx = self._cap(b, bpx, apx)
            if bpx is not None and bpx >= ask:   # post-only: a bid that would now cross is rejected
                bpx = None
            if apx is not None and apx <= bid:
                apx = None
            size = self.notional / self.mid
            self._place(b.bid, bpx is not None, bpx, bid_qty if bpx is not None and abs(bpx - bid) < 1e-9 else math.inf, size)
            self._place(b.ask, apx is not None, apx, ask_qty if apx is not None and abs(apx - ask) < 1e-9 else math.inf, size)
            b.last_quote = (bpx, apx)

    def _standardise(self, b: _MakerBook, x: float) -> float:
        out = 0.0
        if b.z_n >= self.WARMUP:
            sd = math.sqrt(b.z_m2 / (b.z_n - 1))
            if sd > 0:
                out = max(-1.0, min(1.0, (x - b.z_mean) / sd * 0.2))
        b.z_n += 1
        d = x - b.z_mean
        b.z_mean += d / b.z_n
        b.z_m2 += d * (x - b.z_mean)
        return out

    @staticmethod
    def _place(order: _Order, want: bool, px, displayed: float, size: float):
        if not want:
            order.live = False
            return
        if order.live and abs(order.price - px) < 1e-9:
            order.remaining = size        # top up; place in the queue kept
            return
        order.live, order.price, order.queue_ahead, order.remaining = True, px, displayed, size

    def _resolve_marks(self, now_ms: int):
        if math.isnan(self.mid):
            return
        keep = []
        for m in self.pending_marks:
            t_ms, name, side, px, qty, done1, done10 = m
            b = self.books[name]
            if not done1 and now_ms >= t_ms + 1000:
                b.markout_1s += side * (self.mid - px) * qty
                b.marked_1s += px * qty
                m[5] = done1 = True
            if not done10 and now_ms >= t_ms + 10_000:
                b.markout_10s += side * (self.mid - px) * qty
                b.marked_10s += px * qty
                m[6] = done10 = True
            if not (done1 and done10):
                keep.append(m)
        self.pending_marks = keep

    def drain_logs(self) -> tuple[list, list]:
        q, f = self.quote_log, self.fill_log
        self.quote_log, self.fill_log = [], []
        return q, f

    def snapshot(self) -> dict:
        out = {}
        for name, b in self.books.items():
            ok = not math.isnan(self.mid)
            gross = b.cash + b.inv * self.mid if ok else 0.0
            bp = lambda x, v: 1e4 * x / v if v > 0 else None  # noqa: E731
            out[name] = {"inventory_usd": b.inv * self.mid if ok else 0.0, "fills": b.fills, "volume": b.volume,
                         "pnl_zero_fee": gross, "fees": b.fees, "pnl_net": gross - b.fees,
                         "spread_bp": bp(b.spread_captured, b.volume), "markout_1s_bp": bp(b.markout_1s, b.marked_1s),
                         "markout_10s_bp": bp(b.markout_10s, b.marked_10s), "pulls": b.pulls + b.burst_pulls, "widens": b.widens,
                         "bid": b.last_quote[0], "ask": b.last_quote[1]}
        return out


@dataclass
class _Parent:
    side: int                 # +1 buy, -1 sell
    arrival_ms: int
    arrival_mid: float
    size: float               # base asset to trade
    deadline_ms: int
    bench_px: float           # price of trading immediately (ask for a buy, bid for a sell)
    remaining: float = 0.0
    cross_sent: bool = False  # event-driven books: a cross is on its way to the market
    price_cost: float = 0.0   # USDT: side * (fill price - arrival mid) * qty
    fees: float = 0.0
    maker_qty: float = 0.0
    order: _Order = field(default_factory=_Order)


@dataclass
class _ExecBook:
    kind: str                 # "timing" | "passive"
    signal: str | None
    fast: bool = False        # event-driven: acts between seconds (with the latency)
    hawkes: bool = False      # react to the per-trade Hawkes burst detector
    repeg: bool = False       # passive: follow the touch when it moves (validation: costs 0.08 bp per order)
    parent: _Parent | None = None
    fast_crosses: int = 0
    orders: int = 0
    price_saved: float = 0.0  # USDT vs the immediate benchmark, before fees
    fee_diff: float = 0.0     # strategy fees - benchmark fees
    notional: float = 0.0
    maker_qty: float = 0.0
    qty: float = 0.0


class Executor:
    """Step 8 live: execute a $notional order every `every_s` seconds within `window_s`, using each model's forecast.

    timing_<m>:  a taker that waits while the forecast says the price is moving its way and crosses as soon as it
                 is not, or at the deadline.
    passive_<m>: rests at the best bid (buy) / ask (sell) as a maker (fills trade by trade, FIFO queue as in
                 PaperMaker) and crosses the rest as soon as the forecast turns against it, or at the deadline.
    passive_none: the no-signal control (rest, cross only at the deadline).
    Event-driven (when the exact Hawkes fit is available; decisions take effect latency_ms later):
    Hawkes books (timing_/passive_ with_hawkes and hawkes_ensemble) are event-driven by default: besides the
    per-second rule, on every trade they cross the rest when the maker's Hawkes burst detector (PaperMaker.burst)
    fires against the order, the cross reaching the market latency_ms later. Baseline and no-signal books stay
    per-second (controls). fast_exec=True adds validation-only comparison books: slow_* (Hawkes ensemble without
    the trigger) and repeg_* (passive books that follow the touch, which validation found costly).
    Benchmark: trade the whole order immediately at arrival, as a taker. Savings = benchmark cost - strategy cost.
    Paper only.
    """

    def __init__(self, names, notional: float = 10_000.0, every_s: int = 60, window_s: int = 10,
                 taker_fee_bps: float = 10.0, maker_fee_bps: float = 10.0, burst_fn=None, latency_ms: float = 50.0,
                 fast_exec: bool = False, trigger_models=("with_hawkes", "hawkes_ensemble")):
        self.notional, self.every, self.window = notional, every_s * 1000, window_s * 1000
        self.taker_fee, self.maker_fee = taker_fee_bps / 1e4, maker_fee_bps / 1e4
        self.books = {"passive_none": _ExecBook("passive", None)}
        # burst_fn(t_ms) -> +1 buy burst / -1 sell burst / 0 (PaperMaker.burst)
        self.burst_fn, self.latency = burst_fn, latency_ms
        # Hawkes books are fast by default: the per-second forecast plus the per-trade burst trigger (no re-pegging,
        # which lost on 8/9 pairs). Baseline and no-signal books stay per-second: they are the controls.
        trig = set(trigger_models) if burst_fn is not None else set()
        for m in names:
            self.books[f"timing_{m}"] = _ExecBook("timing", m, fast=m in trig, hawkes=m in trig)
            self.books[f"passive_{m}"] = _ExecBook("passive", m, fast=m in trig, hawkes=m in trig)
        if burst_fn is not None and fast_exec:   # validation-only comparison books
            self.books["slow_timing_hawkes"] = _ExecBook("timing", "hawkes_ensemble")
            self.books["slow_passive_hawkes"] = _ExecBook("passive", "hawkes_ensemble")
            self.books["repeg_passive_none"] = _ExecBook("passive", None, fast=True, repeg=True)
            self.books["repeg_passive_hawkes"] = _ExecBook("passive", "hawkes_ensemble", fast=True, hawkes=True, repeg=True)
        self.pending: deque = deque()   # (effective_ms, book name, parent, action, price)
        self.order_log: list | None = None   # offline validation: (arrival_ms, book, saving_bp) per completed order
        self.best = (float("nan"), float("nan"), float("nan"), float("nan"))
        self.next_launch: int | None = None
        self.side = 1

    def on_book(self, bid, bid_qty, ask, ask_qty, t_ms: float | None = None):
        if t_ms is not None:
            self.apply_pending(t_ms)
        moved = (bid, ask) != (self.best[0], self.best[2])
        self.best = (bid, bid_qty, ask, ask_qty)
        for name, b in self.books.items():
            p = b.parent
            if p and p.order.live and abs(p.order.price - (bid if p.side == 1 else ask)) < 1e-9:
                p.order.queue_ahead = min(p.order.queue_ahead, bid_qty if p.side == 1 else ask_qty)
            if moved and t_ms is not None and b.repeg and b.kind == "passive" and p and p.order.live and not p.cross_sent:
                self.pending.append((t_ms + self.latency, name, p, "peg", bid if p.side == 1 else ask))

    def on_flow(self, t_ms: float):
        """A trade updated the exact Hawkes state: fast Hawkes books cross when the burst is against them."""
        if self.burst_fn is None:
            return
        burst = None
        for name, b in self.books.items():
            p = b.parent
            if not (b.hawkes and p and p.remaining > 1e-12 and not p.cross_sent):
                continue
            burst = self.burst_fn(t_ms) if burst is None else burst
            if p.side * burst > 0:                     # buy burst against a buy order, sell burst against a sell
                p.cross_sent = True
                b.fast_crosses += 1
                self.pending.append((t_ms + self.latency, name, p, "cross", None))

    def apply_pending(self, now_ms: float):
        bid, bid_qty, ask, ask_qty = self.best
        while self.pending and self.pending[0][0] <= now_ms:
            _, name, p, action, px = self.pending.popleft()
            if self.books[name].parent is not p or p.remaining <= 1e-12:
                continue                          # parent already finished
            if action == "cross":
                self._cross(p)
            elif not p.cross_sent and p.order.live and abs(p.order.price - px) > 1e-9:
                cur = bid if p.side == 1 else ask   # re-peg to the touch as it is when the order arrives
                if abs(cur - px) < 1e-9:
                    p.order = _Order(True, cur, bid_qty if p.side == 1 else ask_qty, p.remaining)

    def on_trade(self, price: float, qty: float, buyer_is_maker: bool, t_ms: float | None = None):
        if t_ms is not None:
            self.apply_pending(t_ms)
        for b in self.books.values():
            p = b.parent
            if not p or not p.order.live or p.remaining <= 0:
                continue
            if (p.side == 1) != buyer_is_maker:      # our bid is hit by sells; our ask is lifted by buys
                continue
            o = p.order
            gap = (o.price - price) if p.side == 1 else (price - o.price)
            if gap > 1e-9:
                filled = p.remaining
            elif abs(gap) <= 1e-9:
                filled = min(p.remaining, max(0.0, qty - o.queue_ahead))
                o.queue_ahead = max(0.0, o.queue_ahead - qty)
            else:
                continue
            if filled > 0:
                self._fill(p, o.price, filled, maker=True)
                if p.remaining <= 1e-12:
                    o.live = False

    def _fill(self, p: _Parent, px: float, qty: float, maker: bool):
        p.price_cost += p.side * (px - p.arrival_mid) * qty
        p.fees += px * qty * (self.maker_fee if maker else self.taker_fee)
        p.maker_qty += qty if maker else 0.0
        p.remaining -= qty

    def _cross(self, p: _Parent):
        bid, _, ask, _ = self.best
        if p.remaining > 1e-12:
            self._fill(p, ask if p.side == 1 else bid, p.remaining, maker=False)
        p.order.live = False

    def _complete(self, b: _ExecBook):
        p = b.parent
        bench_cost = p.side * (p.bench_px - p.arrival_mid) * p.size
        bench_fees = p.bench_px * p.size * self.taker_fee
        b.orders += 1
        b.price_saved += bench_cost - p.price_cost
        b.fee_diff += p.fees - bench_fees
        if self.order_log is not None:
            name = next(k for k, v in self.books.items() if v is b)
            self.order_log.append((p.arrival_ms, name, 1e4 * ((bench_cost - p.price_cost) - (p.fees - bench_fees))
                                   / (p.arrival_mid * p.size), p.maker_qty / p.size))
        b.notional += p.arrival_mid * p.size
        b.maker_qty += p.maker_qty
        b.qty += p.size
        b.parent = None

    def on_second(self, t_ms: int, preds: dict | None):
        self.apply_pending(t_ms)
        bid, bid_qty, ask, ask_qty = self.best
        if math.isnan(bid_qty) or math.isnan(ask_qty) or ask <= bid:
            return  # needs the live book
        mid = (bid + ask) / 2
        for b in self.books.values():          # manage live parents first
            p = b.parent
            if p is None:
                continue
            adverse = b.signal is not None and preds is not None and p.side * preds[b.signal] > 0
            if p.remaining > 1e-12 and (adverse or t_ms >= p.deadline_ms):
                self._cross(p)
            if p.remaining <= 1e-12:
                self._complete(b)
        if self.next_launch is None:
            self.next_launch = t_ms
        if t_ms >= self.next_launch and all(b.parent is None for b in self.books.values()):
            self.next_launch = t_ms + self.every
            size = self.notional / mid
            bench = ask if self.side == 1 else bid
            for b in self.books.values():
                p = _Parent(self.side, t_ms, mid, size, t_ms + self.window, bench, remaining=size)
                b.parent = p
                if b.kind == "timing":
                    if preds is None or self.side * preds[b.signal] > 0:   # no reason to wait: go now
                        self._cross(p)
                        self._complete(b)
                else:
                    px, disp = (bid, bid_qty) if self.side == 1 else (ask, ask_qty)
                    p.order = _Order(True, px, disp, size)
            self.side = -self.side

    def snapshot(self) -> dict:
        out = {}
        for name, b in self.books.items():
            net = b.price_saved - b.fee_diff
            out[name] = {"orders": b.orders, "price_saved": b.price_saved, "fee_diff": b.fee_diff, "saved": net,
                         "saving_bp": 1e4 * net / b.notional if b.notional else None,
                         "maker_share": b.maker_qty / b.qty if b.qty else None, "active": b.parent is not None,
                         "fast_crosses": b.fast_crosses}
        return out


@dataclass
class Engine:
    """Per-second features, predictions and scoring. Mirrors step 6's offline construction."""

    model: dict
    bucket_ms: int | None = None
    n_buy: int = 0
    n_sell: int = 0
    vol: float = 0.0
    signed_vol: float = 0.0
    last_price: float = float("nan")
    prev_close_price: float = float("nan")
    bid: float = float("nan")
    ask: float = float("nan")
    bid_qty: float = float("nan")
    ask_qty: float = float("nan")
    ewm_b: float | None = None
    ewm_s: float | None = None
    lam_b: float | None = None
    lam_s: float | None = None
    returns: deque = field(default_factory=lambda: deque(maxlen=60))
    et_R: list = field(default_factory=list)
    et_t: float = 0.0
    last_event_key: tuple = ()
    pending: dict | None = None
    scores: dict = field(default_factory=lambda: {m: Score() for m in MODEL_NAMES})
    paired: dict = field(default_factory=lambda: {m: [0, 0.0, 0.0] for m in MODEL_NAMES[1:]})
    late_trades: int = 0
    total_trades: int = 0
    t0_ms: int | None = None  # time origin: epoch seconds as floats would lose ~1e-7 s of precision
    paper: PaperTaker = field(default_factory=PaperTaker)
    maker_kwargs: dict = field(default_factory=dict)
    exec_kwargs: dict = field(default_factory=dict)
    models: tuple | None = None   # None: FAMILY_MODELS per family; a tuple runs exactly those models, in every family

    def __post_init__(self):
        hp = self.model["hawkes_params"]
        self.mu = (hp["mu_buy"], hp["mu_sell"])
        self.alpha = (hp["alpha_bb"], hp["alpha_bs"], hp["alpha_ss"], hp["alpha_sb"])
        self.beta = hp["beta"]
        self.a_ewm = 2.0 / (self.model["ewma_span"] + 1.0)
        self.omegas = [math.log(2.0) / h for h in self.model["event_half_lives_s"]]
        self.et_R = [[0.0, 0.0] for _ in self.omegas]
        self.et_names = []
        for h in self.model["event_half_lives_s"]:
            tag = f"{h:g}s".replace(".", "p")
            self.et_names.append((f"et_log_buy_{tag}", f"et_log_sell_{tag}", f"et_ofi_{tag}"))
        self.names = [m for m in MODEL_NAMES if m in self.model["models"] and (self.models is None or m in self.models)]
        fam = (lambda f: [m for m in self.names if m in FAMILY_MODELS[f]]) if self.models is None else (lambda f: self.names)
        self.exec_model = self.model.get("display", {}).get("exec_model", "hawkes_ensemble")
        self.scores = {m: Score() for m in self.names}
        self.paired = {m: [0, 0.0, 0.0] for m in self.names[1:]}
        self.paper.use(fam("taker"))
        mk = self.model.get("maker", {})
        self.maker = PaperMaker(fam("maker"), **{**{k: mk[k] for k in ("k", "gamma") if k in mk},
                                               **({"tick": self.model["tick"]} if "tick" in self.model else {}),
                                               **({"tau_s": mk["tau_s"]} if "tau_s" in mk else {}),
                                               **({"flow_fn": lambda t: self.exact_lambda(
                                                       t, self.maker_kwargs.get("latency_ms", 50.0) / 1000.0),
                                                   "ev_signal": "hawkes_ensemble" if "hawkes_ensemble" in self.names else None}
                                                  if self.model.get("exact_hawkes") else {}),
                                               **self.maker_kwargs})
        self.executor = Executor(fam("execution"), **{**({"burst_fn": self.maker.burst, "latency_ms": self.maker.latency}
                                                   if self.model.get("exact_hawkes") else {}), **self.exec_kwargs})
        # Exact event-time Hawkes (step 7): lambda_i(t) = mu_i + sum_k sum_j alpha[k][i][j] R_kj(t)
        xh = self.model.get("exact_hawkes")
        self.x_mu = xh["mu"] if xh else None
        self.x_alpha = xh["alpha"] if xh else None
        self.x_omega = xh["omega"] if xh else []
        self.x_R = [[0.0, 0.0] for _ in self.x_omega]

    def executor_models(self) -> list[str]:
        return [b.signal for b in self.executor.books.values() if b.signal]

    def exact_lambda(self, t_ms: float, min_half_life_s: float = 0.0) -> tuple[float, float]:
        """Exact Hawkes intensities (buy, sell) at exchange time t_ms, from the state after the last trade.
        Kernels with a half-life below min_half_life_s are left out (they decay before a decision can act)."""
        dt = max((t_ms - self.t0_ms) / 1000.0 - self.et_t, 0.0) if self.t0_ms is not None else 0.0
        lam = []
        for i in (0, 1):
            v = self.x_mu[i]
            for k, w in enumerate(self.x_omega):
                if math.log(2.0) / w < min_half_life_s:
                    continue
                d = math.exp(-w * dt)
                v += (self.x_alpha[k][i][0] * self.x_R[k][0] + self.x_alpha[k][i][1] * self.x_R[k][1]) * d
            lam.append(max(v, 1e-8))
        return lam[0], lam[1]

    # -- inputs ---------------------------------------------------------------------------
    def on_trade(self, t_ms: int, price: float, qty: float, buyer_is_maker: bool):
        if self.bucket_ms is None:
            self.bucket_ms = t_ms - t_ms % 1000
            self.t0_ms = self.bucket_ms
        if t_ms < self.bucket_ms:
            self.late_trades += 1  # arrived after its second closed: counted in the current one
        side = 1 if buyer_is_maker else 0  # 0 = buy-initiated, 1 = sell-initiated
        if side == 0:
            self.n_buy += 1
            self.signed_vol += qty
        else:
            self.n_sell += 1
            self.signed_vol -= qty
        self.vol += qty
        self.last_price = price
        self.total_trades += 1
        key = (t_ms, side)  # one sweeping order at one timestamp counts once, as offline
        if key != self.last_event_key:
            # Out-of-order trades (a few ms early) are treated as simultaneous; with a ~0.1 ms kernel
            # (ETHUSDT's exact fit) a negative gap would otherwise overflow exp().
            t = max((t_ms - self.t0_ms) / 1000.0, self.et_t)
            for R, omegas in ((self.et_R, self.omegas), (self.x_R, self.x_omega)):
                for k, w in enumerate(omegas):
                    decay = math.exp(-w * (t - self.et_t))
                    R[k][0] *= decay
                    R[k][1] *= decay
                    R[k][side] += w
            self.et_t = t
            self.last_event_key = key
        self.maker.on_trade(t_ms, price, qty, buyer_is_maker)
        self.maker.on_flow(t_ms)
        self.executor.on_trade(price, qty, buyer_is_maker, t_ms)
        self.executor.on_flow(t_ms)

    def on_book(self, bid: float, bid_qty: float, ask: float, ask_qty: float, t_ms: float | None = None):
        """t_ms: estimated exchange time of the update (bookTicker carries none); drives the event-driven makers."""
        self.bid, self.bid_qty, self.ask, self.ask_qty = bid, bid_qty, ask, ask_qty
        if not math.isnan(bid_qty):
            self.maker.on_book(bid, bid_qty, ask, ask_qty, t_ms)
            self.executor.on_book(bid, bid_qty, ask, ask_qty, t_ms)

    # -- bucket close ---------------------------------------------------------------------
    def close_bucket(self) -> dict:
        """Close the current second: score the forecast made for it, then forecast the next."""
        end_s = (self.bucket_ms + 1000 - self.t0_ms) / 1000.0
        price = self.last_price
        if not math.isnan(self.prev_close_price) and not math.isnan(price):
            ret = math.log(price) - math.log(self.prev_close_price)
            self.returns.append(ret)
            self.paper.mark(ret)
            if self.pending is not None:
                # Same rule as step 6: hit rates count only moves larger than a bid-ask bounce.
                self._score(ret, self.model.get("hit_min_abs_move", 0.0) / self.prev_close_price)
        else:
            ret = float("nan")

        nb, ns = float(self.n_buy), float(self.n_sell)
        if self.ewm_b is None:
            self.ewm_b, self.ewm_s = nb, ns
            self.lam_b, self.lam_s = self.mu
        else:
            self.ewm_b = (1 - self.a_ewm) * self.ewm_b + self.a_ewm * nb
            self.ewm_s = (1 - self.a_ewm) * self.ewm_s + self.a_ewm * ns
        a_bb, a_bs, a_ss, a_sb = self.alpha
        lam_b = max(self.mu[0] + self.beta * self.lam_b + a_bb * nb + a_bs * ns, 1e-8)
        lam_s = max(self.mu[1] + self.beta * self.lam_s + a_ss * ns + a_sb * nb, 1e-8)
        self.lam_b, self.lam_s = lam_b, lam_s

        den = self.ewm_b + self.ewm_s
        feats = {
            "order_flow_imbalance": (self.ewm_b - self.ewm_s) / den if den > 0 else 0.0,
            "lambda_buy_ewm": self.ewm_b, "lambda_sell_ewm": self.ewm_s,
            "trade_imbalance": self.signed_vol / self.vol if self.vol > 0 else 0.0,
            "lambda_buy_hawkes": lam_b, "lambda_sell_hawkes": lam_s,
            "hawkes_ofi": (lam_b - lam_s) / (lam_b + lam_s),
            "rv_60": _std(self.returns) if len(self.returns) == 60 else float("nan"),
        }
        for k, w in enumerate(self.omegas):
            decay = math.exp(-w * max(end_s - self.et_t, 0.0))
            b, s = self.et_R[k][0] * decay, self.et_R[k][1] * decay
            nb_name, ns_name, no_name = self.et_names[k]
            feats[nb_name], feats[ns_name] = math.log1p(b), math.log1p(s)
            feats[no_name] = (b - s) / (b + s) if b + s > 1e-12 else 0.0
        if self.x_mu is not None:
            Rx = [[r * math.exp(-w * max(end_s - self.et_t, 0.0)) for r in self.x_R[k]] for k, w in enumerate(self.x_omega)]
            lam_x = [max(self.x_mu[i] + sum(self.x_alpha[k][i][j] * Rx[k][j] for k in range(len(Rx)) for j in (0, 1)),
                         1e-8) for i in (0, 1)]
            feats["hx_log_buy"], feats["hx_log_sell"] = math.log(lam_x[0]), math.log(lam_x[1])
            feats["hx_ofi"] = (lam_x[0] - lam_x[1]) / (lam_x[0] + lam_x[1])

        preds = None
        if not math.isnan(feats["rv_60"]) and not math.isnan(price):
            preds = {name: self._predict(name, feats) for name in self.names}
        self.pending = preds
        if preds is not None:
            self.paper.rebalance(preds)
        self.maker.requote(self.bucket_ms + 1000, preds,
                           feats["rv_60"] * price if not math.isnan(price) else float("nan"))
        mk = self.maker.snapshot()
        self.executor.on_second(self.bucket_ms + 1000, preds)
        ex = self.executor.snapshot()

        row = {
            "t": self.bucket_ms, "last": price, "bid": self.bid, "ask": self.ask,
            "bid_qty": self.bid_qty, "ask_qty": self.ask_qty, "n_buy": self.n_buy, "n_sell": self.n_sell,
            "volume": self.vol, "signed_volume": self.signed_vol, "ret": ret,
            "lam_b": lam_b, "lam_s": lam_s, "ewm_b": self.ewm_b, "ewm_s": self.ewm_s,
            "hawkes_ofi": feats["hawkes_ofi"], "ewma_ofi": feats["order_flow_imbalance"],
            "et_ofi_fast": feats[self.et_names[0][2]], "hx_ofi": feats.get("hx_ofi"), "preds": preds,
            "pnl_net": {m: v["net"] for m, v in self.paper.snapshot().items()},
            "pnl_gross": {m: v["gross"] for m, v in self.paper.snapshot().items()},
            "mm_zero_fee": {m: v["pnl_zero_fee"] for m, v in mk.items()},
            "mm_net": {m: v["pnl_net"] for m, v in mk.items()},
            "mm_inv": {m: v["inventory_usd"] for m, v in mk.items()},
            "ex_saved": {m: v["saved"] for m, v in ex.items()},
            "hit": {m: (sc.hits / sc.n_nonzero if sc.n_nonzero else None) for m, sc in self.scores.items()},
        }
        self.prev_close_price = price
        self.bucket_ms += 1000
        self.n_buy = self.n_sell = 0
        self.vol = self.signed_vol = 0.0
        return row

    def _predict(self, name: str, feats: dict) -> float:
        m = self.model["models"][name]
        z = sum(b * (feats[f] - mu) / sd for f, mu, sd, b in zip(m["features"], m["mu"], m["sigma"], m["beta"]))
        return m["intercept"] + z

    def _score(self, ret: float, hit_min_abs: float = 0.0):
        se = {}
        for name, pred in self.pending.items():
            s = self.scores[name]
            s.n += 1
            s.sse += (ret - pred) ** 2
            s.sst += ret * ret
            se[name] = (ret - pred) ** 2
            if abs(ret) > hit_min_abs and pred != 0:
                s.n_nonzero += 1
                s.hits += int((ret > 0) == (pred > 0))
        for alt in self.names[1:]:
            d = se["baseline_ewma"] - se[alt]
            acc = self.paired[alt]
            acc[0] += 1
            acc[1] += d
            acc[2] += d * d

    def scoreboard(self) -> dict:
        out = {}
        for name, s in self.scores.items():
            lo, hi = wilson(s.hits, s.n_nonzero)
            out[name] = {"n": s.n, "n_nonzero": s.n_nonzero, "hit": s.hits / s.n_nonzero if s.n_nonzero else None,
                         "hit_lo": lo, "hit_hi": hi, "r2": 1 - s.sse / s.sst if s.sst > 0 else None}
        for alt, (n, sd, sd2) in self.paired.items():
            if n > 1:
                mean = sd / n
                var = max(sd2 / n - mean * mean, 0.0)
                out[alt]["mse_gain"] = mean
                out[alt]["mse_gain_se_naive"] = math.sqrt(var / n)
        out["paper"] = {"notional": self.paper.notional, "fee_bps": self.paper.fee_bps, **self.paper.snapshot()}
        out["execution"] = {"notional": self.executor.notional, "every_s": self.executor.every // 1000,
                            "featured": self.exec_model if self.exec_model in self.executor_models() else None,
                             "window_s": self.executor.window // 1000, **self.executor.snapshot()}
        out["maker"] = {"notional": self.maker.notional, "fee_bps": self.maker.fee_bps, "inventory_limit": self.maker.limit,
                        "live_book": not math.isnan(self.maker.best[1]),
                        "ev_latency_ms": self.maker.latency, **self.maker.snapshot()}
        return out


def _std(xs) -> float:
    n = len(xs)
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


class Recorder:
    """Appends raw aggTrades (Binance historical CSV format) and per-second buckets under data/live/<date>/."""

    def __init__(self, root: Path | None):
        self.root = root
        self.day = None
        self.trades_f = self.buckets_f = None
        self.quotes_f = self.fills_f = None

    def _open(self, day: str):
        if self.root is None or day == self.day:
            return
        for f in (self.trades_f, self.buckets_f, self.quotes_f, self.fills_f):
            if f:
                f.close()
        d = self.root / day
        d.mkdir(parents=True, exist_ok=True)
        self.trades_f = open(d / "aggTrades.csv", "a", newline="")
        new = not (d / "buckets_1s.csv").exists()
        self.buckets_f = open(d / "buckets_1s.csv", "a", newline="")
        if new:
            csv.writer(self.buckets_f).writerow(["ts_utc", "last", "bid", "ask", "bid_qty", "ask_qty",
                                                 "buy_events", "sell_events", "volume", "signed_volume"])
        for attr, fname, header in (
                ("quotes_f", "maker_quotes.csv", ["ts_ms", "book", "bid", "ask", "inventory_btc", "mid"]),
                ("fills_f", "maker_fills.csv", ["ts_ms", "book", "side", "price", "qty_btc", "mid_at_fill"])):
            new_f = not (d / fname).exists()
            setattr(self, attr, open(d / fname, "a", newline=""))
            if new_f:
                csv.writer(getattr(self, attr)).writerow(header)
        self.day = day

    def trade(self, d: dict):
        if self.root is None:
            return
        self._open(datetime.fromtimestamp(d["T"] / 1000, timezone.utc).strftime("%Y-%m-%d"))
        csv.writer(self.trades_f).writerow([d["a"], d["p"], d["q"], d["f"], d["l"], d["T"],
                                            "True" if d["m"] else "False", "True" if d.get("M", True) else "False"])

    def bucket(self, row: dict):
        if self.root is None:
            return
        ts = datetime.fromtimestamp(row["t"] / 1000, timezone.utc)
        self._open(ts.strftime("%Y-%m-%d"))
        csv.writer(self.buckets_f).writerow([ts.isoformat(), row["last"], row["bid"], row["ask"], row["bid_qty"],
                                             row["ask_qty"], row["n_buy"], row["n_sell"], row["volume"],
                                             row["signed_volume"]])

    def maker(self, quotes: list, fills: list):
        if self.root is None or self.quotes_f is None:
            return
        csv.writer(self.quotes_f).writerows(quotes)
        csv.writer(self.fills_f).writerows(fills)

    def flush(self):
        for f in (self.trades_f, self.buckets_f, self.quotes_f, self.fills_f):
            if f:
                f.flush()


class Monitor:
    def __init__(self, args):
        self.args = args
        self.model = json.loads(Path(args.model).read_text())
        self.engine = Engine(self.model, paper=PaperTaker(args.paper_notional, args.taker_fee_bps),
                             maker_kwargs={"notional": args.paper_notional, "maker_fee_bps": args.maker_fee_bps,
                                           "inventory_limit": args.inventory_limit, "latency_ms": args.ev_latency_ms},
                             exec_kwargs={"notional": args.paper_notional, "taker_fee_bps": args.taker_fee_bps,
                                          "maker_fee_bps": args.maker_fee_bps})
        self.history: deque = deque(maxlen=HISTORY)
        self.clients: set = set()
        self.recorder = Recorder(None if args.no_record else ROOT / args.record)
        self.status = {"mode": "replay" if args.replay else "live", "connected": False, "reconnects": 0,
                       "symbol": args.symbol.upper(), "started": time.time(), "last_msg": None, "error": None}
        self.offset_ms: float | None = None  # local clock minus exchange clock

    # -- broadcasting ---------------------------------------------------------------------
    def snapshot(self) -> dict:
        return {"type": "init", "history": list(self.history), "status": self.status,
                "scores": self.engine.scoreboard(), "model": {"trained_on": self.model["trained_on"],
                                                               "symbol": self.model["symbol"],
                                                               "hawkes": self.model["hawkes_params"]}}

    async def publish(self, row: dict):
        self.history.append(row)
        self.recorder.bucket(row)
        self.recorder.maker(*self.engine.maker.drain_logs())
        msg = _dumps({"type": "tick", "row": row, "status": self.status, "scores": self.engine.scoreboard(),
                      "late_trades": self.engine.late_trades, "total_trades": self.engine.total_trades})
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except websockets.ConnectionClosed:
                self.clients.discard(ws)

    # -- live feed ------------------------------------------------------------------------
    async def feed(self):
        url = STREAM_URL.format(s=self.args.symbol.lower())
        backoff = 1
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, max_size=2 ** 20) as ws:
                    self.status.update(connected=True, error=None)
                    backoff = 1
                    async for raw in ws:
                        msg = json.loads(raw)
                        d = msg.get("data", {})
                        now = time.time() * 1000
                        self.status["last_msg"] = now
                        if msg.get("stream", "").endswith("@aggTrade"):
                            sample = now - d["E"]
                            self.offset_ms = sample if self.offset_ms is None else min(sample, self.offset_ms + 1)
                            self.engine.on_trade(d["T"], float(d["p"]), float(d["q"]), bool(d["m"]))
                            self.recorder.trade(d)
                        elif msg.get("stream", "").endswith("@bookTicker"):
                            t_est = now - self.offset_ms if self.offset_ms is not None else None
                            self.engine.on_book(float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"]), t_est)
            except Exception as e:  # noqa: BLE001 -- network errors: report on the dashboard and reconnect
                self.status.update(connected=False, error=f"{type(e).__name__}: {e}"[:200])
                self.status["reconnects"] += 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def clock(self):
        """Close each second once the exchange clock is GRACE_MS past its end."""
        while True:
            await asyncio.sleep(0.1)
            eng = self.engine
            if eng.bucket_ms is None or self.offset_ms is None:
                continue
            exch_now = time.time() * 1000 - self.offset_ms
            while exch_now >= eng.bucket_ms + 1000 + GRACE_MS:
                await self.publish(eng.close_bucket())
            self.recorder.flush()

    # -- replay ---------------------------------------------------------------------------
    async def replay(self):
        self.status.update(connected=True)
        eng = self.engine
        with open(self.args.replay, newline="") as f:
            for row in csv.reader(f):
                t_ms = int(row[5])
                if eng.bucket_ms is not None:
                    while t_ms >= eng.bucket_ms + 1000:
                        await self.publish(eng.close_bucket())
                        await asyncio.sleep(1.0 / self.args.speed)
                price = float(row[1])
                eng.on_trade(t_ms, price, float(row[2]), row[6] == "True")
                eng.on_book(price, float("nan"), price, float("nan"))
        self.status.update(connected=False, error="replay finished")

    # -- HTTP + WebSocket server ----------------------------------------------------------
    async def handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(_dumps(self.snapshot()))
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)

    def process_request(self, connection, request):
        if request.path == "/ws":
            return None
        if request.path in ("/", "/index.html"):
            resp = connection.respond(HTTPStatus.OK, DASHBOARD.read_text())
            resp.headers["Content-Type"] = "text/html; charset=utf-8"
            return resp
        if request.path == "/health":
            h = self.history
            resp = connection.respond(HTTPStatus.OK, _dumps({"status": self.status, "seconds": len(h),
                                                             "last": h[-1]["last"] if h else None,
                                                             "first": h[0]["last"] if h else None,
                                                             "trained_on": self.model["trained_on"],
                                                             "scores": self.engine.scoreboard()}))
            resp.headers["Content-Type"] = "application/json"
            return resp
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def run(self):
        async with serve(self.handler, self.args.host, self.args.port, process_request=self.process_request):
            print(f"Dashboard: http://{self.args.host}:{self.args.port}/   (mode: {self.status['mode']}, "
                  f"model trained on {self.model['trained_on']})", flush=True)
            if self.args.replay:
                await self.replay()
                await asyncio.Future()
            else:
                await asyncio.gather(self.feed(), self.clock())


def _dumps(obj) -> str:
    def clean(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        return x
    return json.dumps(clean(obj))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="btcusdt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--record", default="data/live", help="directory for recorded data (under the repo)")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--replay", help="replay a Binance aggTrades CSV instead of the live feed")
    ap.add_argument("--speed", type=float, default=10.0, help="replay speed (simulated seconds per second)")
    ap.add_argument("--model", default=str(MODEL_PATH), help="live model JSON (default: the BTCUSDT step 6-7 models)")
    ap.add_argument("--paper-notional", type=float, default=10_000.0, help="paper position size in USDT")
    ap.add_argument("--taker-fee-bps", type=float, default=10.0, help="taker fee per side (Binance standard: 10)")
    ap.add_argument("--maker-fee-bps", type=float, default=10.0, help="maker fee (Binance standard: 10)")
    ap.add_argument("--inventory-limit", type=float, default=50_000.0, help="paper market maker inventory cap, USDT")
    ap.add_argument("--ev-latency-ms", type=float, default=50.0,
                    help="event-driven makers: delay between seeing an event and the order reaching the market")
    args = ap.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"{args.model} missing: run `make validate-binance` (BTCUSDT) or scripts/build_pair_model.py.")
    try:
        asyncio.run(Monitor(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
