"""Live BTCUSDT order-flow monitor: the project's Hawkes model on the real-time Binance feed.

One process does three things:
  1. collects Binance's public aggTrade and bookTicker streams (no key, no orders),
  2. rebuilds, every second, the exact features the step 6 models were trained on (EWMA and
     one-step-ahead Hawkes intensities, event-time decayed counts, realized vol, trade imbalance),
     predicts the next second's return with the baseline, +Hawkes and +event-time models
     (output/step6_binance_validation/live_model.json) and scores each prediction when the
     second it forecasts closes: a live, genuinely out-of-sample version of step 6;
  3. serves a dashboard at http://localhost:<port>/ and streams updates over a WebSocket.

It also records what it sees under data/live/<UTC date>/: aggTrades.csv in Binance's historical
format (so step 6 can read a live-collected day) and buckets_1s.csv with the best bid/ask each
second, which the historical files lack.

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
MODEL_NAMES = ("baseline_ewma", "with_hawkes", "event_time")


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
    """Option 1: trade each model's forecast directly. Hold +notional or -notional according to the sign of
    the 1 s forecast, trade only when the sign flips, pay the taker fee on every dollar traded. Paper only."""

    notional: float = 10_000.0
    fee_bps: float = 10.0
    pos: dict = field(default_factory=lambda: {m: 0 for m in MODEL_NAMES})
    gross: dict = field(default_factory=lambda: {m: 0.0 for m in MODEL_NAMES})
    fees: dict = field(default_factory=lambda: {m: 0.0 for m in MODEL_NAMES})
    traded: dict = field(default_factory=lambda: {m: 0.0 for m in MODEL_NAMES})

    def mark(self, ret: float):
        """P&L of the positions held over the second that just closed (log return -> simple return)."""
        move = math.expm1(ret)
        for m, q in self.pos.items():
            self.gross[m] += q * self.notional * move

    def rebalance(self, preds: dict):
        for m, pred in preds.items():
            new = (pred > 0) - (pred < 0)
            dq = abs(new - self.pos[m])
            if dq:
                self.traded[m] += dq * self.notional
                self.fees[m] += dq * self.notional * self.fee_bps / 1e4
                self.pos[m] = new

    def snapshot(self) -> dict:
        return {m: {"pos": self.pos[m], "gross": self.gross[m], "fees": self.fees[m],
                    "net": self.gross[m] - self.fees[m], "traded": self.traded[m]} for m in MODEL_NAMES}


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
            t = (t_ms - self.t0_ms) / 1000.0
            for k, w in enumerate(self.omegas):
                decay = math.exp(-w * (t - self.et_t))
                self.et_R[k][0] *= decay
                self.et_R[k][1] *= decay
                self.et_R[k][side] += w
            self.et_t = t
            self.last_event_key = key

    def on_book(self, bid: float, bid_qty: float, ask: float, ask_qty: float):
        self.bid, self.bid_qty, self.ask, self.ask_qty = bid, bid_qty, ask, ask_qty

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
            decay = math.exp(-w * (end_s - self.et_t))
            b, s = self.et_R[k][0] * decay, self.et_R[k][1] * decay
            nb_name, ns_name, no_name = self.et_names[k]
            feats[nb_name], feats[ns_name] = math.log1p(b), math.log1p(s)
            feats[no_name] = (b - s) / (b + s) if b + s > 1e-12 else 0.0

        preds = None
        if not math.isnan(feats["rv_60"]) and not math.isnan(price):
            preds = {name: self._predict(name, feats) for name in MODEL_NAMES}
        self.pending = preds
        if preds is not None:
            self.paper.rebalance(preds)

        row = {
            "t": self.bucket_ms, "last": price, "bid": self.bid, "ask": self.ask,
            "bid_qty": self.bid_qty, "ask_qty": self.ask_qty, "n_buy": self.n_buy, "n_sell": self.n_sell,
            "volume": self.vol, "signed_volume": self.signed_vol, "ret": ret,
            "lam_b": lam_b, "lam_s": lam_s, "ewm_b": self.ewm_b, "ewm_s": self.ewm_s,
            "hawkes_ofi": feats["hawkes_ofi"], "ewma_ofi": feats["order_flow_imbalance"],
            "et_ofi_fast": feats[self.et_names[0][2]], "preds": preds,
            "pnl_net": {m: v["net"] for m, v in self.paper.snapshot().items()},
            "pnl_gross": {m: v["gross"] for m, v in self.paper.snapshot().items()},
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
        for alt in MODEL_NAMES[1:]:
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

    def _open(self, day: str):
        if self.root is None or day == self.day:
            return
        for f in (self.trades_f, self.buckets_f):
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

    def flush(self):
        for f in (self.trades_f, self.buckets_f):
            if f:
                f.flush()


class Monitor:
    def __init__(self, args):
        self.args = args
        self.model = json.loads(MODEL_PATH.read_text())
        self.engine = Engine(self.model, paper=PaperTaker(args.paper_notional, args.taker_fee_bps))
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
                            self.engine.on_book(float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"]))
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
            resp = connection.respond(HTTPStatus.OK, _dumps({"status": self.status, "seconds": len(self.history),
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
    ap.add_argument("--paper-notional", type=float, default=10_000.0, help="paper position size in USDT")
    ap.add_argument("--taker-fee-bps", type=float, default=10.0, help="taker fee per side (Binance standard: 10)")
    args = ap.parse_args()
    if not MODEL_PATH.exists():
        raise SystemExit(f"{MODEL_PATH} missing: run `make validate-binance` first.")
    try:
        asyncio.run(Monitor(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
