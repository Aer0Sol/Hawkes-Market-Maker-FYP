"""Run several live monitors side by side and serve one page that toggles between them.

Each pair in config/live_pairs.json is its own scripts/live_monitor.py process (own Binance stream, port,
paper books and recording), so pairs run in parallel and one crashing does not stop the others. The hub
only reads their /health endpoints; it never places orders and has no keys (docs/plan.md section 0.2).

  uv run --group live python scripts/live_hub.py start     # start every managed pair not already running
  uv run --group live python scripts/live_hub.py serve     # hub page on http://127.0.0.1:8800/
  uv run --group live python scripts/live_hub.py status
  uv run --group live python scripts/live_hub.py stop ETHUSDT SOLUSDT   # stop named managed pairs

Pairs with managed=false (the original BTCUSDT monitor) are shown but never started or stopped here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "live_pairs.json"
PAGE = Path(__file__).resolve().parent / "live_hub.html"
STRATEGY_LABEL = {"touch": "Maker at the touch", "as_none": "A–S maker", "passive_none": "Passive (no signal)",
                  "ev_none": "Fast A–S maker", "ev_hawkes": "Fast A–S maker · Hawkes"}


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def health(port: int, timeout: float = 1.5) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:  # noqa: S310 -- localhost
            return json.load(r)
    except Exception:  # noqa: BLE001 -- down or starting
        return None


def validation(symbol: str) -> dict | None:
    p = ROOT / "output" / "pairs" / symbol / "validation.json"
    return json.loads(p.read_text()) if p.exists() else None


def pids(symbol: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", f"live_monitor.py --symbol {symbol.lower()} "], capture_output=True, text=True)
    return [int(x) for x in out.stdout.split()]


# ------------------------------------------------------------------ summary for the page
MODEL_SHORT = {"baseline_ewma": "baseline", "with_hawkes": "Hawkes (1 s)", "event_time": "+event-time",
               "hawkes_ensemble": "Hawkes ensemble", "none": "no signal"}


def _label(name: str) -> str:
    if name in STRATEGY_LABEL:
        return STRATEGY_LABEL[name]
    for prefix, kind in (("as_", "A–S maker"), ("timing_", "Timing"), ("passive_", "Passive")):
        if name.startswith(prefix):
            m = name[len(prefix):]
            return f"{kind} · {MODEL_SHORT.get(m, m.replace('_', ' '))}"
    return name.replace("_", " ")


def summarise(pair: dict, h: dict | None) -> dict:
    out = {"symbol": pair["symbol"], "port": pair["port"], "managed": pair.get("managed", True), "up": h is not None,
           "validation": validation(pair["symbol"])}
    if h is None:
        return out
    st, sc = h["status"], h["scores"]
    maker = {k: v for k, v in sc.get("maker", {}).items() if isinstance(v, dict)}
    taker = {k: v for k, v in sc.get("paper", {}).items() if isinstance(v, dict)}
    execu = {k: v for k, v in sc.get("execution", {}).items() if isinstance(v, dict)}
    last = h.get("last")
    if last is None:   # monitors started before /health carried the price: use the touch quotes
        t = maker.get("touch", {})
        if t.get("bid") and t.get("ask"):
            last = (t["bid"] + t["ask"]) / 2
    board = ([{"name": k, "label": _label(k), "kind": "maker", "pnl": v["pnl_net"], "gross": v["pnl_zero_fee"]} for k, v in maker.items()]
             + [{"name": k, "label": f"Taker · {MODEL_SHORT.get(k, k)}", "kind": "taker", "pnl": v["net"], "gross": v["gross"]} for k, v in taker.items()]
             + [{"name": k, "label": _label(k), "kind": "exec", "pnl": v["saved"], "gross": v["price_saved"],
                 "bp": v["saving_bp"], "orders": v["orders"]} for k, v in execu.items()])
    board.sort(key=lambda r: r["pnl"], reverse=True)
    models = [k for k, v in sc.items() if isinstance(v, dict) and "hit" in v]
    out.update({
        "connected": st.get("connected"), "error": st.get("error"), "uptime_s": time.time() - st["started"],
        "seconds": h.get("seconds"), "last": last, "first": h.get("first"), "trained_on": h.get("trained_on"),
        "board": board, "models": models,
        "hit": {m: sc[m].get("hit") for m in models}, "mse_gain": {m: sc[m].get("mse_gain") for m in models},
        "maker_edge_0fee_bp": {k: 1e4 * v["pnl_zero_fee"] / v["volume"] for k, v in maker.items() if v.get("volume")},
        "featured_exec": sc.get("execution", {}).get("featured"),
        "maker_fills": sum(v.get("fills", 0) for v in maker.values()),
        "exec_orders": max((v.get("orders", 0) for v in execu.values()), default=0),
    })
    return out


def snapshot() -> dict:
    cfg = load_config()
    with ThreadPoolExecutor(max_workers=len(cfg["pairs"])) as ex:
        hs = list(ex.map(lambda p: health(p["port"]), cfg["pairs"]))
    return {"t": time.time(), "pairs": [summarise(p, h) for p, h in zip(cfg["pairs"], hs)]}


# ------------------------------------------------------------------ process control
def start(symbols: list[str]):
    cfg = load_config()
    for p in cfg["pairs"]:
        sym = p["symbol"]
        if symbols and sym not in symbols:
            continue
        if not p.get("managed", True):
            print(f"{sym}: not managed by the hub, left alone")
            continue
        if health(p["port"]) or pids(sym):
            print(f"{sym}: already running on :{p['port']}")
            continue
        if not (ROOT / p["model"]).exists():
            print(f"{sym}: no model at {p['model']} (run scripts/train_pair.py), skipped")
            continue
        log_dir = ROOT / p["record"]
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "scripts/live_monitor.py", "--symbol", sym.lower(), "--port", str(p["port"]),
               "--model", p["model"], "--record", p["record"]]
        with open(log_dir / "monitor.log", "a") as log:
            subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        print(f"{sym}: started on :{p['port']}")


def stop(symbols: list[str]):
    cfg = load_config()
    for p in cfg["pairs"]:
        sym = p["symbol"]
        if symbols and sym not in symbols:
            continue
        if not p.get("managed", True):
            print(f"{sym}: not managed by the hub, left alone")
            continue
        found = pids(sym)
        for pid in found:
            os.kill(pid, signal.SIGTERM)
        print(f"{sym}: {'stopped ' + str(found) if found else 'not running'}")


def status():
    for s in snapshot()["pairs"]:
        if not s["up"]:
            print(f"{s['symbol']:9s} :{s['port']}  DOWN")
            continue
        lead = f"{s['board'][0]['label']} {s['board'][0]['pnl']:+.2f}" if s["board"] else "-"
        state = "live" if s["connected"] else "reconnecting"
        print(f"{s['symbol']:9s} :{s['port']}  {state:12s} up {s['uptime_s'] / 3600:5.1f} h  last {s['last']}  "
              f"models {len(s['models'])}  leader {lead}")


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 -- http.server API
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/pairs":
            self._send(HTTPStatus.OK, _dumps(snapshot()).encode(), "application/json")
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found\n", "text/plain")

    def log_message(self, *a):
        pass


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


def serve(host: str, port: int):
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Hub: http://{host}:{port}/", flush=True)
    srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["serve", "start", "stop", "status"], nargs="?", default="serve")
    ap.add_argument("symbols", nargs="*", help="limit start/stop to these symbols (e.g. ETHUSDT)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int)
    args = ap.parse_args()
    syms = [s.upper() for s in args.symbols]
    if args.command == "serve":
        serve(args.host, args.port or load_config()["hub_port"])
    elif args.command == "start":
        start(syms)
    elif args.command == "stop":
        stop(syms)
    else:
        status()


if __name__ == "__main__":
    main()
