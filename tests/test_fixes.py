"""Tests for the gap fixes: replay fills, causal OFI standardisation, event-time features, live/offline parity."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import step5_alpha_signal as s5

TS = pd.date_range("2024-06-03 13:30", periods=6, freq="1s", tz="UTC").astype(str)


def flat_book(path: Path, n: int = 6, bid_sz: float = 100.0) -> Path:
    """Flat 100.00 / 100.01 book; quotes clamp to the touch with a 1-tick floor and large k."""
    pd.DataFrame({
        "ts": TS[:n], "bid_px_1": 100.00, "ask_px_1": 100.01, "bid_sz_1": bid_sz, "ask_sz_1": bid_sz,
        "depth_bid_10": 1000.0, "depth_ask_10": 1000.0, "midprice": 100.005, "spread": 0.01, "microprice": 100.005,
        "rv_60": 1e-5, "imbalance_l1": 0.0, "lambda_buy_ewm": 1.0, "lambda_sell_ewm": 1.0,
        "order_flow_imbalance": 0.0, "lambda_buy_hawkes": 1.0, "lambda_sell_hawkes": 1.0, "hawkes_ofi": 0.0,
    }).to_csv(path, index=False)
    return path


def run_replay(engine_bin, tmp_path, trades: list[tuple[int, float, float, int]], bid_sz: float = 100.0,
               extra: tuple[str, ...] = ()) -> pd.DataFrame:
    f = flat_book(tmp_path / "f.csv", bid_sz=bid_sz)
    pd.DataFrame([{"ts": TS[i], "price": p, "size": s, "sign": sg} for i, p, s, sg in trades],
                 columns=["ts", "price", "size", "sign"]).to_csv(tmp_path / "t.csv", index=False)
    out = tmp_path / "o"
    subprocess.run([str(engine_bin), "--features", str(f), "--outdir", str(out), "--fill-model", "replay",
                    "--trades", str(tmp_path / "t.csv"), "--k", "1000", "--min-spread-ticks", "1",
                    "--microprice-skew-coef", "0", "--ofi-skew-coef", "0", *extra], check=True, capture_output=True)
    return pd.read_csv(out / "engine_results.csv")


def test_replay_quotes_sit_at_the_touch(engine_bin, tmp_path):
    r = run_replay(engine_bin, tmp_path, [])
    assert (r.quoted_bid == 100.00).all() and (r.quoted_ask == 100.01).all()
    assert r.bid_fill_qty.sum() == 0 and r.ask_fill_qty.sum() == 0


def test_replay_trade_through_fills_in_full(engine_bin, tmp_path):
    # A sell printed below our bid in second 1 consumed every bid at 100.00, ours included.
    r = run_replay(engine_bin, tmp_path, [(1, 99.99, 5, -1)])
    assert r.loc[0, "bid_fill_qty"] == 25 and r.bid_fill_qty.sum() == 25


def test_replay_fifo_queue_at_the_touch(engine_bin, tmp_path):
    # 100 shares ahead of us. 80 traded at our price in second 1 (queue -> 20), 30 in second 2:
    # the first 20 finish the queue, the next 10 fill us.
    r = run_replay(engine_bin, tmp_path, [(1, 100.00, 80, -1), (2, 100.00, 30, -1)])
    assert r.loc[0, "bid_fill_qty"] == 0
    assert r.loc[1, "bid_fill_qty"] == 10


def test_replay_buys_never_fill_the_bid(engine_bin, tmp_path):
    r = run_replay(engine_bin, tmp_path, [(1, 99.99, 500, 1), (2, 100.00, 500, 1)])
    assert r.bid_fill_qty.sum() == 0
    assert r.ask_fill_qty.sum() == 0  # buys at or below the bid do not reach our 100.01 ask


def test_default_engine_unchanged_by_new_flags(engine_bin, tmp_path):
    f = flat_book(tmp_path / "f.csv")
    a, b = tmp_path / "a", tmp_path / "b"
    subprocess.run([str(engine_bin), "--features", str(f), "--outdir", str(a), "--A", "2"], check=True, capture_output=True)
    subprocess.run([str(engine_bin), "--features", str(f), "--outdir", str(b), "--A", "2", "--fill-model", "parametric",
                    "--pressure-source", "signal"], check=True, capture_output=True)
    ra, rb = (pd.read_csv(d / "engine_results.csv").drop(columns=["latency_us", "latency_budget_exceeded"]) for d in (a, b))
    pd.testing.assert_frame_equal(ra, rb)


def test_ofi_standardisation_is_causal(engine_bin, tmp_path):
    n = 400
    rng = np.random.default_rng(0)
    ofi = rng.normal(0.1, 0.05, n)
    base = pd.read_csv(flat_book(tmp_path / "seed.csv"))
    df = pd.concat([base.iloc[[0]]] * n, ignore_index=True)
    df["ts"] = pd.date_range("2024-06-03 13:30", periods=n, freq="1s", tz="UTC").astype(str)
    df["order_flow_imbalance"] = ofi
    df.to_csv(tmp_path / "a.csv", index=False)
    df2 = df.copy()
    df2.loc[300:, "order_flow_imbalance"] = 5.0  # wild future values
    df2.to_csv(tmp_path / "b.csv", index=False)
    outs = []
    for name in ("a", "b"):
        subprocess.run([str(engine_bin), "--features", str(tmp_path / f"{name}.csv"), "--outdir", str(tmp_path / f"o{name}"),
                        "--ofi-standardize", "--ofi-warmup-rows", "50", "--ofi-skew-coef", "1.0"],
                       check=True, capture_output=True)
        outs.append(pd.read_csv(tmp_path / f"o{name}" / "engine_results.csv"))
    np.testing.assert_allclose(outs[0].reservation_price[:300], outs[1].reservation_price[:300])
    # During warm-up the standardised signal is 0, so the OFI skew contributes nothing.
    np.testing.assert_allclose(outs[0].reservation_price[:50], outs[0].midprice[:50], atol=1e-9)


def test_event_time_features_match_brute_force():
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2024-06-03 13:30", tz="UTC")
    ev_s = np.sort(rng.uniform(0, 20, 300))
    ev_ts = pd.Series(t0 + pd.to_timedelta(ev_s, unit="s"))
    sign = rng.choice([-1, 1], 300)
    df = pd.DataFrame({"ts": pd.date_range(t0, periods=20, freq="1s")})
    out = s5.add_event_time_features(df, ev_ts, sign, half_lives=(0.5,))
    w = np.log(2) / 0.5
    for row in (3, 11, 19):
        end = row + 1.0
        m = ev_s < end
        b = (w * np.exp(-w * (end - ev_s[m & (sign > 0)]))).sum()
        s = (w * np.exp(-w * (end - ev_s[m & (sign < 0)]))).sum()
        assert out.loc[row, "et_log_buy_0p5s"] == pytest.approx(np.log1p(b), rel=1e-9)
        assert out.loc[row, "et_log_sell_0p5s"] == pytest.approx(np.log1p(s), rel=1e-9)


def test_event_time_features_ignore_future_events():
    t0 = pd.Timestamp("2024-06-03 13:30", tz="UTC")
    rng = np.random.default_rng(2)
    ev_s = np.sort(rng.uniform(0, 30, 400))
    sign = rng.choice([-1, 1], 400)
    df = pd.DataFrame({"ts": pd.date_range(t0, periods=30, freq="1s")})
    full = s5.add_event_time_features(df, pd.Series(t0 + pd.to_timedelta(ev_s, unit="s")), sign)
    cut = ev_s < 15.0
    part = s5.add_event_time_features(df, pd.Series(t0 + pd.to_timedelta(ev_s[cut], unit="s")), sign[cut])
    cols = s5.event_time_feature_names()
    pd.testing.assert_frame_equal(full.loc[:13, cols], part.loc[:13, cols])  # rows ending at or before 15 s


def test_live_engine_matches_offline_features(tmp_path):
    pytest.importorskip("websockets")
    import live_monitor as lm
    import step6_binance_validation as s6

    rng = np.random.default_rng(3)
    n = 3000
    t_ms = np.sort(rng.integers(1_717_372_800_000, 1_717_372_800_000 + 200_000, n))
    price = 67000 + np.cumsum(rng.choice([-0.01, 0.0, 0.01], n))
    trades = pd.DataFrame({"ts": pd.to_datetime(t_ms, unit="ms", utc=True), "price": price,
                           "quantity": rng.uniform(0.001, 0.1, n), "trade_sign": rng.choice([-1, 1], n)})
    off, _, _ = s6.build_feature_table(trades)
    hp = {"mu_buy": 0.2, "mu_sell": 0.25, "alpha_bb": 0.12, "alpha_bs": 0.01, "alpha_ss": 0.11, "alpha_sb": 0.01,
          "beta": 0.82}
    off_h = s5.hawkes_one_step(np.array(list(hp.values())), off["buy_events"].to_numpy(float),
                               off["sell_events"].to_numpy(float))
    feats = s6.HAWKES_FEATURES + s5.event_time_feature_names()
    model = {"hawkes_params": hp, "ewma_span": 60, "event_half_lives_s": list(s5.EVENT_HALF_LIVES_S),
             "models": {m: {"features": feats, "mu": [0.0] * len(feats), "sigma": [1.0] * len(feats),
                            "beta": [0.0] * len(feats), "intercept": 0.0} for m in lm.MODEL_NAMES}}
    eng = lm.Engine(model)
    live = []
    for t, p, q, sgn in zip(t_ms, price, trades["quantity"], trades["trade_sign"]):
        while eng.bucket_ms is not None and t >= eng.bucket_ms + 1000:
            live.append(eng.close_bucket())
        eng.on_trade(int(t), float(p), float(q), sgn < 0)
    live.append(eng.close_bucket())
    k = min(len(live), len(off))
    live_df = pd.DataFrame(live[:k])
    np.testing.assert_allclose(live_df["ewm_b"], off["lambda_buy_ewm"][:k], rtol=1e-9)
    np.testing.assert_allclose(live_df["ewma_ofi"], off["order_flow_imbalance"][:k], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(live_df["lam_b"], off_h["lambda_buy_hawkes"][:k], rtol=1e-9)
    np.testing.assert_allclose(live_df["hawkes_ofi"], off_h["hawkes_ofi"][:k], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(live_df["et_ofi_fast"], off["et_ofi_0p1s"][:k], rtol=1e-6, atol=1e-9)
    json.dumps(lm._dumps({"x": float("nan")}))  # NaN-safe serialisation for the browser


EXACT = {"mu": [0.3, 0.4], "omega": [300.0, 5.0, 0.1],
         "alpha": [[[0.20, 0.05], [0.04, 0.18]], [[0.10, 0.02], [0.03, 0.12]], [[0.05, 0.01], [0.01, 0.06]]]}


def test_exact_hawkes_intensity_matches_brute_force():
    import step7_hawkes_ensemble as s7
    rng = np.random.default_rng(7)
    t0 = pd.Timestamp("2024-06-03 13:30", tz="UTC")
    ev_s = np.sort(rng.uniform(0, 15, 250))
    sign = rng.choice([-1, 1], 250)
    df = pd.DataFrame({"ts": pd.date_range(t0, periods=15, freq="1s")})
    out = s7.exact_intensity(df, pd.Series(t0 + pd.to_timedelta(ev_s, unit="s")), sign, EXACT)
    a, w, mu = np.array(EXACT["alpha"]), np.array(EXACT["omega"]), np.array(EXACT["mu"])
    for row in (2, 9, 14):
        end = row + 1.0
        lam = mu.copy()
        for t, sg in zip(ev_s, sign):
            if t < end:
                j = 0 if sg > 0 else 1
                lam += (a[:, :, j] * (w * np.exp(-w * (end - t)))[:, None]).sum(axis=0)
        assert out.loc[row, "hx_log_buy"] == pytest.approx(np.log(lam[0]), rel=1e-7)  # ns timestamp rounding
        assert out.loc[row, "hx_ofi"] == pytest.approx((lam[0] - lam[1]) / lam.sum(), rel=1e-6, abs=1e-9)


def test_live_exact_hawkes_matches_offline():
    pytest.importorskip("websockets")
    import live_monitor as lm
    import step7_hawkes_ensemble as s7

    rng = np.random.default_rng(8)
    n = 2000
    t_ms = np.sort(rng.integers(1_717_372_800_000, 1_717_372_800_000 + 120_000, n))
    sign = rng.choice([-1, 1], n)
    ts = pd.Series(pd.to_datetime(t_ms, unit="ms", utc=True))
    grid = pd.DataFrame({"ts": pd.date_range(ts.iloc[0].floor("1s"), ts.iloc[-1].floor("1s"), freq="1s")})
    off = s7.exact_intensity(grid, ts, sign, EXACT)
    feats = ["order_flow_imbalance"]
    model = {"hawkes_params": {"mu_buy": 0.2, "mu_sell": 0.2, "alpha_bb": 0.1, "alpha_bs": 0.0, "alpha_ss": 0.1,
                               "alpha_sb": 0.0, "beta": 0.8},
             "ewma_span": 60, "event_half_lives_s": [0.1], "exact_hawkes": EXACT,
             "models": {"baseline_ewma": {"features": feats, "mu": [0.0], "sigma": [1.0], "beta": [0.0], "intercept": 0.0}}}
    eng = lm.Engine(model)
    live = []
    for t, sg in zip(t_ms, sign):
        while eng.bucket_ms is not None and t >= eng.bucket_ms + 1000:
            live.append(eng.close_bucket())
        eng.on_trade(int(t), 67000.0, 0.01, sg < 0)
    live.append(eng.close_bucket())
    k = min(len(live), len(off))
    np.testing.assert_allclose(pd.DataFrame(live[:k])["hx_ofi"], off["hx_ofi"][:k], rtol=1e-6, atol=1e-9)


def _maker(**kw):
    pytest.importorskip("websockets")
    import live_monitor as lm
    m = lm.PaperMaker(["with_hawkes"], notional=6500.0, maker_fee_bps=10.0, **kw)
    m.on_book(65000.00, 2.0, 65000.01, 3.0)   # 2 BTC ahead on the bid, 3 BTC on the ask
    m.requote(0, None, float("nan"))           # touch book: 0.1 BTC each side
    return m


def test_maker_fifo_queue_and_through_fill():
    m = _maker()
    b = m.books["touch"]
    m.on_trade(10, 65000.00, 1.5, True)        # sell at our bid: queue 2.0 -> 0.5, no fill
    assert b.inv == 0 and b.bid.queue_ahead == pytest.approx(0.5)
    m.on_trade(20, 65000.00, 0.55, True)       # 0.5 finishes the queue, 0.05 fills us
    assert b.inv == pytest.approx(0.05)
    m.on_trade(30, 64999.99, 0.01, True)       # printed through our bid: rest of our order fills
    assert b.inv == pytest.approx(0.1) and not b.bid.live
    assert b.cash == pytest.approx(-0.1 * 65000.00)
    assert b.fees == pytest.approx(0.1 * 65000.00 * 10 / 1e4)
    m.requote(1100, None, float("nan"))        # 1 s markout resolves at the current mid
    assert b.markout_1s == pytest.approx(0.1 * (65000.005 - 65000.00))


def test_maker_queue_bounded_by_displayed_size_and_kept_on_requote():
    m = _maker()
    b = m.books["touch"]
    m.on_book(65000.00, 0.3, 65000.01, 3.0)    # displayed bid size fell (cancels ahead): queue <= 0.3
    assert b.bid.queue_ahead == pytest.approx(0.3)
    m.requote(1000, None, float("nan"))        # same price: keep our place
    assert b.bid.queue_ahead == pytest.approx(0.3)
    m.on_book(65000.01, 1.0, 65000.02, 1.0)    # touch moved: re-quote joins the back of the new level
    m.requote(2000, None, float("nan"))
    assert b.bid.price == 65000.01 and b.bid.queue_ahead == pytest.approx(1.0)


def test_maker_avellaneda_stoikov_quotes():
    import math
    m = _maker(k=0.26, gamma=0.01, tau_s=10.0)
    sigma = 4.5
    m.requote(1000, None, sigma)
    half = (1 / 0.01) * math.log(1 + 0.01 / 0.26) + 0.5 * 0.01 * sigma ** 2 * 10
    q = m.books["as_none"].last_quote
    assert q[0] == pytest.approx(math.floor((65000.005 - half) / 0.01) * 0.01)
    assert q[1] == pytest.approx(math.ceil((65000.005 + half) / 0.01) * 0.01)
    assert m.books["as_none"].bid.queue_ahead == math.inf   # behind the touch: through-trades only


def test_maker_signal_pull_and_inventory_cap():
    m = _maker(inventory_limit=6000.0)
    for i in range(80):                        # history for the causal standardisation
        m.requote(1000 * (i + 1), {"with_hawkes": 1e-6 * (-1) ** i}, 4.5)
    m.requote(100_000, {"with_hawkes": 5e-5}, 4.5)   # strong predicted rise: pull the ask
    hb = m.books["as_with_hawkes"]
    assert hb.ask.live is False and hb.bid.live is True and hb.pulls >= 1
    assert m.books["as_none"].ask.live is True
    tb = m.books["touch"]
    tb.inv = 0.1                               # $6,500 long > $6,000 cap: stop bidding
    m.requote(101_000, None, 4.5)
    assert tb.bid.live is False and tb.ask.live is True


class _Flow:
    """Stand-in for Engine.exact_lambda: settable (buy, sell) intensities."""
    def __init__(self):
        self.lam = (1.0, 1.0)

    def __call__(self, t_ms):
        return self.lam


def _ev_maker(latency_ms=50.0):
    pytest.importorskip("websockets")
    import live_monitor as lm
    flow = _Flow()
    m = lm.PaperMaker(["hawkes_ensemble"], notional=6500.0, k=0.26, gamma=0.01, tau_s=10.0,
                      flow_fn=flow, ev_signal="hawkes_ensemble", latency_ms=latency_ms)
    m.on_book(65000.00, 2.0, 65000.01, 3.0, t_ms=0)
    m.requote(1000, None, 0.5)                   # sets sigma; schedules the event-driven quotes for t = 1050
    return m, flow


def test_event_driven_quotes_take_effect_only_after_latency():
    m, _ = _ev_maker()
    ev = m.books["ev_none"]
    assert not ev.bid.live                       # decided at 1000, not yet at the market
    m.on_trade(1049, 65000.02, 0.01, False)      # before 1050: nothing to fill
    assert not ev.bid.live and ev.fills == 0
    m.on_trade(1051, 65000.02, 0.01, False)      # the pending quote is applied first
    assert ev.bid.live and ev.ask.live
    assert ev.last_quote == m.books["as_none"].last_quote   # same A-S prices, only the timing differs


def test_event_driven_reprices_on_book_change_with_latency():
    m, _ = _ev_maker()
    m.apply_pending(1050)
    ev = m.books["ev_none"]
    old = ev.bid.price
    m.on_book(65000.50, 1.0, 65000.51, 1.0, t_ms=1200)   # touch moves up 50 cents
    assert ev.bid.price == old                    # still the stale quote for 50 ms
    m.on_book(65000.50, 1.0, 65000.51, 1.0, t_ms=1250)
    assert ev.bid.price > old


def test_event_driven_hawkes_pulls_threatened_side_and_post_only():
    m, flow = _ev_maker(latency_ms=10.0)
    for i in range(m.BURST_WARMUP + m.BURST_REFRESH):   # history of the log intensity ratio
        flow.lam = (1.0 + 0.05 * (-1) ** i, 1.0)
        m.on_flow(2000 + i)
    m.apply_pending(10_000)
    hb, nb = m.books["ev_hawkes"], m.books["ev_none"]
    assert hb.ask.live and hb.bid.live
    flow.lam = (50.0, 1.0)                       # buy self-excitation burst
    m.on_flow(20_000)
    m.apply_pending(20_010)
    assert not hb.ask.live and hb.bid.live and hb.burst_pulls >= 1
    assert nb.ask.live                           # the no-Hawkes control keeps quoting
    m.ev_pending.append((20_020, "ev_none", 65000.02, None))  # a bid that would cross the 65000.01 ask
    m.apply_pending(20_020)
    assert not nb.bid.live                       # post-only: rejected rather than executed


def _fast_exec():
    pytest.importorskip("websockets")
    import live_monitor as lm
    z = {"v": 0}
    ex = lm.Executor(["hawkes_ensemble"], notional=6500.0, every_s=60, window_s=10,
                     burst_fn=lambda t: z["v"], latency_ms=50.0, fast_exec=True)
    ex.on_book(65000.00, 2.0, 65000.01, 3.0, t_ms=0)
    ex.on_second(1000, {"hawkes_ensemble": -1e-5})   # buy order; forecast favourable: timing waits, passive rests
    return ex, z


def test_repeg_passive_follows_touch_after_latency_and_default_books_do_not():
    ex, _ = _fast_exec()
    p = ex.books["repeg_passive_none"].parent
    assert p.order.price == 65000.00
    ex.on_book(65000.05, 1.0, 65000.06, 1.0, t_ms=1100)   # bid moves up
    assert p.order.price == 65000.00                       # still in flight
    ex.on_book(65000.05, 1.0, 65000.06, 1.0, t_ms=1150)
    assert p.order.price == 65000.05 and p.order.queue_ahead == 1.0
    assert ex.books["passive_none"].parent.order.price == 65000.00   # the slow control never moves
    assert ex.books["passive_hawkes_ensemble"].parent.order.price == 65000.00   # fast Hawkes books do not re-peg


def test_fast_hawkes_crosses_on_adverse_burst_with_latency():
    ex, z = _fast_exec()
    fb, ft = ex.books["passive_hawkes_ensemble"], ex.books["timing_hawkes_ensemble"]   # fast by default
    z["v"] = -1                                    # sell burst: favourable for a buy, keep waiting
    ex.on_flow(1200)
    assert not ex.pending
    z["v"] = 1                                     # buy burst against our buy
    ex.on_flow(1300)
    assert fb.parent.cross_sent and ft.parent.cross_sent and fb.fast_crosses == 1
    ex.on_book(65000.00, 2.0, 65000.01, 3.0, t_ms=1340)
    assert fb.parent.remaining > 0                 # not yet at the market
    ex.on_book(65000.02, 2.0, 65000.03, 3.0, t_ms=1350)   # arrives 50 ms later: pays the ask at that time
    assert fb.parent.remaining == 0 and fb.parent.price_cost > 0
    assert ex.books["slow_passive_hawkes"].parent.remaining > 0   # the per-second book is still resting
    assert ex.books["slow_timing_hawkes"].parent.remaining > 0


def test_python_replay_rule_matches_cpp_engine(engine_bin, tmp_path):
    import reconcile_live_maker as rl
    trades = [(1, 100.00, 80, -1), (2, 100.00, 30, -1), (3, 99.99, 5, -1), (4, 100.01, 7, 1), (5, 100.02, 1, 1)]
    r = run_replay(engine_bin, tmp_path, trades)           # engine quotes 100.00 / 100.01 at the touch, 25 shares
    by_row = {i: np.array([[p, sz, sg]], dtype=float) for i, p, sz, sg in trades}
    bid_o, ask_o = rl.Resting(), rl.Resting()
    py_bid, py_ask = [], []
    for i in range(len(r)):
        nxt = by_row.get(i + 1)
        py_bid.append(rl.replay_fill(bid_o, True, 100.00, 100.00, 100.0, True, 25, 0.01, nxt))
        py_ask.append(rl.replay_fill(ask_o, True, 100.01, 100.01, 100.0, False, 25, 0.01, nxt))
    np.testing.assert_allclose(py_bid, r.bid_fill_qty.to_numpy())
    np.testing.assert_allclose(py_ask, r.ask_fill_qty.to_numpy())


def _executor():
    pytest.importorskip("websockets")
    import live_monitor as lm
    ex = lm.Executor(["hawkes_ensemble"], notional=6500.0, every_s=60, window_s=10)
    ex.on_book(65000.00, 2.0, 65000.10, 2.0)   # 10-cent spread to make savings visible
    return ex


def test_executor_passive_fill_saves_the_spread():
    ex = _executor()
    ex.on_second(1_000, {"hawkes_ensemble": -1e-5})   # launch a buy; forecast says the price will fall: wait
    b = ex.books["passive_hawkes_ensemble"]
    assert b.parent is not None and b.parent.order.price == 65000.00
    ex.on_trade(64999.90, 0.01, True)                 # sell printed through our bid: maker fill in full
    ex.on_second(2_000, {"hawkes_ensemble": -1e-5})
    assert b.orders == 1 and b.parent is None
    size = 6500.0 / 65000.05
    assert b.price_saved == pytest.approx(0.10 * size)  # bought at the bid instead of the ask
    assert b.fee_diff == pytest.approx(65000.00 * size * 10 / 1e4 - 65000.10 * size * 10 / 1e4)


def test_executor_timing_waits_then_crosses_on_adverse_forecast():
    ex = _executor()
    ex.on_second(1_000, {"hawkes_ensemble": -1e-5})    # buy; price expected to fall: wait
    t = ex.books["timing_hawkes_ensemble"]
    assert t.parent is not None and t.orders == 0
    ex.on_book(64999.90, 2.0, 65000.00, 2.0)           # price fell 10 cents
    ex.on_second(2_000, {"hawkes_ensemble": +1e-5})     # now expected to rise: buy at the ask
    assert t.orders == 1
    size = 6500.0 / 65000.05
    assert t.price_saved == pytest.approx((65000.10 - 65000.00) * size)


def test_executor_control_crosses_at_deadline():
    ex = _executor()
    ex.on_second(1_000, None)
    c = ex.books["passive_none"]
    for s in range(2, 11):
        ex.on_second(s * 1000, None)
    assert c.parent is not None                      # 10 s window not over before 11 s
    ex.on_second(11_000, None)
    assert c.orders == 1 and c.parent is None and c.maker_qty == 0
