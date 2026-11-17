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
