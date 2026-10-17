"""Engine behaviour on small synthetic feature tables (fixed semantics, see docs/changes.md)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def make_features(path: Path, n: int = 400, ofi: float = 0.0, drift: float = 0.0, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    mid = 100.0 + drift * np.arange(n) + np.cumsum(rng.choice([-0.005, 0.0, 0.005], size=n))
    bid = np.round(mid - 0.005, 2)
    ask = bid + 0.01
    df = pd.DataFrame({
        "ts": pd.date_range("2024-06-03 13:30", periods=n, freq="1s", tz="UTC").astype(str),
        "bid_px_1": bid, "ask_px_1": ask, "bid_sz_1": 300.0, "ask_sz_1": 300.0,
        "depth_bid_10": 1500.0, "depth_ask_10": 1500.0, "midprice": (bid + ask) / 2, "spread": 0.01,
        "microprice": (bid + ask) / 2, "rv_60": 6e-5, "imbalance_l1": 0.0,
        "lambda_buy_ewm": 4.0, "lambda_sell_ewm": 4.0, "order_flow_imbalance": ofi,
        "lambda_buy_hawkes": 4.0, "lambda_sell_hawkes": 4.0, "hawkes_ofi": ofi,
    })
    df.to_csv(path, index=False)
    return path


def run(engine_bin: Path, features: Path, outdir: Path, *args: str) -> tuple[dict, pd.DataFrame]:
    subprocess.run([str(engine_bin), "--features", str(features), "--outdir", str(outdir), *args],
                   check=True, capture_output=True)
    return json.loads((outdir / "step3_summary.json").read_text()), pd.read_csv(outdir / "engine_results.csv")


def test_markout_positive_means_profitable(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv")
    _, r = run(engine_bin, f, tmp_path / "o", "--A", "1.0", "--k", "4", "--markout-horizon-steps", "3")
    b = r[r.bid_fill == 1]
    a = r[r.ask_fill == 1]
    assert len(b) and len(a)
    np.testing.assert_allclose(b.bid_markout, b.next_midprice - b.quoted_bid, atol=1e-7)
    np.testing.assert_allclose(a.ask_markout, a.quoted_ask - a.next_midprice, atol=1e-7)
    # Quotes are far from mid in a flat market, so fills are profitable at the horizon.
    assert b.bid_markout.mean() > 0 and a.ask_markout.mean() > 0


def test_adverse_selection_is_mid_move_against_position(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv", drift=-0.002)  # falling market: bought inventory loses
    _, r = run(engine_bin, f, tmp_path / "o", "--A", "1.0", "--k", "4", "--markout-horizon-steps", "5")
    expected = (np.maximum(0, r.midprice - r.next_midprice) * r.bid_fill_qty
                + np.maximum(0, r.next_midprice - r.midprice) * r.ask_fill_qty)
    np.testing.assert_allclose(r.adverse_selection_cost, expected, atol=1e-6)
    assert r.loc[r.bid_fill == 1, "adverse_selection_cost"].mean() > 0


def test_buy_flow_threatens_the_ask(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv", ofi=0.6)
    s, r = run(engine_bin, f, tmp_path / "o")
    assert (r.quote_ask_active == 0).all() and (r.quote_bid_active == 1).all()
    assert s["ask_quote_pulls"] == len(r) and s["bid_quote_pulls"] == 0
    assert (r.toxicity_haircut_ask < 1).all() and (r.toxicity_haircut_bid == 1).all()


def test_sell_flow_widens_bid_without_pull(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv", ofi=-0.3)
    no_skew = ["--ofi-skew-coef", "0", "--A", "1e-9"]  # isolate widening: no OFI skew, no fills/inventory
    _, widened = run(engine_bin, f, tmp_path / "o", *no_skew)
    f0 = make_features(tmp_path / "f0.csv", ofi=0.0)
    _, base = run(engine_bin, f0, tmp_path / "o0", *no_skew)
    assert (widened.quote_bid_active == 1).all()
    np.testing.assert_allclose(widened.bid_distance - base.bid_distance, 0.03, atol=1e-7)  # 3 ticks
    np.testing.assert_allclose(widened.ask_distance, base.ask_distance, atol=1e-7)


def test_pull_threshold_is_configurable(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv", ofi=0.6)
    _, r = run(engine_bin, f, tmp_path / "o", "--toxic-pull-threshold", "0.9")
    assert (r.quote_ask_active == 1).all()


def test_inventory_limit_is_hard(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv", n=2000, drift=-0.003)
    s, r = run(engine_bin, f, tmp_path / "o", "--A", "5", "--inventory-limit", "100", "--gamma", "0.01")
    assert r.inventory.abs().max() <= 100
    assert s["risk_reject_steps"] > 0


def test_inventory_skews_reservation_price(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv", n=1500, drift=-0.002)
    _, r = run(engine_bin, f, tmp_path / "o", "--A", "2", "--k", "20")
    long_ = r.inventory > 50
    assert long_.any()
    # Long inventory pulls the reservation price below mid (A-S inventory term is active).
    assert (r.loc[long_, "reservation_price"] < r.loc[long_, "midprice"]).all()


def test_legacy_session_tau_makes_inventory_term_negligible(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv", n=1500, drift=-0.002)
    _, r = run(engine_bin, f, tmp_path / "o", "--A", "2", "--k", "20", "--risk-horizon-s", "0")
    skew = (r.reservation_price - r.midprice).abs().max()
    assert skew < 1e-3


def test_summary_records_seed_and_flags(engine_bin, tmp_path):
    f = make_features(tmp_path / "f.csv")
    s, _ = run(engine_bin, f, tmp_path / "o", "--seed", "7", "--disable-one-sided-pull")
    assert s["params"]["seed"] == 7
    assert s["params"]["allow_one_sided_pull"] is False
