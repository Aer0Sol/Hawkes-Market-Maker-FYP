"""Phase 0 smoke tests: pipeline artifacts exist and are sane."""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from conftest import STEP1_FEATURES, STEP2_DIR, STEP2_FEATURES_CSV, require

STEP1_COLUMNS = [
    "ts", "bid_px_1", "ask_px_1", "bid_sz_1", "ask_sz_1", "depth_bid_10", "depth_ask_10",
    "midprice", "microprice", "spread", "spread_bps", "imbalance_l1", "imbalance_l10",
    "log_mid", "ret_1", "rv_60", "rv_300", "mid_move", "microprice_alpha",
    "trade_count", "trade_volume", "signed_volume", "buy_count", "sell_count",
    "buy_volume", "sell_volume", "vwap", "last_trade_price", "trade_imbalance",
    "future_ret_1s", "future_ret_5s", "future_ret_10s",
]
STEP2_EXTRA_COLUMNS = [
    "buy_events", "sell_events", "lambda_buy_poisson", "lambda_sell_poisson",
    "lambda_buy_ewm", "lambda_sell_ewm", "lambda_total_ewm", "order_flow_imbalance",
    "lambda_buy_hawkes", "lambda_sell_hawkes", "hawkes_ofi",
]
# Fields the engine needs on every row (rv_60 is allowed a 60-row warm-up of NaN).
ENGINE_REQUIRED = ["bid_px_1", "ask_px_1", "midprice", "spread", "microprice", "imbalance_l1"]
ENGINE_RESULTS_COLUMNS = (
    "ts,midprice,next_midprice,best_bid,best_ask,reservation_price,quoted_bid,quoted_ask,half_spread,"
    "inventory,cash,mtm_pnl,sigma,bid_distance,ask_distance,buy_pressure,sell_pressure,"
    "queue_haircut_bid,queue_haircut_ask,toxicity_haircut_bid,toxicity_haircut_ask,"
    "p_bid_fill,p_ask_fill,bid_fill,ask_fill,bid_fill_qty,ask_fill_qty,can_bid,can_ask,"
    "quote_bid_active,quote_ask_active,fee_paid,bid_markout,ask_markout,adverse_selection_cost,"
    "realized_spread_component,latency_us,latency_budget_exceeded"
).split(",")


@pytest.fixture(scope="module")
def step1() -> pd.DataFrame:
    return pd.read_parquet(require(STEP1_FEATURES, "committed step 1 output"))


@pytest.fixture(scope="module")
def step2() -> pd.DataFrame:
    return pd.read_csv(require(STEP2_FEATURES_CSV, "committed step 2 output"))


def test_step1_feature_table_columns(step1):
    assert list(step1.columns) == STEP1_COLUMNS


def test_step1_one_second_grid(step1):
    ts = pd.to_datetime(step1["ts"], utc=True)
    assert ts.is_monotonic_increasing
    assert (ts.diff().dropna() == pd.Timedelta("1s")).all()
    assert len(step1) == 23400  # 09:30-16:00 ET


def test_step1_required_fields_not_nan(step1):
    assert not step1[ENGINE_REQUIRED].isna().any().any()
    assert step1["rv_60"].isna().sum() == 60
    assert (step1["spread"] > 0).all()


def test_step2_feature_table_columns(step2):
    assert list(step2.columns) == STEP1_COLUMNS + STEP2_EXTRA_COLUMNS
    assert not step2[ENGINE_REQUIRED + STEP2_EXTRA_COLUMNS].isna().any().any()


def test_step2_hawkes_params_stable():
    summary = json.loads((STEP2_DIR / "step2_summary.json").read_text())
    fit = summary["hawkes_fit"]
    assert fit["success"]
    p = fit["params"]
    assert 0.0 <= p["beta"] < 1.0
    assert p["mu_buy"] > 0 and p["mu_sell"] > 0
    assert all(p[k] >= 0 for k in ("alpha_bb", "alpha_bs", "alpha_ss", "alpha_sb"))
    # Spectral radius of the one-step companion matrix must be < 1 (stationarity).
    m = np.array([[p["beta"] + p["alpha_bb"], p["alpha_bs"]], [p["alpha_sb"], p["beta"] + p["alpha_ss"]]])
    assert max(abs(np.linalg.eigvals(m))) < 1.0


def test_step2_hawkes_refit_reproduces_committed_params():
    """Refitting on the committed bucket counts reproduces the committed parameters."""
    import step2_trade_label_and_hawkes as s2

    counts = pd.read_parquet(require(STEP2_DIR / "event_counts_1s.parquet", "committed step 2 output"))
    fit = s2.fit_discrete_hawkes(counts)
    ref = json.loads((STEP2_DIR / "step2_summary.json").read_text())["hawkes_fit"]
    for k, v in ref["params"].items():
        assert fit["params"][k] == pytest.approx(v, rel=1e-3, abs=1e-6), k
    assert fit["fun"] == pytest.approx(ref["fun"], rel=1e-7)


def test_engine_runs_and_writes_results(engine_bin, tmp_path):
    proc = subprocess.run(
        [str(engine_bin), "--features", str(STEP2_FEATURES_CSV), "--outdir", str(tmp_path), "--use-hawkes"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    res = pd.read_csv(tmp_path / "engine_results.csv")
    assert list(res.columns) == ENGINE_RESULTS_COLUMNS
    assert len(res) == 23400
    summary = json.loads((tmp_path / "step3_summary.json").read_text())
    assert summary["n_steps"] == 23400
    assert np.isfinite(res["mtm_pnl"]).all()
    assert (res["inventory"].abs() <= 500).all()


def test_engine_rejects_unknown_flag(engine_bin, tmp_path):
    proc = subprocess.run([str(engine_bin), "--features", str(STEP2_FEATURES_CSV), "--bogus"],
                          capture_output=True, text=True)
    assert proc.returncode == 1
    assert "Unknown argument" in proc.stderr
