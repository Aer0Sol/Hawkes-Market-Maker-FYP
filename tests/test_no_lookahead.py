"""Leakage tests: features at time t must not change when the future is altered or removed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hawkes_diagnostics as hd
import step1_build_feature_table as s1
import step2_trade_label_and_hawkes as s2
import step5_alpha_signal as s5
import step6_binance_validation as s6

PARAMS = np.array([0.04, 0.03, 0.03, 0.01, 0.02, 0.02, 0.94])


@pytest.fixture
def counts() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.poisson(3, 2000).astype(float), rng.poisson(2, 2000).astype(float)


def test_hawkes_recursion_ignores_future(counts):
    yb, ys = counts
    lb, ls = s2.simulate_discrete_hawkes(PARAMS, yb, ys)
    t = 1200
    yb2, ys2 = yb.copy(), ys.copy()
    yb2[t:] = np.random.default_rng(1).permutation(yb2[t:]) + 50  # scramble and shift the future
    ys2[t:] = 0
    lb2, ls2 = s2.simulate_discrete_hawkes(PARAMS, yb2, ys2)
    # lambda[t] uses counts up to t-1 only, so rows <= t are unchanged.
    np.testing.assert_array_equal(lb[: t + 1], lb2[: t + 1])
    np.testing.assert_array_equal(ls[: t + 1], ls2[: t + 1])
    lb3, _ = s2.simulate_discrete_hawkes(PARAMS, yb[:t], ys[:t])
    np.testing.assert_array_equal(lb[:t], lb3)


def test_fast_recursion_matches_step2_and_naive_sum(counts):
    yb, ys = counts
    lb, ls = s2.simulate_discrete_hawkes(PARAMS, yb, ys)
    fb, fs = hd.discrete_intensities(PARAMS, yb, ys)
    np.testing.assert_allclose(lb, fb, rtol=1e-12)
    np.testing.assert_allclose(ls, fs, rtol=1e-12)
    # Naive O(n^2) closed form of the recursion for one row.
    mu_b, _, a_bb, a_bs, _, _, beta = PARAMS
    t = 500
    naive = beta ** t * mu_b + sum(beta ** (t - u) * (mu_b + a_bb * yb[u - 1] + a_bs * ys[u - 1])
                                   for u in range(1, t + 1))
    assert lb[t] == pytest.approx(naive, rel=1e-10)


def test_ewma_intensity_ignores_future(counts):
    yb, ys = counts
    df = pd.DataFrame({"ts": pd.date_range("2024-06-03", periods=len(yb), freq="1s"),
                       "buy_events": yb, "sell_events": ys})
    full = s2.estimate_baseline_intensities(df)
    cut = s2.estimate_baseline_intensities(df.iloc[:1000])
    cols = ["lambda_buy_ewm", "lambda_sell_ewm", "order_flow_imbalance"]
    pd.testing.assert_frame_equal(full[cols].iloc[:1000], cut[cols])


def test_rolling_volatility_ignores_future():
    rng = np.random.default_rng(2)
    n = 3000
    ts = pd.date_range("2024-06-03 13:30", periods=n, freq="250ms", tz="UTC")
    mid = 100 + np.cumsum(rng.normal(0, 0.01, n))
    mbp = pd.DataFrame({"ts": ts, "bid_px_1": mid - 0.005, "ask_px_1": mid + 0.005, "bid_sz_1": 100.0,
                        "ask_sz_1": 100.0, "depth_bid_10": 500.0, "depth_ask_10": 500.0, "midprice": mid,
                        "microprice": mid, "spread": 0.01, "spread_bps": 1.0, "imbalance_l1": 0.0,
                        "imbalance_l10": 0.0})
    full = s1.resample_mbp_features(mbp)
    cut = s1.resample_mbp_features(mbp.iloc[: n // 2])
    m = len(cut) - 1  # the last bucket of the truncated table is incomplete
    for col in ["midprice", "rv_60", "rv_300", "ret_1", "microprice_alpha"]:
        pd.testing.assert_series_equal(full[col].iloc[:m], cut[col].iloc[:m], check_names=False)


def test_lee_ready_uses_only_past_quotes():
    # Feature row labelled 10:00:00 holds the book at the END of that second.
    feats = pd.DataFrame({
        "ts": pd.to_datetime(["2024-06-03 10:00:00", "2024-06-03 10:00:01"], utc=True),
        "bid_px_1": [99.99, 100.09], "ask_px_1": [100.01, 100.11], "midprice": [100.00, 100.10],
    })
    trades = pd.DataFrame({
        "ts": pd.to_datetime(["2024-06-03 10:00:01.5", "2024-06-03 10:00:00.5"], utc=True),
        "price": [100.02, 100.05], "size": [100, 100], "native_side": [np.nan, np.nan],
    })
    lab = s2.lee_ready_label(trades, feats).sort_values("ts").reset_index(drop=True)
    # 10:00:00.5 has no quote known yet (the 10:00:00 row is only known at 10:00:01).
    assert np.isnan(lab.loc[0, "bid_px_1"])
    # 10:00:01.5 sees the 10:00:00 row (book as of 10:00:01), not the 10:00:01 row.
    assert lab.loc[1, "midprice"] == 100.00
    assert lab.loc[1, "trade_sign"] == 1 and lab.loc[1, "label_source"] == "quote"


@pytest.mark.filterwarnings("ignore:Mean of empty slice")
def test_native_side_overrides_quote_rule():
    feats = pd.DataFrame({"ts": pd.to_datetime(["2024-06-03 10:00:00"], utc=True),
                          "bid_px_1": [99.99], "ask_px_1": [100.01], "midprice": [100.0]})
    trades = pd.DataFrame({"ts": pd.to_datetime(["2024-06-03 10:00:02"], utc=True), "price": [100.01],
                           "size": [10], "native_side": [-1.0]})
    lab = s2.lee_ready_label(trades, feats)
    assert lab.loc[0, "trade_sign"] == -1 and lab.loc[0, "label_source"] == "native"


@pytest.mark.parametrize("mod", [s1, s2])
def test_aggressor_side_mapping(mod):
    got = mod.map_aggressor_side(pd.Series(["B", "A", "N", "b", "-1", "1", None, "Buy", "Sell"]))
    np.testing.assert_array_equal(got, [1, -1, np.nan, 1, -1, 1, np.nan, 1, -1])


def test_flow_target_is_shifted_not_leaked():
    # future_ofi_1s at row t must equal the (buy-sell)/(buy+sell) imbalance realized at row t+1,
    # not anything computable from row t's own counts.
    df = pd.DataFrame({
        "ts": pd.date_range("2024-06-03", periods=5, freq="1s", tz="UTC"),
        "buy_events": [10, 4, 0, 6, 2], "sell_events": [0, 6, 0, 2, 2],
    })
    total = df["buy_events"] + df["sell_events"]
    expected_next = ((df["buy_events"] - df["sell_events"]) / total.replace(0, np.nan)).shift(-1)
    out = s5.add_flow_target(df)
    got = out[s5.FLOW_TARGET]
    pd.testing.assert_series_equal(got, expected_next, check_names=False)
    assert pd.isna(got.iloc[-1])  # last row has no future bucket: must be NaN, not leaked/guessed


def test_walk_forward_folds_never_see_their_own_future():
    folds = s5.walk_forward_folds(n=1000, n_folds=5, warmup_frac=0.2)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()  # strictly chronological: train ends before test starts
        assert set(train_idx).isdisjoint(set(test_idx))
    # expanding window: later folds' training sets are supersets of earlier ones
    for (tr1, _), (tr2, _) in zip(folds, folds[1:]):
        assert len(tr2) > len(tr1)
        assert set(tr1).issubset(set(tr2))


def test_per_fold_hawkes_refit_ignores_test_window(counts):
    yb, ys = counts
    df = pd.DataFrame({"buy_events": yb, "sell_events": ys})
    end = 1500
    scrambled = df.copy()
    scrambled.loc[end:, "buy_events"] = np.random.default_rng(5).poisson(20, len(df) - end)
    a = s5.hawkes_features_fitted_on(df, end)
    b = s5.hawkes_features_fitted_on(scrambled, end)
    # Parameters come from rows < end only, and row t's one-step-ahead intensity uses counts up to
    # t, so every row before `end` is identical whatever the test window holds.
    pd.testing.assert_frame_equal(a.iloc[:end], b.iloc[:end])


def test_one_step_ahead_hawkes_matches_shifted_recursion(counts):
    yb, ys = counts
    df = pd.DataFrame({"buy_events": yb, "sell_events": ys})
    nxt = s5.hawkes_features_fitted_on(df, 1500)
    fit = hd.fit_discrete(yb[:1500], ys[:1500])
    lb, _ = hd.discrete_intensities(fit["x"], yb, ys)
    # lambda_{t+1|t} from the closed form equals the recursion's next value.
    np.testing.assert_allclose(nxt["lambda_buy_hawkes"].to_numpy()[:-1], lb[1:], rtol=1e-12)


def test_diebold_mariano_hac_is_wider_than_naive_on_autocorrelated_diffs():
    rng = np.random.default_rng(6)
    e = rng.normal(size=5000)
    d = 0.02 + np.convolve(e, np.ones(10) / 10, mode="same")  # strongly autocorrelated, small mean
    _, p_hac = s5.diebold_mariano_hac(d, lag=20)
    p_naive = __import__("scipy").stats.ttest_1samp(d, 0.0).pvalue
    assert p_hac > p_naive


def test_binance_aggressor_mapping():
    # Binance: is_buyer_maker=True means the buyer was resting, so a sell order hit the bid ->
    # the aggressor is the seller. False means the buyer crossed the spread -> aggressor is buyer.
    got = s6.map_binance_aggressor(pd.Series([True, False, True, False]))
    np.testing.assert_array_equal(got, [-1, 1, -1, 1])


def test_ridge_fit_predict_uses_only_train_fold_statistics():
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(200, 3))
    y_train = x_train[:, 0] * 2.0 + rng.normal(scale=0.01, size=200)
    x_test = rng.normal(size=(50, 3))
    pred_a = s5.ridge_fit_predict(x_train, y_train, x_test, alpha=1.0)
    # Perturbing the test features must not change train-fold standardization (mu/sigma come
    # from x_train only), so predictions for the first row are identical whatever follows it.
    x_test_shrunk = x_test[:10].copy()
    pred_b = s5.ridge_fit_predict(x_train, y_train, x_test_shrunk, alpha=1.0)
    np.testing.assert_allclose(pred_a[:10], pred_b, rtol=1e-12)
