from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path.suffix}")


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def parse_timestamp(df: pd.DataFrame) -> pd.Series:
    candidates = ["ts", "ts_event", "ts_recv", "timestamp", "time", "datetime"]
    col = find_first_existing(df, candidates)
    if col is None:
        raise KeyError(f"Could not find timestamp column. Available columns: {list(df.columns)}")

    s = df[col]
    if isinstance(s.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_dtype(s):
        return pd.to_datetime(s, utc=True)
    if np.issubdtype(s.dtype, np.number):
        s_nonnull = s.dropna()
        vmax = s_nonnull.max()
        if vmax > 1e17:
            return pd.to_datetime(s, unit="ns", utc=True)
        elif vmax > 1e14:
            return pd.to_datetime(s, unit="us", utc=True)
        elif vmax > 1e11:
            return pd.to_datetime(s, unit="ms", utc=True)
        else:
            return pd.to_datetime(s, unit="s", utc=True)

    return pd.to_datetime(s, utc=True, errors="coerce")


# Databento trades carry the aggressor side: 'B' = buy aggressor, 'A' = sell aggressor,
# 'N' = none/unknown. Exact token matching; the old substring regex mapped 'A' to
# NaN and '-1' to buy (it contains '1').
_BUY_TOKENS = {"b", "buy", "bid", "1", "1.0", "true"}
_SELL_TOKENS = {"a", "s", "sell", "ask", "-1", "-1.0", "false"}


def map_aggressor_side(series: pd.Series) -> np.ndarray:
    raw = series.astype(str).str.strip().str.lower()
    return np.where(raw.isin(_BUY_TOKENS), 1.0, np.where(raw.isin(_SELL_TOKENS), -1.0, np.nan))


def normalize_price(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s_nonnull = s.dropna()
    if len(s_nonnull) == 0:
        return s
    median_abs = s_nonnull.abs().median()
    if median_abs > 10_000:
        return s / 1e9
    if median_abs > 1_000 and median_abs < 10_000_000:
        return s / 1e4
    return s


def prepare_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ts" not in df.columns:
        df["ts"] = parse_timestamp(df)
    else:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.sort_values("ts").reset_index(drop=True)
    needed = ["ts", "bid_px_1", "ask_px_1", "midprice", "spread", "microprice"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Feature table missing columns: {missing}")
    return df


def prepare_trades_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = parse_timestamp(df)

    price_col = find_first_existing(df, ["price", "px"])
    size_col = find_first_existing(df, ["size", "qty", "quantity"])
    side_col = find_first_existing(df, ["side", "aggressor_side", "native_side", "is_aggressor"])

    if price_col is None or size_col is None:
        raise KeyError(f"Could not find trade price/size columns. Available columns: {list(df.columns)}")

    df["price"] = normalize_price(df[price_col])
    df["size"] = pd.to_numeric(df[size_col], errors="coerce")

    if side_col is not None:
        df["native_side"] = map_aggressor_side(df[side_col])
    else:
        df["native_side"] = np.nan

    keep = ["ts", "price", "size", "native_side"]
    return df[keep].dropna(subset=["ts", "price", "size"]).sort_values("ts").reset_index(drop=True)


def lee_ready_label(trades: pd.DataFrame, features: pd.DataFrame, lag_ms: int = 0) -> pd.DataFrame:
    tr = trades.copy()
    ft = features[["ts", "bid_px_1", "ask_px_1", "midprice"]].copy().sort_values("ts").reset_index(drop=True)
    # Step 1 rows are labelled by bucket start but hold the book at the bucket END
    # (resample(...).last()). Re-stamp them at the end so a trade is only matched to
    # quotes that existed before it (previously: up to one bucket of look-ahead).
    bucket = ft["ts"].diff().median()
    if pd.notna(bucket):
        ft["ts"] = ft["ts"] + bucket

    if lag_ms != 0:
        tr["ts_match"] = tr["ts"] - pd.to_timedelta(lag_ms, unit="ms")
    else:
        tr["ts_match"] = tr["ts"]

    tr = pd.merge_asof(
        tr.sort_values("ts_match"),
        ft,
        left_on="ts_match",
        right_on="ts",
        direction="backward",
        suffixes=("", "_quote"),
    ).rename(columns={"ts_x": "ts_trade", "ts_y": "ts_quote"})

    if "ts_trade" in tr.columns:
        tr["ts"] = tr["ts_trade"]
        tr = tr.drop(columns=[c for c in ["ts_trade", "ts_quote", "ts_match"] if c in tr.columns])
    else:
        tr = tr.drop(columns=[c for c in ["ts_match"] if c in tr.columns])

    # The exchange-reported aggressor side is ground truth when present; Lee-Ready
    # (quote rule, then tick rule) only labels trades without it.
    tr["lr_side"] = tr["native_side"]
    tr["label_source"] = np.where(tr["native_side"].notna(), "native", "")
    open_ = tr["lr_side"].isna()

    buy_mask = open_ & (tr["price"] >= tr["ask_px_1"])
    sell_mask = open_ & (tr["price"] <= tr["bid_px_1"])
    mid_buy = open_ & (tr["price"] > tr["midprice"]) & (~buy_mask) & (~sell_mask)
    mid_sell = open_ & (tr["price"] < tr["midprice"]) & (~buy_mask) & (~sell_mask)

    tr.loc[buy_mask, "lr_side"] = 1
    tr.loc[sell_mask, "lr_side"] = -1
    tr.loc[mid_buy, "lr_side"] = 1
    tr.loc[mid_sell, "lr_side"] = -1
    tr.loc[open_ & tr["lr_side"].notna(), "label_source"] = "quote"

    unresolved = tr["lr_side"].isna()
    if unresolved.any():
        price_diff = tr["price"].diff()
        tick_rule = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, np.nan))
        tr.loc[unresolved, "lr_side"] = tick_rule[unresolved]
        tr.loc[unresolved & tr["lr_side"].notna(), "label_source"] = "tick"

    unresolved = tr["lr_side"].isna()
    if unresolved.any():
        tr["lr_side"] = tr["lr_side"].ffill().bfill()
        tr.loc[unresolved, "label_source"] = "fill"

    tr["trade_sign"] = tr["lr_side"].astype(int)
    tr["signed_size"] = tr["size"] * tr["trade_sign"]
    tr["dollar_volume"] = tr["price"] * tr["size"]

    return tr


def build_bucketed_event_counts(trades_labeled: pd.DataFrame, bucket: str = "1s") -> pd.DataFrame:
    tr = trades_labeled.copy().sort_values("ts").set_index("ts")

    buy = tr.loc[tr["trade_sign"] == 1, "trade_sign"].resample(bucket).count()
    sell = tr.loc[tr["trade_sign"] == -1, "trade_sign"].resample(bucket).count()

    idx = buy.index.union(sell.index).sort_values()
    out = pd.DataFrame(index=idx)
    out["buy_events"] = buy.reindex(idx, fill_value=0)
    out["sell_events"] = sell.reindex(idx, fill_value=0)
    out = out.reset_index().rename(columns={"index": "ts"})
    return out


def simulate_discrete_hawkes(params: np.ndarray, buy_counts: np.ndarray, sell_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_b, mu_s, a_bb, a_bs, a_ss, a_sb, beta = params
    n = len(buy_counts)
    lam_b = np.zeros(n)
    lam_s = np.zeros(n)

    lam_b[0] = max(mu_b, 1e-8)
    lam_s[0] = max(mu_s, 1e-8)

    for t in range(1, n):
        lam_b[t] = mu_b + beta * lam_b[t - 1] + a_bb * buy_counts[t - 1] + a_bs * sell_counts[t - 1]
        lam_s[t] = mu_s + beta * lam_s[t - 1] + a_ss * sell_counts[t - 1] + a_sb * buy_counts[t - 1]
        lam_b[t] = max(lam_b[t], 1e-8)
        lam_s[t] = max(lam_s[t], 1e-8)

    return lam_b, lam_s


def poisson_negloglik(params: np.ndarray, buy_counts: np.ndarray, sell_counts: np.ndarray) -> float:
    lam_b, lam_s = simulate_discrete_hawkes(params, buy_counts, sell_counts)

    eps = 1e-12
    ll_buy = np.sum(buy_counts * np.log(lam_b + eps) - lam_b)
    ll_sell = np.sum(sell_counts * np.log(lam_s + eps) - lam_s)

    penalty = 0.0
    beta = params[-1]
    if beta >= 0.999:
        penalty += 1e6 * (beta - 0.999 + 1e-6)
    if beta < 0:
        penalty += 1e6 * abs(beta)

    excitation_sum = params[2] + params[3] + params[4] + params[5]
    if excitation_sum > 5.0:
        penalty += 1e3 * (excitation_sum - 5.0)

    return -(ll_buy + ll_sell) + penalty


def fit_discrete_hawkes(counts_df: pd.DataFrame) -> Dict[str, Any]:
    yb = counts_df["buy_events"].to_numpy(dtype=float)
    ys = counts_df["sell_events"].to_numpy(dtype=float)

    mu_b0 = max(yb.mean() * 0.5, 1e-3)
    mu_s0 = max(ys.mean() * 0.5, 1e-3)
    x0 = np.array([mu_b0, mu_s0, 0.10, 0.03, 0.10, 0.03, 0.60], dtype=float)

    bounds = [
        (1e-6, 50.0),
        (1e-6, 50.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (0.0, 5.0),
        (0.0, 0.999),
    ]

    res = minimize(
        poisson_negloglik,
        x0=x0,
        args=(yb, ys),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )

    params = res.x
    lam_b, lam_s = simulate_discrete_hawkes(params, yb, ys)

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "fun": float(res.fun),
        "params": {
            "mu_buy": float(params[0]),
            "mu_sell": float(params[1]),
            "alpha_bb": float(params[2]),
            "alpha_bs": float(params[3]),
            "alpha_ss": float(params[4]),
            "alpha_sb": float(params[5]),
            "beta": float(params[6]),
        },
        "lambda_buy_hawkes": lam_b,
        "lambda_sell_hawkes": lam_s,
    }


def estimate_baseline_intensities(counts_df: pd.DataFrame) -> pd.DataFrame:
    out = counts_df.copy()
    out["lambda_buy_poisson"] = out["buy_events"].astype(float)
    out["lambda_sell_poisson"] = out["sell_events"].astype(float)

    out["lambda_buy_ewm"] = out["lambda_buy_poisson"].ewm(span=60, adjust=False).mean()
    out["lambda_sell_ewm"] = out["lambda_sell_poisson"].ewm(span=60, adjust=False).mean()
    out["lambda_total_ewm"] = out["lambda_buy_ewm"] + out["lambda_sell_ewm"]

    denom = out["lambda_buy_ewm"] + out["lambda_sell_ewm"]
    out["order_flow_imbalance"] = np.where(
        denom > 0,
        (out["lambda_buy_ewm"] - out["lambda_sell_ewm"]) / denom,
        0.0,
    )
    return out


def build_hawkes_intensity_df(counts_df: pd.DataFrame, hawkes_fit: Dict[str, Any]) -> pd.DataFrame:
    out = counts_df[["ts", "buy_events", "sell_events"]].copy()

    if not hawkes_fit.get("success", False):
        out["lambda_buy_hawkes"] = np.nan
        out["lambda_sell_hawkes"] = np.nan
        out["hawkes_ofi"] = np.nan
        return out

    out["lambda_buy_hawkes"] = hawkes_fit["lambda_buy_hawkes"]
    out["lambda_sell_hawkes"] = hawkes_fit["lambda_sell_hawkes"]

    denom = out["lambda_buy_hawkes"] + out["lambda_sell_hawkes"]
    out["hawkes_ofi"] = np.where(
        denom > 0,
        (out["lambda_buy_hawkes"] - out["lambda_sell_hawkes"]) / denom,
        0.0,
    )
    return out


def attach_intensities_to_features(features: pd.DataFrame, baseline_int: pd.DataFrame, hawkes_int: pd.DataFrame) -> pd.DataFrame:
    f = features.copy()
    f["ts"] = pd.to_datetime(f["ts"], utc=True, errors="coerce")

    for col in ["lambda_buy_hawkes", "lambda_sell_hawkes", "hawkes_ofi"]:
        if col not in hawkes_int.columns:
            hawkes_int[col] = np.nan

    out = f.merge(
        baseline_int[[
            "ts", "buy_events", "sell_events",
            "lambda_buy_poisson", "lambda_sell_poisson",
            "lambda_buy_ewm", "lambda_sell_ewm",
            "lambda_total_ewm", "order_flow_imbalance"
        ]],
        on="ts", how="left"
    ).merge(
        hawkes_int[["ts", "lambda_buy_hawkes", "lambda_sell_hawkes", "hawkes_ofi"]],
        on="ts", how="left"
    )

    for c in [
        "buy_events", "sell_events",
        "lambda_buy_poisson", "lambda_sell_poisson",
        "lambda_buy_ewm", "lambda_sell_ewm",
        "lambda_total_ewm", "order_flow_imbalance"
    ]:
        out[c] = out[c].fillna(0.0)

    return out


def build_summary(trades_labeled: pd.DataFrame, baseline_int: pd.DataFrame, hawkes_fit: Dict[str, Any]) -> Dict[str, Any]:
    n = len(trades_labeled)
    buy_n = int((trades_labeled["trade_sign"] == 1).sum())
    sell_n = int((trades_labeled["trade_sign"] == -1).sum())

    summary = {
        "n_trades": int(n),
        "n_buy": buy_n,
        "n_sell": sell_n,
        "buy_fraction": float(buy_n / n) if n else None,
        "sell_fraction": float(sell_n / n) if n else None,
        "label_sources": {str(k): int(v) for k, v in trades_labeled["label_source"].value_counts().items()}
        if "label_source" in trades_labeled else None,
        "avg_lambda_buy_ewm": float(baseline_int["lambda_buy_ewm"].mean()),
        "avg_lambda_sell_ewm": float(baseline_int["lambda_sell_ewm"].mean()),
        "hawkes_fit": {
            "success": bool(hawkes_fit.get("success", False)),
            "message": hawkes_fit.get("message"),
            "fun": hawkes_fit.get("fun"),
            "params": hawkes_fit.get("params"),
        },
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Step 2: trade-side labeling and discrete Hawkes intensity estimation")
    parser.add_argument("--features", required=True, help="Step 1 feature table parquet/csv")
    parser.add_argument("--trades", required=True, help="Raw trades parquet/csv")
    parser.add_argument("--outdir", default="output/step2_hawkes", help="Output directory")
    parser.add_argument("--bucket", default="1s", help="Bucket size, e.g. 1s")
    parser.add_argument("--lee-ready-lag-ms", type=int, default=0, help="Optional Lee-Ready quote lag in ms")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading Step 1 feature table...")
    features = prepare_feature_table(load_table(args.features))
    print(f"Feature rows: {len(features):,}")

    print("Loading raw trades...")
    trades_raw = prepare_trades_raw(load_table(args.trades))
    print(f"Trade rows: {len(trades_raw):,}")

    print("Running Lee-Ready style trade classification...")
    trades_labeled = lee_ready_label(trades_raw, features, lag_ms=args.lee_ready_lag_ms)
    labeled_path = outdir / "trades_labeled.parquet"
    trades_labeled.to_parquet(labeled_path, index=False)

    print("Building bucketed event counts...")
    counts_df = build_bucketed_event_counts(trades_labeled, bucket=args.bucket)
    counts_path = outdir / f"event_counts_{args.bucket}.parquet"
    counts_df.to_parquet(counts_path, index=False)

    print("Estimating baseline intensities...")
    baseline_int = estimate_baseline_intensities(counts_df)
    baseline_path = outdir / f"baseline_intensities_{args.bucket}.parquet"
    baseline_int.to_parquet(baseline_path, index=False)

    print("Fitting discrete bivariate Hawkes model...")
    hawkes_fit = fit_discrete_hawkes(counts_df)

    print("Building Hawkes intensity time series...")
    hawkes_int = build_hawkes_intensity_df(counts_df, hawkes_fit)
    hawkes_path = outdir / f"hawkes_intensities_{args.bucket}.parquet"
    hawkes_int.to_parquet(hawkes_path, index=False)

    print("Attaching intensities to Step 1 feature table...")
    feature_step2 = attach_intensities_to_features(features, baseline_int, hawkes_int)
    feature_step2_path = outdir / f"feature_table_step2_{args.bucket}.parquet"
    feature_step2_csv = outdir / f"feature_table_step2_{args.bucket}.csv"
    feature_step2.to_parquet(feature_step2_path, index=False)
    feature_step2.to_csv(feature_step2_csv, index=False)

    summary = build_summary(trades_labeled, baseline_int, hawkes_fit)
    summary_path = outdir / "step2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved files:")
    print(" -", labeled_path)
    print(" -", counts_path)
    print(" -", baseline_path)
    print(" -", hawkes_path)
    print(" -", feature_step2_path)
    print(" -", feature_step2_csv)
    print(" -", summary_path)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()