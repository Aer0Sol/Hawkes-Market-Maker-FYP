from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


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
    candidates = [
        "ts_event", "ts_recv", "timestamp", "ts", "time", "datetime"
    ]
    col = find_first_existing(df, candidates)
    if col is None:
        raise KeyError(f"Could not find timestamp column. Available columns: {list(df.columns)}")

    s = df[col]

    if np.issubdtype(s.dtype, np.number):
        s_nonnull = s.dropna()
        if len(s_nonnull) == 0:
            raise ValueError("Timestamp column is empty.")
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


# Databento aggressor side: 'B' = buy, 'A' = sell, 'N' = unknown.
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


def prepare_mbp10(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = parse_timestamp(df)
    df = df.sort_values("ts").reset_index(drop=True)

    bid_px_candidates = ["bid_px_00", "bid_px_0", "bid_px1", "bid_px_1", "best_bid_px", "bid_px"]
    ask_px_candidates = ["ask_px_00", "ask_px_0", "ask_px1", "ask_px_1", "best_ask_px", "ask_px"]
    bid_sz_candidates = ["bid_sz_00", "bid_sz_0", "bid_sz1", "bid_sz_1", "best_bid_sz", "bid_sz"]
    ask_sz_candidates = ["ask_sz_00", "ask_sz_0", "ask_sz1", "ask_sz_1", "best_ask_sz", "ask_sz"]

    bid_px_col = find_first_existing(df, bid_px_candidates)
    ask_px_col = find_first_existing(df, ask_px_candidates)
    bid_sz_col = find_first_existing(df, bid_sz_candidates)
    ask_sz_col = find_first_existing(df, ask_sz_candidates)

    if not all([bid_px_col, ask_px_col, bid_sz_col, ask_sz_col]):
        raise KeyError(
            "Could not find top-of-book columns.\n"
            f"Columns available:\n{list(df.columns)}"
        )

    df["bid_px_1"] = normalize_price(df[bid_px_col])
    df["ask_px_1"] = normalize_price(df[ask_px_col])
    df["bid_sz_1"] = pd.to_numeric(df[bid_sz_col], errors="coerce")
    df["ask_sz_1"] = pd.to_numeric(df[ask_sz_col], errors="coerce")

    depth_bid_cols = [c for c in df.columns if c.startswith("bid_sz_")]
    depth_ask_cols = [c for c in df.columns if c.startswith("ask_sz_")]

    if len(depth_bid_cols) == 0:
        depth_bid_cols = [bid_sz_col]
    if len(depth_ask_cols) == 0:
        depth_ask_cols = [ask_sz_col]

    df["depth_bid_10"] = df[depth_bid_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    df["depth_ask_10"] = df[depth_ask_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    df["midprice"] = (df["bid_px_1"] + df["ask_px_1"]) / 2.0
    df["spread"] = df["ask_px_1"] - df["bid_px_1"]
    df["spread_bps"] = (df["spread"] / df["midprice"]) * 10_000.0

    denom = (df["bid_sz_1"] + df["ask_sz_1"]).replace(0, np.nan)
    df["imbalance_l1"] = (df["bid_sz_1"] - df["ask_sz_1"]) / denom
    df["microprice"] = (
        df["ask_px_1"] * df["bid_sz_1"] + df["bid_px_1"] * df["ask_sz_1"]
    ) / denom

    denom10 = (df["depth_bid_10"] + df["depth_ask_10"]).replace(0, np.nan)
    df["imbalance_l10"] = (df["depth_bid_10"] - df["depth_ask_10"]) / denom10

    df["log_mid"] = np.log(df["midprice"].replace(0, np.nan))
    df["log_ret_event"] = df["log_mid"].diff()
    df["fwd_ret_1_event"] = df["log_mid"].shift(-1) - df["log_mid"]
    df["fwd_ret_10_event"] = df["log_mid"].shift(-10) - df["log_mid"]

    keep_cols = [
        "ts",
        "bid_px_1", "ask_px_1", "bid_sz_1", "ask_sz_1",
        "depth_bid_10", "depth_ask_10",
        "midprice", "microprice",
        "spread", "spread_bps",
        "imbalance_l1", "imbalance_l10",
        "log_mid", "log_ret_event", "fwd_ret_1_event", "fwd_ret_10_event",
    ]
    return df[keep_cols].dropna(subset=["ts", "bid_px_1", "ask_px_1"]).reset_index(drop=True)


def prepare_trades(df: pd.DataFrame, mbp_state: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = parse_timestamp(df)
    df = df.sort_values("ts").reset_index(drop=True)

    price_col = find_first_existing(df, ["price", "px"])
    size_col = find_first_existing(df, ["size", "qty", "quantity"])
    side_col = find_first_existing(df, ["side", "aggressor_side", "action", "is_aggressor"])

    if price_col is None or size_col is None:
        raise KeyError(f"Could not find trade price/size columns. Available columns: {list(df.columns)}")

    df["price"] = normalize_price(df[price_col])
    df["size"] = pd.to_numeric(df[size_col], errors="coerce")

    if side_col is not None:
        df["trade_sign"] = map_aggressor_side(df[side_col])
    else:
        df["trade_sign"] = np.nan

    if df["trade_sign"].isna().any() and mbp_state is not None:
        ref = mbp_state[["ts", "midprice"]].sort_values("ts").reset_index(drop=True)
        df = pd.merge_asof(
            df.sort_values("ts"),
            ref,
            on="ts",
            direction="backward"
        )
        inferred = np.where(df["price"] >= df["midprice"], 1, -1)
        df["trade_sign"] = df["trade_sign"].fillna(pd.Series(inferred, index=df.index))

    df["signed_size"] = df["size"] * df["trade_sign"].fillna(0)
    df["dollar_volume"] = df["price"] * df["size"]

    keep_cols = ["ts", "price", "size", "trade_sign", "signed_size", "dollar_volume"]
    return df[keep_cols].dropna(subset=["ts", "price", "size"]).reset_index(drop=True)


def resample_mbp_features(mbp: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    x = mbp.copy().set_index("ts").sort_index()

    out = pd.DataFrame(index=x.resample(freq).last().index)
    out["bid_px_1"] = x["bid_px_1"].resample(freq).last()
    out["ask_px_1"] = x["ask_px_1"].resample(freq).last()
    out["bid_sz_1"] = x["bid_sz_1"].resample(freq).last()
    out["ask_sz_1"] = x["ask_sz_1"].resample(freq).last()
    out["depth_bid_10"] = x["depth_bid_10"].resample(freq).last()
    out["depth_ask_10"] = x["depth_ask_10"].resample(freq).last()
    out["midprice"] = x["midprice"].resample(freq).last()
    out["microprice"] = x["microprice"].resample(freq).last()
    out["spread"] = x["spread"].resample(freq).last()
    out["spread_bps"] = x["spread_bps"].resample(freq).last()
    out["imbalance_l1"] = x["imbalance_l1"].resample(freq).last()
    out["imbalance_l10"] = x["imbalance_l10"].resample(freq).last()

    out = out.ffill()

    out["log_mid"] = np.log(out["midprice"].replace(0, np.nan))
    out["ret_1"] = out["log_mid"].diff()
    out["rv_60"] = out["ret_1"].rolling(60).std()
    out["rv_300"] = out["ret_1"].rolling(300).std()
    out["mid_move"] = out["midprice"].diff()
    out["microprice_alpha"] = out["microprice"] - out["midprice"]

    return out.reset_index().rename(columns={"index": "ts"})


def resample_trade_features(trades: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    t = trades.copy().set_index("ts").sort_index()

    grp = t.resample(freq)
    out = pd.DataFrame(index=grp.size().index)
    out["trade_count"] = grp.size()
    out["trade_volume"] = grp["size"].sum()
    out["signed_volume"] = grp["signed_size"].sum()
    out["buy_count"] = grp.apply(lambda x: (x["trade_sign"] == 1).sum())
    out["sell_count"] = grp.apply(lambda x: (x["trade_sign"] == -1).sum())
    out["buy_volume"] = grp.apply(lambda x: x.loc[x["trade_sign"] == 1, "size"].sum())
    out["sell_volume"] = grp.apply(lambda x: x.loc[x["trade_sign"] == -1, "size"].sum())
    out["vwap"] = grp.apply(
        lambda x: np.nan if x["size"].sum() == 0 else (x["price"] * x["size"]).sum() / x["size"].sum()
    )
    out["last_trade_price"] = grp["price"].last()

    denom = out["buy_volume"] + out["sell_volume"]
    out["trade_imbalance"] = np.where(denom > 0, (out["buy_volume"] - out["sell_volume"]) / denom, 0.0)

    return out.reset_index().rename(columns={"index": "ts"})


def build_ohlcv(trades: pd.DataFrame, freq: str = "1s") -> pd.DataFrame:
    t = trades.copy().set_index("ts").sort_index()

    ohlc = t["price"].resample(freq).ohlc()
    vol = t["size"].resample(freq).sum().rename("volume")
    ntr = t["size"].resample(freq).count().rename("n_trades")
    vwap = t.resample(freq).apply(
        lambda x: np.nan if x["size"].sum() == 0 else (x["price"] * x["size"]).sum() / x["size"].sum()
    ).rename("vwap")

    bars = pd.concat([ohlc, vol, ntr, vwap], axis=1).reset_index()
    return bars


def merge_feature_table(mbp_feat: pd.DataFrame, trade_feat: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(mbp_feat, trade_feat, on="ts", how="left")
    trade_cols = [
        "trade_count", "trade_volume", "signed_volume",
        "buy_count", "sell_count", "buy_volume", "sell_volume",
        "vwap", "last_trade_price", "trade_imbalance"
    ]
    for c in trade_cols:
        if c in df.columns:
            if c in ["vwap", "last_trade_price"]:
                df[c] = df[c].ffill()
            else:
                df[c] = df[c].fillna(0)

    df["future_ret_1s"] = df["log_mid"].shift(-1) - df["log_mid"]
    df["future_ret_5s"] = df["log_mid"].shift(-5) - df["log_mid"]
    df["future_ret_10s"] = df["log_mid"].shift(-10) - df["log_mid"]

    return df


def main():
    parser = argparse.ArgumentParser(description="Build feature table from Databento MBP-10 + trades")
    parser.add_argument("--mbp", required=True, help="Path to MBP-10 parquet/csv")
    parser.add_argument("--trades", required=True, help="Path to trades parquet/csv")
    parser.add_argument("--outdir", default="output/step1_features", help="Output directory")
    parser.add_argument("--freq", default="1s", help="Resample frequency, e.g. 100ms, 1s")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading MBP-10...")
    mbp_raw = load_table(args.mbp)
    print(f"MBP-10 rows: {len(mbp_raw):,}")

    print("Preparing MBP-10 top-of-book features...")
    mbp_event = prepare_mbp10(mbp_raw)
    mbp_feat = resample_mbp_features(mbp_event, freq=args.freq)

    print("Loading trades...")
    trades_raw = load_table(args.trades)
    print(f"Trades rows: {len(trades_raw):,}")

    print("Preparing trades...")
    trades_clean = prepare_trades(trades_raw, mbp_state=mbp_event)
    trade_feat = resample_trade_features(trades_clean, freq=args.freq)

    print("Merging feature table...")
    features = merge_feature_table(mbp_feat, trade_feat)

    print("Building OHLCV bars...")
    bars = build_ohlcv(trades_clean, freq=args.freq)

    feature_path_parquet = outdir / f"feature_table_{args.freq}.parquet"
    feature_path_csv = outdir / f"feature_table_{args.freq}.csv"
    bars_path_parquet = outdir / f"ohlcv_{args.freq}.parquet"
    bars_path_csv = outdir / f"ohlcv_{args.freq}.csv"

    features.to_parquet(feature_path_parquet, index=False)
    features.to_csv(feature_path_csv, index=False)
    bars.to_parquet(bars_path_parquet, index=False)
    bars.to_csv(bars_path_csv, index=False)

    print(f"Saved feature table: {feature_path_parquet}")
    print(f"Saved feature table CSV: {feature_path_csv}")
    print(f"Saved OHLCV bars: {bars_path_parquet}")
    print(f"Saved OHLCV CSV: {bars_path_csv}")

    print("\nFeature columns:")
    print(features.columns.tolist())
    print("\nHead:")
    print(features.head())


if __name__ == "__main__":
    main()