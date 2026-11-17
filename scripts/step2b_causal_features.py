"""Step 2b: a causal copy of the step 2 feature table for the engine, plus the trade file the
replay fill model needs. The step 2 outputs themselves are left unchanged.

What changes relative to output/step2_hawkes/feature_table_step2_1s.csv:

- lambda_buy_hawkes, lambda_sell_hawkes, hawkes_ofi are walk-forward and one-step-ahead.
  The day is cut into 30-minute blocks; block j uses parameters fitted on blocks 0..j-1 only,
  and row t carries lambda_{t+1|t} (counts up to t, as the EWMA does). Block 0 has no past to
  fit on, so its Hawkes columns are empty and the engine falls back to the EWMA there, in both
  modes alike.
- trade_imbalance, buy_volume, sell_volume, signed_volume are rebuilt from the corrected trade
  labels in trades_labeled.parquet (step 1's were built from the legacy labels that lost most
  sells).

Also writes trades_by_bucket.csv (ts bucket label, price, size, sign) for --fill-model replay.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hawkes_diagnostics as hd  # noqa: E402

STEP2 = Path("output/step2_hawkes")
OUT = Path("output/step2b_causal")
BLOCK_ROWS = 1800


def walk_forward_hawkes(buy: np.ndarray, sell: np.ndarray, block_rows: int = BLOCK_ROWS) -> tuple[pd.DataFrame, list[dict]]:
    n = len(buy)
    lam_b = np.full(n, np.nan)
    lam_s = np.full(n, np.nan)
    fits = []
    for start in range(block_rows, n, block_rows):
        end = min(start + block_rows, n)
        fit = hd.fit_discrete(buy[:start], sell[:start])
        p = fit["x"]
        lb_t, ls_t = hd.discrete_intensities(p, buy[:end], sell[:end])
        mu_b, mu_s, a_bb, a_bs, a_ss, a_sb, beta = p
        nb = mu_b + beta * lb_t + a_bb * buy[:end] + a_bs * sell[:end]
        ns = mu_s + beta * ls_t + a_ss * sell[:end] + a_sb * buy[:end]
        lam_b[start:end] = np.maximum(nb[start:end], 1e-8)
        lam_s[start:end] = np.maximum(ns[start:end], 1e-8)
        fits.append({"block_start_row": start, "fitted_on_rows": [0, start], **fit["params"],
                     "spectral_radius_branching": fit["spectral_radius_branching"]})
    denom = lam_b + lam_s
    ofi = np.where(denom > 0, (lam_b - lam_s) / denom, np.nan)
    return pd.DataFrame({"lambda_buy_hawkes": lam_b, "lambda_sell_hawkes": lam_s, "hawkes_ofi": ofi}), fits


def bucket_trade_volumes(trades: pd.DataFrame, bucket_ts: pd.Series) -> pd.DataFrame:
    t = trades.assign(bucket=trades["ts"].dt.floor("1s"))
    buy = t[t["trade_sign"] == 1].groupby("bucket")["size"].sum()
    sell = t[t["trade_sign"] == -1].groupby("bucket")["size"].sum()
    out = pd.DataFrame({"buy_volume": buy, "sell_volume": sell}).reindex(bucket_ts).fillna(0.0)
    total = out["buy_volume"] + out["sell_volume"]
    out["signed_volume"] = out["buy_volume"] - out["sell_volume"]
    out["trade_imbalance"] = np.where(total > 0, out["signed_volume"] / total, 0.0)
    return out.reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(STEP2 / "feature_table_step2_1s.csv")
    trades = pd.read_parquet(STEP2 / "trades_labeled.parquet")
    ts = pd.to_datetime(feats["ts"], utc=True)

    hawkes, fits = walk_forward_hawkes(feats["buy_events"].to_numpy(float), feats["sell_events"].to_numpy(float))
    for col in hawkes.columns:
        feats[col] = hawkes[col].to_numpy()

    vols = bucket_trade_volumes(trades, ts)
    for col in vols.columns:
        feats[col] = vols[col].to_numpy()

    feats.to_csv(OUT / "feature_table_causal_1s.csv", index=False)

    tb = pd.DataFrame({
        "ts": trades["ts"].dt.floor("1s").astype(str),
        "price": trades["price"], "size": trades["size"], "sign": trades["trade_sign"],
    })
    tb.to_csv(OUT / "trades_by_bucket.csv", index=False)

    (OUT / "walk_forward_hawkes_fits.json").write_text(json.dumps(
        {"block_rows": BLOCK_ROWS, "feature": "one-step-ahead lambda_{t+1|t}", "fits": fits}, indent=2))
    print(f"Wrote {OUT}/feature_table_causal_1s.csv ({len(feats)} rows, {len(fits)} refits), "
          f"trades_by_bucket.csv ({len(tb)} trades)")


if __name__ == "__main__":
    main()
