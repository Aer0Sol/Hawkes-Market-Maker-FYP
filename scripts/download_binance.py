"""Download one day of Binance aggTrades (public, free, no API key) for the cross-asset
Hawkes validation in scripts/step6_binance_validation.py.

Source: Binance's public historical data dump (data.binance.vision), the same data the
exchange's own API serves, bundled as a daily CSV. No key, no rate limit beyond a plain HTTPS
GET. Not committed to the repo (data/ is gitignored) -- this script re-fetches it.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
OUTDIR = Path("data/binance")
HOLDOUT = Path(__file__).resolve().parents[1] / "config" / "holdout.json"


def holdout_dates(symbol: str) -> set[str]:
    if not HOLDOUT.exists():
        return set()
    return set(json.loads(HOLDOUT.read_text()).get("binance", {}).get(symbol, []))


def check_not_holdout(symbol: str, date: str, allow: bool) -> None:
    if date in holdout_dates(symbol) and not allow:
        raise SystemExit(f"{symbol} {date} is in the recorded holdout (config/holdout.json). It is evaluated once, "
                         "at the end, with --allow-holdout.")


def download(symbol: str, date: str) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fname = f"{symbol}-aggTrades-{date}"
    out_csv = OUTDIR / f"{fname}.csv"
    if out_csv.exists():
        print(f"Already have {out_csv}")
        return out_csv

    url = f"{BASE_URL}/{symbol}/{fname}.zip"
    print(f"Fetching {url} ...")
    with urlopen(url, timeout=120) as resp:  # noqa: S310 -- fixed, trusted Binance data host
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        inner = zf.namelist()[0]
        with zf.open(inner) as f:
            out_csv.write_bytes(f.read())
    print(f"Saved {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--date", default="2024-06-03", help="YYYY-MM-DD (UTC trading day)")
    ap.add_argument("--allow-holdout", action="store_true", help="only for the single final evaluation")
    args = ap.parse_args()
    check_not_holdout(args.symbol, args.date, args.allow_holdout)
    try:
        download(args.symbol, args.date)
    except Exception as e:  # noqa: BLE001 -- network/parsing failure, report and exit non-zero
        print(f"ERROR: could not download Binance data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
