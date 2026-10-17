# Raw data

Raw market data is licensed (Databento) and is **never committed**. Everything in
`data/` except this file is gitignored.

## What the pipeline expects

| File | Schema | Used by |
|---|---|---|
| `data/databento/aapl_mbp-10_2024-06-03.csv` | Databento `mbp-10` | `make features` (step 1) |
| `data/databento/aapl_trades_2024-06-03.csv` | Databento `trades` | `make features`, `make hawkes` (steps 1–2) |

## How to obtain it

The original request (from `scripts/download_script.py`):

| Field | Value |
|---|---|
| Dataset | `XNAS.ITCH` (Nasdaq TotalView-ITCH; confirm the code shown in your Databento account) |
| Symbol | `AAPL`, `stype_in="raw_symbol"` |
| Schemas | `mbp-10`, `trades` |
| Start | `2024-06-03T13:30:00Z` (09:30 ET) |
| End | `2024-06-03T20:00:00Z` (16:00 ET) |

```bash
cp .env.example .env              # then put your key in DATABENTO_API_KEY
uv pip install databento python-dotenv   # not part of the pinned core env
uv run python scripts/download_script.py
```

This costs Databento credits. **Ask the project owner before running it or
requesting any additional dates or symbols.**

## Without raw data

Steps 1–2 outputs derived from this one day are committed under
`output/step1_features/` and `output/step2_hawkes/`, so `make all` runs the
engine and analysis without raw data. `make features` and `make hawkes`
fail with a pointer to this file.
