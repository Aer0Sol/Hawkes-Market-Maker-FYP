import os
from pathlib import Path

import databento as db
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("DATABENTO_API_KEY")
if not API_KEY:
    raise ValueError("Missing DATABENTO_API_KEY in environment or .env file")

# Replace this with the exact dataset code shown in your Databento account
DATASET = "XNAS.ITCH"

SYMBOL = "AAPL"
START = "2024-06-03T13:30:00Z"   # 09:30 ET
END   = "2024-06-03T20:00:00Z"   # 16:00 ET

outdir = Path("data/databento")
outdir.mkdir(parents=True, exist_ok=True)

client = db.Historical(API_KEY)

def fetch_and_save(schema: str, filename: str):
    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=SYMBOL,
        stype_in="raw_symbol",
        schema=schema,
        start=START,
        end=END,
    )
    df = data.to_df()
    path = outdir / filename
    df.to_csv(path, index=False)
    print(f"{schema}: saved {len(df):,} rows to {path}")

fetch_and_save("mbp-10", "aapl_mbp-10_2024-06-03.csv")
fetch_and_save("trades", "aapl_trades_2024-06-03.csv")
