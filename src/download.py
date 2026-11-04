"""Download Dutch ENTSO-E data year by year and cache it as parquet."""
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()
client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)
TZ = "Europe/Amsterdam"
YEARS = range(2019, 2026)

QUERIES = {
    "prices": client.query_day_ahead_prices,
    "load": client.query_load,
    "generation": client.query_generation,
    "wind_solar_forecast": client.query_wind_and_solar_forecast,
}


def fetch(name, func, year):
    out = RAW / f"{name}_{year}.parquet"
    if out.exists():
        print(f"skip {out.name} (cached)")
        return

    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end = pd.Timestamp(f"{year + 1}-01-01", tz=TZ)
    for attempt in range(1, 4):
        try:
            data = func("NL", start=start, end=end)
            break
        except Exception as e:
            print(f"  {name} {year} attempt {attempt} failed: {e}")
            time.sleep(10 * attempt)
    else:
        print(f"GAVE UP on {name} {year}")
        return

    if isinstance(data, pd.Series):
        data = data.to_frame(name)
    if isinstance(data.columns, pd.MultiIndex):  # e.g. ('Solar', 'Actual Aggregated')
        data.columns = ["_".join(map(str, c)).strip("_") for c in data.columns]

    data.to_parquet(out)
    print(f"saved {out.name}: {len(data):,} rows, {data.index.min()} -> {data.index.max()}")


if __name__ == "__main__":
    for year in YEARS:
        for name, func in QUERIES.items():
            fetch(name, func, year)
