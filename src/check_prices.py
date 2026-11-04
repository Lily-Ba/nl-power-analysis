from pathlib import Path
import pandas as pd

files = sorted(Path("data/raw").glob("prices_*.parquet"))
prices = pd.concat(pd.read_parquet(f) for f in files)["prices"].sort_index()
prices = prices[~prices.index.duplicated()]

# Hourly before Oct 2025, 15-minute after: average everything to hourly
hourly = prices.resample("h").mean()

summary = hourly.groupby(hourly.index.year).agg(
    hours="count",
    avg_price="mean",
    min_price="min",
    max_price="max",
    negative_hours=lambda s: (s < 0).sum(),
)
print(summary.round(1))
