"""Combine raw ENTSO-E files into one clean hourly table for modelling."""
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/hourly.parquet")
TZ = "Europe/Amsterdam"
START, END = "2019-01-01", "2026-01-01"  # END is exclusive


def load_hourly(name):
    """Read all yearly files for one dataset and average to hourly."""
    files = sorted(RAW.glob(f"{name}_[0-9]*.parquet"))
    df = pd.concat(pd.read_parquet(f) for f in files).sort_index()
    df = df[~df.index.duplicated()]
    df.index = df.index.tz_convert(TZ)
    return df.resample("h").mean()


# --- 1. Load each dataset ---------------------------------------------------
prices = load_hourly("prices").iloc[:, 0].rename("price")

load = load_hourly("load").iloc[:, 0].rename("load_actual")
load_fc = load_hourly("load_forecast").iloc[:, 0].rename("load_fc")

fc = load_hourly("wind_solar_forecast").rename(columns={
    "Solar": "solar_fc", "Wind Offshore": "wind_off_fc", "Wind Onshore": "wind_on_fc",
})

gen = load_hourly("generation")
gen = gen[[c for c in gen.columns if c.endswith("Actual Aggregated")]]
gen.columns = [
    "gen_" + c.replace("_Actual Aggregated", "").lower().replace(" ", "_").replace("/", "_")
    for c in gen.columns
]

# --- 2. Join on a complete hourly index ---------------------------------------
index = pd.date_range(START, END, freq="h", tz=TZ, inclusive="left")
df = pd.concat([prices, load, load_fc, fc, gen], axis=1).reindex(index)
df.index.name = "timestamp"

missing_before = df.isna().sum()
df = df.interpolate(method="time", limit=3, limit_area="inside")  # fill short gaps only
missing_after = df.isna().sum()

# --- 3. Derived columns -----------------------------------------------------
df["wind_fc"] = df["wind_off_fc"] + df["wind_on_fc"]
df["residual_load_fc"] = df["load_fc"] - df["solar_fc"] - df["wind_fc"]
df["gen_total"] = df.filter(like="gen_").sum(axis=1, min_count=1)
if "gen_solar" in df:
    df["solar_share"] = df["gen_solar"] / df["gen_total"]
df["is_negative"] = (df["price"] < 0).astype(int)

# --- 4. Time features -------------------------------------------------------
nl_holidays = holidays.Netherlands(years=range(2019, 2026))
df["date"] = df.index.date
df["year"] = df.index.year
df["month"] = df.index.month
df["hour"] = df.index.hour
df["weekday"] = df.index.day_of_week  # 0 = Monday
df["is_weekend"] = (df["weekday"] >= 5).astype(int)
df["is_holiday"] = np.array([d in nl_holidays for d in df["date"]], dtype=int)
df["season"] = df["month"].map({12: "Winter", 1: "Winter", 2: "Winter",
                                3: "Spring", 4: "Spring", 5: "Spring",
                                6: "Summer", 7: "Summer", 8: "Summer",
                                9: "Autumn", 10: "Autumn", 11: "Autumn"})
df["daily_spread"] = df.groupby("date")["price"].transform(lambda s: s.max() - s.min())

# --- 5. Save and report -----------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT)

print(f"Saved {OUT}: {len(df):,} rows x {df.shape[1]} columns")
print(f"Range: {df.index.min()} -> {df.index.max()}\n")
report = pd.DataFrame({"missing_before": missing_before, "missing_after": missing_after})
print("Missing values in key columns (after filling gaps of up to 3 hours):")
print(report.loc[["price", "load_actual", "load_fc", "solar_fc", "wind_off_fc", "wind_on_fc"]])
print("\nGeneration columns:", [c for c in df.columns if c.startswith("gen_")])
