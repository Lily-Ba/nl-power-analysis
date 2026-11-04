from pathlib import Path
import pandas as pd

raw = Path("data/raw")
print(f"Files: {len(list(raw.glob('*.parquet')))}\n")

for name in ["load", "generation", "wind_solar_forecast"]:
    files = sorted(raw.glob(f"{name}_*.parquet"))
    df = pd.read_parquet(files[-1])  # most recent year
    print(f"=== {name}: {len(files)} files, latest = {files[-1].name} ===")
    print(f"rows: {len(df):,}  |  time step: {df.index.to_series().diff().mode()[0]}")
    print("columns:", list(df.columns))
    print(df.head(3).round(0), "\n")
