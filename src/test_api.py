import os
import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient

load_dotenv()
client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])

start = pd.Timestamp("2024-05-01", tz="Europe/Amsterdam")
end = pd.Timestamp("2024-05-08", tz="Europe/Amsterdam")
prices = client.query_day_ahead_prices("NL", start=start, end=end)

print(prices.head(24))
print(f"Rows: {len(prices)}")
print(f"Negative-price hours this week: {(prices < 0).sum()}")
