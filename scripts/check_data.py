import pandas as pd
from pathlib import Path

data_path = Path("data/historical/crypto/btcusdt.parquet")
df = pd.read_parquet(data_path)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns[:15])}")

ts = df["timestamp"]
if ts.dtype == "int64" or ts.dtype == "float64":
    print(f"First timestamp: {pd.to_datetime(ts.iloc[0], unit='ms')}")
    print(f"Last timestamp: {pd.to_datetime(ts.iloc[-1], unit='ms')}")
else:
    print(f"First timestamp: {ts.iloc[0]}")
    print(f"Last timestamp: {ts.iloc[-1]}")
