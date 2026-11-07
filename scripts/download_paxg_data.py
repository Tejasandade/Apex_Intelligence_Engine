"""
Download historical 5m candles for PAXGUSDT from Binance API.
Saves to data/historical/crypto/paxgusdt.parquet and data/historical/crypto/paxgusdt.csv.
"""
import time
import requests
import pandas as pd
from pathlib import Path

def fetch_paxg_5m_candles(target_count: int = 10000):
    url = "https://api.binance.com/api/v3/klines"
    all_candles = []
    end_time = int(time.time() * 1000)
    batch_size = 1000

    print(f"Fetching {target_count} 5m candles for PAXGUSDT...")
    while len(all_candles) < target_count:
        params = {
            "symbol": "PAXGUSDT",
            "interval": "5m",
            "limit": batch_size,
            "endTime": end_time
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"Error fetching data: {resp.status_code} {resp.text}")
            break
        
        data = resp.json()
        if not data:
            print("No more data returned.")
            break

        all_candles = data + all_candles
        end_time = data[0][0] - 1  # Move before the earliest candle in this batch
        print(f"  Fetched {len(all_candles)} candles so far (earliest: {pd.to_datetime(end_time, unit='ms', utc=True)})...")
        time.sleep(0.3)

    # Deduplicate and sort
    seen_ts = set()
    cleaned = []
    for c in all_candles:
        ts = int(c[0])
        if ts not in seen_ts:
            seen_ts.add(ts)
            cleaned.append({
                "timestamp": ts,
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5])
            })

    cleaned.sort(key=lambda x: x["timestamp"])
    df = pd.DataFrame(cleaned)
    print(f"Total cleaned candles: {len(df)}")
    print(df.head(2))
    print(df.tail(2))

    out_dir = Path("data/historical/crypto")
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "paxgusdt.parquet"
    csv_path = out_dir / "paxgusdt.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved to {parquet_path} ({parquet_path.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Saved to {csv_path} ({csv_path.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    fetch_paxg_5m_candles(10000)

