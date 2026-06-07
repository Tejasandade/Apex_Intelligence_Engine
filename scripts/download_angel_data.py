"""
Apex Intelligence Engine V5 — Download Angel One Data
======================================================
Fetches historical 1-minute candle data from Angel One SmartAPI.
Since Angel restricts 1m data fetches to 30 days per request, this script
iterates backwards to download up to 90 days of data and merges it into a parquet file.

Usage:
    python -m scripts.download_angel_data --symbol BANKNIFTY --days 90
"""

import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from src.core.config import DATA_DIR
from src.core.logging import get_logger
from src.data.providers.angel_api import AngelApiProvider

logger = get_logger("apex.scripts.download_angel_data")


def main():
    parser = argparse.ArgumentParser(description="Download historical data from Angel One")
    parser.add_argument("--symbol", type=str, default="BANKNIFTY", help="Symbol to download (e.g., BANKNIFTY, NIFTY)")
    parser.add_argument("--days", type=int, default=30, help="Number of days to download")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    days = args.days
    
    provider = AngelApiProvider()
    if not provider.connect():
        print(f"Failed to connect to Angel One API. Check credentials in .env")
        return

    print(f"\n[DOWNLOADER] Fetching {days} days of {symbol} 1m data...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Angel One restricts 1-minute data to 30 days. We'll fetch in 10-day chunks just to be safe.
    chunk_size = 10
    
    current_end = end_date
    all_data = []

    while current_end > start_date:
        current_start = max(current_end - timedelta(days=chunk_size), start_date)
        
        print(f"  Fetching: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        df = provider.fetch_historical_candles(
            symbol=symbol,
            interval="ONE_MINUTE",
            limit=9000,  # roughly 10 days of 1m bars (375 bars/day)
            from_date=current_start,
            to_date=current_end
        )
        
        if not df.empty:
            all_data.append(df)
            print(f"  -> Downloaded {len(df)} candles")
        else:
            print(f"  -> No data found for this period")
            
        current_end = current_start
        # Respect rate limits (Angel allows 3 req/sec but play it safe)
        time.sleep(0.5)

    if not all_data:
        print("\n[DOWNLOADER] Failed to download any data.")
        return

    # Merge and clean
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df.sort_values(by="timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    
    print(f"\n[DOWNLOADER] Total candles downloaded: {len(full_df)}")
    
    # Save to parquet
    out_dir = DATA_DIR / "historical" / "india_equity"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"{symbol.lower()}_1m.parquet"
    full_df.to_parquet(out_file, index=False)
    
    print(f"[DOWNLOADER] Saved to {out_file}")

if __name__ == "__main__":
    main()
