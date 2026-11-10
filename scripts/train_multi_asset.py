"""
Multi-Asset ML Training Pipeline: Ethereum & Gold (Apex v7.2)
=============================================================
Trains the 24-model ensemble (4 Regimes x 2 Directions x 3 Bootstrap Bags)
for Ethereum (ETHUSDT) and Gold (PAXGUSDT / XAUUSD).
"""

import sys
import os
import asyncio
import time
import json
import argparse
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import requests
import pandas as pd
import numpy as np

from src.models.training.trainer import UnifiedTrainer
from src.models.xgboost_model import ApexXGBoostModel

MODEL_DIR = Path("data/models")


def fetch_historical_5m(symbol: str, pages: int = 8, page_size: int = 1000) -> pd.DataFrame:
    """Fetch multi-page 5m OHLCV data from Binance Futures or Spot."""
    print(f"\n[DATA] Fetching {pages} x {page_size} 5m candles for {symbol}...")
    all_candles = []
    end_time = None

    # Use Futures for ETHUSDT, Spot for PAXGUSDT (Gold)
    base_url = "https://fapi.binance.com/fapi/v1/klines" if symbol.upper() == "ETHUSDT" else "https://api.binance.com/api/v3/klines"

    for i in range(pages):
        params = {"symbol": symbol.upper(), "interval": "5m", "limit": page_size}
        if end_time:
            params["endTime"] = end_time

        try:
            res = requests.get(base_url, params=params, timeout=12)
            batch = res.json()
            if not isinstance(batch, list) or len(batch) == 0:
                break
            all_candles = batch + all_candles
            end_time = batch[0][0] - 1
            earliest_str = pd.Timestamp(batch[0][0], unit="ms").strftime("%Y-%m-%d %H:%M")
            print(f"  Page {i+1}: {len(batch)} candles (earliest: {earliest_str})")
            time.sleep(0.2)
        except Exception as e:
            print(f"  Page {i+1} fetch error: {e}")
            break

    cols = ["timestamp", "open", "high", "low", "close", "volume", "ct", "qv", "t", "tb", "tbq", "i"]
    df = pd.DataFrame(all_candles, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    print(f"  [OK] Successfully loaded {len(df)} candles for {symbol}")
    return df


def verify_models(symbol_key: str) -> int:
    """Verify how many models exist for the given symbol."""
    print(f"\n[VERIFY] Checking trained models for {symbol_key}...")
    count = 0
    regimes = ["ranging_high_vol", "ranging_low_vol", "trending_low_vol", "trending_high_vol"]
    for reg in regimes:
        for d in ["LONG", "SHORT"]:
            for b in [1, 2, 3]:
                name = f"{symbol_key}_crypto_{reg}_{d}_bag{b}"
                m_file = MODEL_DIR / f"{name}.json"
                meta_file = MODEL_DIR / f"{name}_metadata.json"
                if m_file.exists() and meta_file.exists():
                    count += 1
    print(f"  Found {count} / 24 models on disk for {symbol_key}")
    return count


async def train_asset(asset_key: str, binance_symbol: str):
    print("=" * 65)
    print(f"  TRAINING ML ENSEMBLE: {asset_key.upper()} ({binance_symbol})")
    print("=" * 65)

    # 1. Fetch 50,000 candles (~173 days of 5m data)
    df = fetch_historical_5m(binance_symbol, pages=50, page_size=1000)
    if len(df) < 500:
        print(f"ERROR: Insufficient data for {binance_symbol}")
        return
    try:
        hist_dir = root_dir / "data" / "historical" / "crypto"
        hist_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(hist_dir / f"{asset_key}_5m.parquet", index=False)
    except Exception:
        pass

    # 2. Run UnifiedTrainer
    trainer = UnifiedTrainer("crypto", asset_key)
    print(f"\nRunning UnifiedTrainer for {asset_key}...")
    start_t = time.time()
    result = await trainer.run(data=df, save_models=True)
    elapsed = time.time() - start_t
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")

    # 3. Verify
    verify_models(asset_key)


async def main():
    parser = argparse.ArgumentParser(description="Train ML ensembles for ETH and Gold")
    parser.add_argument("--asset", choices=["eth", "gold", "all"], default="all", help="Which asset to train")
    args = parser.parse_args()

    if args.asset in ("eth", "all"):
        await train_asset("ethusdt", "ETHUSDT")

    if args.asset in ("gold", "all"):
        await train_asset("paxgusdt", "PAXGUSDT")

    print("\n" + "=" * 65)
    print("  MULTI-ASSET TRAINING FINISHED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())

