import asyncio
import sys
from pathlib import Path
import pandas as pd
import time
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.live_store import LiveFeatureStore
from src.core.config import DATA_DIR
from src.models.xgboost_model import ApexXGBoostModel
from src.features.store import FeatureStore

async def main():
    print("Loading historical data for warmup...")
    cache_path = DATA_DIR / "historical" / "crypto" / "btcusdt.parquet"
    if not cache_path.exists():
        print(f"Error: {cache_path} not found.")
        return
    
    df = pd.read_parquet(cache_path)
    warmup_df = df.iloc[:-10]
    test_df = df.iloc[-10:]
    
    print("Initializing LiveFeatureStore...")
    live_store = LiveFeatureStore(market_type="crypto", buffer_size=200)
    live_store.warm_up(warmup_df)
    print(f"Is warmed up: {live_store.is_warmed_up}")
    
    print("Loading models...")
    models = {}
    feature_store = FeatureStore("crypto")
    for regime in ["trending", "ranging"]:
        model_name = f"btcusdt_crypto_{regime}"
        model_path = DATA_DIR / "models" / f"{model_name}.json"
        if model_path.exists():
            model = ApexXGBoostModel(name=model_name, feature_columns=feature_store.feature_columns)
            model.load(DATA_DIR / "models")
            models[regime] = model
            print(f"Loaded {model_name}")
    
    print("\nProcessing 5 distinct candles...")
    for i, (_, row) in enumerate(test_df.head(5).iterrows()):
        candle = {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        features = live_store.update(candle)
        if features is None:
            print(f"Candle {i}: Features is None")
            continue
            
        print(f"\n--- Candle {i} (Close: {candle['close']}) ---")
        # Print a couple feature values to see if they change
        print(f"RSI: {features.iloc[0]['RSI']:.4f}, ATR: {features.iloc[0]['ATR']:.4f}")
        
        for regime, model in models.items():
            prob = model.predict(features)
            print(f"Model {regime} predicted: {prob:.6f}")

if __name__ == "__main__":
    asyncio.run(main())
