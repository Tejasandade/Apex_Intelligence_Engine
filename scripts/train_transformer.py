"""
Apex Intelligence Engine V5 — Dedicated Transformer Trainer
============================================================
Trains the Time-Series Transformer architecture on historical data.
"""

import argparse
import asyncio
import gc
import sys
from pathlib import Path

# Ensure local imports work correctly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.core.logging import get_logger
from src.core.config import DATA_DIR
from src.features.store import FeatureStore
from src.features.labeling.triple_barrier import apply_triple_barrier_labels
from src.models.transformer_model import ApexTransformerModel

logger = get_logger("apex.scripts.train_transformer")


def train_transformer(market: str, symbol: str):
    logger.info("starting_transformer_training", market=market, symbol=symbol)
    
    # 1. Load Data
    data_dir = DATA_DIR / "historical" / market
    csv_path = data_dir / f"{symbol}.csv"
    parquet_path = data_dir / f"{symbol}.parquet"
    
    df = None
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
    if df is None:
        logger.error("training_data_not_found", symbol=symbol)
        return

    # ── MEMORY CONSTRAINT FIX ──
    # Deep Learning models are RAM intensive, capping at 10k rows
    if len(df) > 10000:
        df = df.iloc[-10000:].copy()
        
    logger.info("data_loaded", rows=len(df))

    # 2. Compute Features
    store = FeatureStore(market_type=market)
    feature_matrix = store.build_features(df, timestamp_col="timestamp")
    
    combined = df.copy()
    for col in feature_matrix.columns:
        combined[col] = feature_matrix[col]
        
    # 3. Apply Labels
    logger.info("applying_labels")
    labeled_df = apply_triple_barrier_labels(
        combined,
        profit_target_atr=3.0,
        stop_loss_atr=2.0,
        max_holding_bars=45,
        atr_col="ATR"
    )
    
    labeled_df = labeled_df.dropna(subset=["target"])
    
    X = labeled_df[store.feature_columns]
    y = labeled_df["target"]
    
    # Chronological Split (80/20) - Transformers require temporal continuity
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    # 4. Train Model
    model_name = f"{symbol}_{market}_transformer"
    transformer = ApexTransformerModel(
        name=model_name,
        feature_columns=store.feature_columns,
        seq_len=60
    )
    
    logger.info("training_transformer_model", name=model_name)
    metrics = transformer.train(X_train, y_train, X_val, y_val)
    
    # 5. Save Model
    models_dir = DATA_DIR / "models"
    transformer.save(models_dir)
    logger.info("transformer_training_complete", metrics=metrics, path=str(models_dir))
    
    # Cleanup memory
    del transformer, X_train, y_train, X_val, y_val, df, combined, labeled_df
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default="crypto")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    args = parser.parse_args()
    
    train_transformer(args.market, args.symbol)
