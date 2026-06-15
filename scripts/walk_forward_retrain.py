import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone

from src.core.logging import get_logger
from src.models.training.trainer import UnifiedTrainer
from src.core.config import DATA_DIR
from src.features.store import FeatureStore

logger = get_logger("apex.retrain_pipeline")

async def run_pipeline(symbol: str, market: str, days_lookback: int):
    """
    Continuous Walk-Forward Retraining Pipeline.
    
    1. Downloads latest data
    2. Slices the most recent `days_lookback` days
    3. Trains the 4-quadrant ensemble models
    4. Evaluates and promotes to production
    """
    logger.info("starting_walk_forward_pipeline", symbol=symbol, market=market, days=days_lookback)
    
    # In a real production setup, this would query the exchange for the missing days
    # and append to the historical CSV. For this script, we assume the CSV is updated
    # via `scripts/download_data.py` or a live data scraper.
    
    csv_path = DATA_DIR / "historical" / market / f"{symbol}.csv"
    if not csv_path.exists():
        logger.error("data_not_found", path=str(csv_path))
        return
        
    import pandas as pd
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Slice the walk-forward window
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_lookback)
    df_window = df[df["timestamp"] >= cutoff].copy()
    
    logger.info("data_sliced", total_rows=len(df), window_rows=len(df_window), cutoff=cutoff.isoformat())
    
    if len(df_window) < 5000:
        logger.error("insufficient_data_in_window", rows=len(df_window), required=5000)
        return
        
    # Start the UnifiedTrainer
    trainer = UnifiedTrainer(symbol=symbol, market_type=market)
    
    # Run the training pipeline (this automatically saves passing models)
    logger.info("training_ensemble_models")
    results = trainer.run_pipeline(save_models=True)
    
    success_count = sum(1 for res in results.values() if not res.get("skipped", False) and "final_metrics" in res)
    logger.info("pipeline_complete", successful_regimes=success_count, total_regimes=4)
    
    # In the future, we can add a local validation backtest here to compare the 
    # new ensemble's Sharpe ratio against the old ensemble before replacing the files.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--market", type=str, default="crypto")
    parser.add_argument("--days", type=int, default=30, help="Days of data to train on")
    
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.symbol, args.market, args.days))
