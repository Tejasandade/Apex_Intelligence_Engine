"""
Apex Intelligence Engine V6 — RL Meta-Controller Training
=========================================================
Script to train the PPO RL agent on historical data.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.core.logging import get_logger
from src.rl.gym_env import ApexMetaEnv
from src.features.store import FeatureStore
from src.models.xgboost_model import ApexXGBoostModel
from src.core.config import DATA_DIR, get_market

logger = get_logger("apex.scripts.train_rl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--bars", type=int, default=20000)
    args = parser.parse_args()
    
    logger.info(f"Starting RL Meta-Controller Training for {args.symbol}")
    
    # 1. Load Data
    data_file = DATA_DIR / "historical" / "crypto" / f"{args.symbol}.parquet"
    if not data_file.exists():
        logger.error(f"Historical data not found: {data_file}")
        return
        
    df = pd.read_parquet(data_file)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if len(df) > args.bars:
        df = df.iloc[-args.bars:].reset_index(drop=True)
        
    logger.info(f"Loaded {len(df)} bars of historical data.")
    
    # 2. Build Features
    store = FeatureStore("crypto")
    features = store.build_features(df)
    
    # Forward return for reward calculation (assuming we trade at close and exit at next close)
    # Using 3 bars ahead to allow trade to play out
    df['forward_return'] = (df['close'].shift(-3) - df['close']) / df['close']
    df['forward_return'] = df['forward_return'].fillna(0.0)
    
    # 3. Load Models & Generate Predictions
    trending_model = ApexXGBoostModel("btcusdt_crypto_trending", store.feature_columns)
    trending_model.load()
    ranging_model = ApexXGBoostModel("btcusdt_crypto_ranging", store.feature_columns)
    ranging_model.load()
    
    logger.info("Generating base model predictions...")
    pred_trending = trending_model.predict(features)
    pred_ranging = ranging_model.predict(features)
    
    # 4. Prepare RL Dataset
    rl_data = pd.DataFrame()
    rl_data['prob_trending'] = features['Volatility_Regime'].apply(lambda x: 1.0 if x > 0.5 else 0.0) # Simplified prob
    rl_data['prob_ranging'] = 1.0 - rl_data['prob_trending']
    rl_data['atr_ratio'] = features['ATR_Ratio']
    rl_data['recent_accuracy'] = 0.5  # Static for training, dynamic in live
    rl_data['ob_bull_dist'] = features['OB_bull_dist']
    rl_data['ob_bear_dist'] = features['OB_bear_dist']
    
    rl_data['pred_trending'] = pred_trending
    rl_data['pred_ranging'] = pred_ranging
    rl_data['forward_return'] = df['forward_return']
    
    logger.info(f"RL dataset prepared with {len(rl_data)} rows.")
    
    # 5. Train RL Agent
    env = ApexMetaEnv(data=rl_data)
    
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("stable-baselines3 not installed. Run `pip install stable-baselines3`.")
        return
        
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)
    
    logger.info("Training PPO Agent...")
    total_timesteps = args.episodes * len(rl_data)
    model.learn(total_timesteps=total_timesteps)
    
    # 6. Save Model
    model_dir = DATA_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "rl_meta_controller"
    model.save(str(model_path))
    logger.info(f"Meta-Controller model saved to {model_path}.zip")

if __name__ == "__main__":
    main()
