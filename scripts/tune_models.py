import argparse
import asyncio
import sys
import site
import os

# Ensure user site-packages are accessible so optuna can be found
user_site = site.USER_SITE
if user_site not in sys.path:
    sys.path.append(user_site)

import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json

from src.core.logging import get_logger
from src.features.store import FeatureStore
from src.features.labeling.triple_barrier import apply_triple_barrier_labels
from src.models.xgboost_model import ApexXGBoostModel

logger = get_logger("apex.tune_models")
CONFIG_PATH = Path("configs/models.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def objective(trial, X, y, feature_columns):
    """Optuna objective function for XGBoost tuning."""
    
    # Suggest hyperparameters
    hp = {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
    }

    # ── Purged K-Fold Cross Validation ──
    from src.models.cv_utils import PurgedTimeSeriesSplit
    
    cv = PurgedTimeSeriesSplit(n_splits=3, embargo_size=100)
    scores = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = ApexXGBoostModel(
            name=f"trial_{trial.number}",
            feature_columns=feature_columns,
            hyperparams=hp
        )

        metrics = model.train(X_train, y_train, X_val, y_val)
        
        # We want to maximize accuracy but heavily penalize log_loss
        score = metrics["accuracy"] - (metrics["log_loss"] * 0.1)
        scores.append(score)
        
    return float(np.mean(scores))

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default="crypto")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()

    logger.info("loading_data_for_tuning", symbol=args.symbol, market=args.market)
    
    data_path = Path("data/historical") / args.market / f"{args.symbol}.csv"
    if not data_path.exists():
        logger.error("data_not_found", path=str(data_path))
        return
        
    df = pd.read_csv(data_path)
    if df is None or len(df) < 500:
        logger.error("insufficient_data", symbol=args.symbol)
        return

    # Cap to 10000 to prevent swapping on low free-RAM systems
    if len(df) > 10000:
        df = df.iloc[-10000:].copy()

    logger.info("data_loaded", rows=len(df))
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # ── MEMORY CONSTRAINT FIX ──
    # Downsample to last 50,000 rows to prevent RAM OOM during Optuna tuning
    if len(df) > 50000:
        df = df.iloc[-50000:].copy()
    
    # Build features first so we have ATR for labeling
    store = FeatureStore(market_type=args.market)
    feature_matrix = store.build_features(df, timestamp_col="timestamp")
    
    # Combine original data and features
    combined = df.copy()
    for col in feature_matrix.columns:
        combined[col] = feature_matrix[col]
    
    # Apply triple barrier labeling
    config = load_config()
    labeling_cfg = config.get("crypto", {}).get("labeling", {})
    
    logger.info("applying_triple_barrier_labels", config=labeling_cfg)
    labeled_df = apply_triple_barrier_labels(
        combined,
        profit_target_atr=labeling_cfg.get("profit_target_atr", 3.0),
        stop_loss_atr=labeling_cfg.get("stop_loss_atr", 2.0),
        max_holding_bars=labeling_cfg.get("max_holding_bars", 45),
        atr_col="ATR"
    )
    
    # Drop unlabeled rows
    labeled_df = labeled_df.dropna(subset=["target"])
    
    # For tuning, we train on the entire dataset
    X = labeled_df[store.feature_columns]
    y = labeled_df["target"]
    
    logger.info("starting_optuna_study", trials=args.trials, samples=len(X))
    
    study = optuna.create_study(direction="maximize", study_name=f"{args.symbol}_tuning")
    study.optimize(lambda trial: objective(trial, X, y, store.feature_columns), n_trials=args.trials)
    
    logger.info("study_completed", best_score=study.best_value, best_params=study.best_params)
    
    # Save best parameters to models.yaml
    if args.market not in config:
        config[args.market] = {}
        
    if "hyperparameters" not in config[args.market]:
        config[args.market]["hyperparameters"] = {}
        
    config[args.market]["hyperparameters"].update(study.best_params)
    save_config(config)
    logger.info("updated_models_yaml_with_best_params")

if __name__ == "__main__":
    asyncio.run(main())
