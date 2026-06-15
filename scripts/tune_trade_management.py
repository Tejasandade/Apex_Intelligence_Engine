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
import logging

from src.core.logging import get_logger
from src.core.config import DATA_DIR
from src.features.store import FeatureStore
from src.engine.signal_engine import LiveSignalEngine
from src.execution.position_manager import PositionManager
from src.execution.paper_broker import PaperBroker
from src.models.xgboost_model import ApexXGBoostModel
from src.features.store import FeatureStore

# Suppress heavy logging during tuning
logging.getLogger("apex").setLevel(logging.WARNING)
logger = get_logger("apex.tune_execution")
CONFIG_PATH = Path("configs/models.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

async def run_simulation(df, params):
    broker = PaperBroker(initial_balance=1000.0, commission_bps=2.0, slippage_bps=0.5)
    
    # Configure dynamic trailing stop multipliers
    dynamic_breakeven = {"TRENDING": params["breakeven"], "RANGING": params["breakeven"], "VOLATILE": params["breakeven"], "QUIET": params["breakeven"]}
    dynamic_trailing = {"TRENDING": params["trail_trigger"], "RANGING": params["trail_trigger"], "VOLATILE": params["trail_trigger"], "QUIET": params["trail_trigger"]}
    dynamic_mults = {"TRENDING": params["trail_mult"], "RANGING": params["trail_mult"], "VOLATILE": params["trail_mult"], "QUIET": params["trail_mult"]}
    
    position_mgr = PositionManager(
        broker=broker,
        max_positions=3,
        daily_loss_limit=500.0,
        enable_trailing_stop=True,
        enable_scale_out=False,
        dynamic_breakeven_triggers=dynamic_breakeven,
        dynamic_trailing_triggers=dynamic_trailing,
        dynamic_trail_multipliers=dynamic_mults
    )
    
    MODELS_DIR = Path("data/models")
    feature_store = FeatureStore("crypto")
    models = {}
    for regime in ["TRENDING", "RANGING"]:
        model_name = f"btcusdt_crypto_{regime.lower()}"
        model = ApexXGBoostModel(
            name=model_name,
            feature_columns=feature_store.feature_columns,
        )
        model.load(MODELS_DIR)
        models[regime] = model
    
    signal_engine = LiveSignalEngine(
        symbol="btcusdt",
        market_type="crypto",
        models=models,
        risk_per_trade=50.0,
        conviction_threshold=0.52,
        enable_council=False,
        enable_ensemble=False
    )
    
    signals_count = 0
    
    warmup_bars = 60
    warmup_df = df.iloc[:warmup_bars]
    signal_engine.warm_up(warmup_df)
    
    for i in range(warmup_bars, len(df)):
        row = df.iloc[i]
        current_time = row["timestamp"]
        
        broker.update_price("btcusdt", row["close"], current_time)
        await position_mgr.update_price("btcusdt", row["close"], current_time)
        
        positions = await broker.get_positions()
        active_positions = [p for p in positions if p.side.value != "FLAT"]
        signal_engine.set_open_positions(len(active_positions))
        
        signal = await signal_engine.process_candle(row)
        
        position_mgr.update_on_candle_close(
            symbol="btcusdt",
            close_price=row["close"],
            atr=signal_engine._last_signal.atr if signal_engine._last_signal else (row["close"] * 0.005),
            regime=signal_engine.regime_detector.current_regime.value if signal_engine.regime_detector else "TRENDING"
        )
        
        if signal:
            signals_count += 1
            await position_mgr.open_position(
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                direction=signal.direction,
                quantity=signal.quantity,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                regime=signal.regime,
                council_score=signal.council_score,
                atr=signal.atr,
            )
            
    final_balance = await broker.get_balance()
    trades = broker.get_trade_history()
    print(f"DEBUG: signals_count={signals_count}, trades_count={len(trades)}")
    if len(trades) == 0:
        return -9999.0  # Penalize 0 trades heavily
    return final_balance.total_equity - 1000.0

def objective(trial, df):
    # We want to tune trade management
    params = {
        "breakeven": trial.suggest_float("breakeven", 0.5, 2.5, step=0.1),
        "trail_trigger": trial.suggest_float("trail_trigger", 1.0, 3.5, step=0.1),
        "trail_mult": trial.suggest_float("trail_mult", 0.5, 3.0, step=0.1)
    }
    
    # Ensure trail_trigger is >= breakeven
    if params["trail_trigger"] <= params["breakeven"]:
        params["trail_trigger"] = params["breakeven"] + 0.1
        
    pnl = asyncio.run(run_simulation(df, params))
    return pnl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    data_path = Path("data/historical/crypto/btcusdt.csv")
    if not data_path.exists():
        return
        
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Use only last 2500 bars for extremely fast tuning (approx 1.5 days of 1m data)
    # This prevents the tuning script from taking hours, while still giving us a decent sample size.
    df = df.iloc[-2500:].copy()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df), n_trials=args.trials)
    
    print("\n\n=========================================")
    print("  TRADE MANAGEMENT TUNING COMPLETE")
    print("=========================================")
    print(f"Best P&L: ${study.best_value:.2f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.2f}")
        
    # Save best parameters to models.yaml
    config = load_config()
    if "trade_management" not in config:
        config["trade_management"] = {}
    config["trade_management"].update(study.best_params)
    save_config(config)

if __name__ == "__main__":
    main()
