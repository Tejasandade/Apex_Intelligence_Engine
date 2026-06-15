import argparse
import asyncio
import sys
import site
import os

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

logging.getLogger("apex").setLevel(logging.WARNING)
logger = get_logger("apex.tune_council")
CONFIG_PATH = Path("configs/models.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

async def run_simulation(df, params, symbol):
    broker = PaperBroker(initial_balance=1000.0, commission_bps=2.0, slippage_bps=0.5)
    
    # Use the best trailing stop parameters found earlier
    config = load_config()
    tm_config = config.get("trade_management", {})
    breakeven = tm_config.get("breakeven", 2.4)
    trail_trigger = tm_config.get("trail_trigger", 2.0)
    trail_mult = tm_config.get("trail_mult", 1.1)
    
    dynamic_breakeven = {"TRENDING": breakeven, "RANGING": breakeven, "VOLATILE": breakeven, "QUIET": breakeven}
    dynamic_trailing = {"TRENDING": trail_trigger, "RANGING": trail_trigger, "VOLATILE": trail_trigger, "QUIET": trail_trigger}
    dynamic_mults = {"TRENDING": trail_mult, "RANGING": trail_mult, "VOLATILE": trail_mult, "QUIET": trail_mult}
    
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
    # Load bagging models
    regimes = ["trending_high_vol", "trending_low_vol", "ranging_high_vol", "ranging_low_vol"]
    for regime in regimes:
        for bag in [1, 2, 3]:
            model_name = f"{symbol}_crypto_{regime}_bag{bag}"
            model_path = MODELS_DIR / f"{model_name}.json"
            if model_path.exists():
                model = ApexXGBoostModel(
                    name=model_name,
                    feature_columns=feature_store.feature_columns,
                )
                model.load(MODELS_DIR)
                models[f"{regime}_bag{bag}"] = model
    
    signal_engine = LiveSignalEngine(
        symbol=symbol,
        market_type="crypto",
        models=models,
        risk_per_trade=50.0,
        conviction_threshold=0.52,  # Fixed conviction, relying on council
        consensus_threshold=params["council_threshold"],  # Dynamic Threshold
        enable_council=True,
        enable_ensemble=False
    )
    
    # Inject dynamic weights into the advisors
    if signal_engine.council:
        for advisor in signal_engine.council.advisors:
            if advisor.name == "Momentum":
                advisor.weight = params["weight_momentum"]
            elif advisor.name == "Structure":
                advisor.weight = params["weight_structure"]
            elif advisor.name == "Volume":
                advisor.weight = params["weight_volume"]
            elif advisor.name == "Regime":
                advisor.weight = params["weight_regime"]
            elif advisor.name == "Session":
                advisor.weight = params["weight_session"]
    
    signals_count = 0
    
    warmup_bars = 60
    warmup_df = df.iloc[:warmup_bars]
    signal_engine.warm_up(warmup_df)
    
    for i in range(warmup_bars, len(df)):
        row = df.iloc[i]
        current_time = row["timestamp"]
        
        broker.update_price(symbol, row["close"], current_time)
        await position_mgr.update_price(symbol, row["close"], current_time)
        
        positions = await broker.get_positions()
        active_positions = [p for p in positions if p.side.value != "FLAT"]
        signal_engine.set_open_positions(len(active_positions))
        
        signal = await signal_engine.process_candle(row)
        
        position_mgr.update_on_candle_close(
            symbol=symbol,
            close_price=row["close"],
            atr=signal_engine._last_signal.atr if signal_engine._last_signal else (row["close"] * 0.005),
            regime=signal_engine.regime_detector.current_regime.value if signal_engine.regime_detector else "TRENDING"
        )
        
        if signal and signal.council_approved:
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
    
    # Very minor penalty for executing too few trades to encourage high volume
    trade_penalty = max(0, (50 - len(trades)) * 1.0)
    
    if len(trades) == 0:
        return -9999.0
    return (final_balance.total_equity - 1000.0) - trade_penalty

def objective(trial, df, symbol):
    params = {
        "council_threshold": trial.suggest_float("council_threshold", 0.40, 0.85, step=0.05),
        "weight_momentum": trial.suggest_float("weight_momentum", 0.0, 3.0, step=0.2),
        "weight_structure": trial.suggest_float("weight_structure", 0.0, 3.0, step=0.2),
        "weight_volume": trial.suggest_float("weight_volume", 0.0, 3.0, step=0.2),
        "weight_regime": trial.suggest_float("weight_regime", 0.0, 3.0, step=0.2),
        "weight_session": trial.suggest_float("weight_session", 0.0, 3.0, step=0.2),
    }
    
    pnl = asyncio.run(run_simulation(df, params, symbol))
    return pnl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--symbol", type=str, default="btcusdt")
    args = parser.parse_args()

    # Try parquet first, fallback to csv
    data_path_pq = Path(f"data/historical/crypto/{args.symbol}.parquet")
    data_path_csv = Path(f"data/historical/crypto/{args.symbol}.csv")
    
    if data_path_pq.exists():
        df = pd.read_parquet(data_path_pq)
    elif data_path_csv.exists():
        df = pd.read_csv(data_path_csv)
    else:
        print(f"Data missing for {args.symbol}")
        return
        
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    df = df.iloc[-3500:].copy()  # ~2.5 days of data
    
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, df, args.symbol), n_trials=args.trials)
    
    print("\n\n=========================================")
    print("  COUNCIL WEIGHT TUNING COMPLETE")
    print("=========================================")
    print(f"Best P&L (Score): {study.best_value:.2f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v:.2f}")
        
    # Save best parameters to models.yaml
    config = load_config()
    if "council_weights" not in config:
        config["council_weights"] = {}
    if args.symbol not in config["council_weights"]:
        config["council_weights"][args.symbol] = {}
        
    config["council_weights"][args.symbol].update(study.best_params)
    save_config(config)

if __name__ == "__main__":
    main()
