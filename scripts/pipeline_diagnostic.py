# -*- coding: utf-8 -*-
"""
PIPELINE DIAGNOSTIC + STRIPPED BACKTEST
========================================
Step 1: Count where signals die in the current pipeline
Step 2: Run backtest with filters disabled one-by-one
"""
import sys
import asyncio
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.features.store import FeatureStore
from src.features.live_store import LiveFeatureStore
from src.engine.signal_engine import LiveSignalEngine
from src.execution.position_manager import PositionManager
from src.execution.paper_broker import PaperBroker


async def run_diagnostic():
    symbol = "btcusdt"
    market_type = "crypto"
    
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/historical/cleaned/BTCUSDT_1m_cleaned.csv")
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})
    
    # Limit to 3000 bars for speed
    df = df.iloc[-3000:].reset_index(drop=True)
    # Convert timestamps to Unix ms for the signal engine
    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**6
    print(f"  Using {len(df):,} bars")
    
    # Load models
    from src.models.xgboost_model import ApexXGBoostModel
    from src.core.config import DATA_DIR
    MODELS_DIR = DATA_DIR / "models"
    feature_store = FeatureStore(market_type)
    
    models = {}
    regimes = ["trending_high_vol", "trending_low_vol", "ranging_high_vol", "ranging_low_vol"]
    for regime in regimes:
        for bag in [1, 2, 3]:
            model_name = f"{symbol}_{market_type}_{regime}_bag{bag}"
            model_path = MODELS_DIR / f"{model_name}.json"
            if model_path.exists():
                model = ApexXGBoostModel(
                    name=model_name,
                    feature_columns=feature_store.feature_columns,
                )
                model.load(MODELS_DIR)
                models[f"{regime}_bag{bag}"] = model
                print(f"  Loaded {model_name}")
    
    print(f"  Loaded {len(models)} models")
    
    # ================================================================
    # TEST CONFIGURATIONS
    # ================================================================
    configs = [
        {
            "name": "FULL PIPELINE (current - all filters ON)",
            "enable_ensemble": True,
            "enable_council": True,
            "consensus_threshold": 0.50,
            "conviction_threshold": 0.58,
        },
        {
            "name": "NO COUNCIL (model + ensemble only)",
            "enable_ensemble": True,
            "enable_council": False,
            "consensus_threshold": 0.50,
            "conviction_threshold": 0.58,
        },
        {
            "name": "NO ENSEMBLE (single model + council)",
            "enable_ensemble": False,
            "enable_council": True,
            "consensus_threshold": 0.50,
            "conviction_threshold": 0.58,
        },
        {
            "name": "MINIMAL (single model, no council, no ensemble)",
            "enable_ensemble": False,
            "enable_council": False,
            "consensus_threshold": 0.50,
            "conviction_threshold": 0.58,
        },
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"  CONFIG: {config['name']}")
        print(f"{'='*60}")
        
        # Setup broker + position manager
        broker = PaperBroker(initial_balance=10000, maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps=1.0)
        await broker.connect()
        
        config_path = Path("configs/models.yaml")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
            tm_config = yaml_config.get("trade_management", {})
        
        breakeven = tm_config.get("breakeven", 1.5)
        trail_trigger = tm_config.get("trail_trigger", 2.0)
        trail_mult = tm_config.get("trail_mult", 1.5)
        
        dynamic_breakeven = {"TRENDING": breakeven, "RANGING": breakeven, "VOLATILE": breakeven, "QUIET": breakeven}
        dynamic_trailing = {"TRENDING": trail_trigger, "RANGING": trail_trigger, "VOLATILE": trail_trigger, "QUIET": trail_trigger}
        dynamic_mults = {"TRENDING": trail_mult, "RANGING": trail_mult, "VOLATILE": trail_mult, "QUIET": trail_mult}
        
        position_mgr = PositionManager(
            broker=broker,
            max_positions=3,
            max_drawdown_pct=100.0,
            daily_loss_limit=10000 * 0.50,
            enable_scale_out=False,
            max_daily_trades=50,
            enable_trailing_stop=True,
            dynamic_breakeven_triggers=dynamic_breakeven,
            dynamic_trailing_triggers=dynamic_trailing,
            dynamic_trail_multipliers=dynamic_mults,
        )
        await position_mgr.initialize()
        
        signal_engine = LiveSignalEngine(
            symbol=symbol,
            market_type=market_type,
            models=models,
            conviction_threshold=config["conviction_threshold"],
            risk_per_trade=50.0,
            enable_ensemble=config["enable_ensemble"],
            enable_monitoring=False,
            enable_council=config["enable_council"],
            enable_walk_forward=False,
            max_concurrent=3,
            consensus_threshold=config["consensus_threshold"],
        )
        
        # Warmup
        signal_engine.warm_up(df.iloc[:60])
        
        # Run simulation
        signals_total = 0
        signals_approved = 0
        trades_taken = 0
        
        current_date = None
        
        for idx in range(60, len(df)):
            row = df.iloc[idx].to_dict()
            
            timestamp_val = row.get("timestamp")
            if isinstance(timestamp_val, (int, float)):
                current_time = pd.to_datetime(timestamp_val, unit='ms', utc=True).to_pydatetime()
                row_date = current_time.date()
            else:
                current_time = pd.to_datetime(timestamp_val, utc=True).to_pydatetime()
                row_date = current_time.date()
            
            if row_date and row_date != current_date:
                current_date = row_date
                position_mgr.reset_daily()
            
            # Update prices
            broker.update_price(symbol, row["high"], current_time)
            await position_mgr.update_price(symbol, row["high"], current_time)
            broker.update_price(symbol, row["low"], current_time)
            await position_mgr.update_price(symbol, row["low"], current_time)
            broker.update_price(symbol, row["close"], current_time)
            await position_mgr.update_price(symbol, row["close"], current_time)
            
            # Sync positions
            positions = await broker.get_positions()
            active_positions = [p for p in positions if p.side.value != "FLAT"]
            signal_engine.set_open_positions(len(active_positions))
            
            # Feed candle
            signal = await signal_engine.process_candle(row)
            
            # Ratchet trailing stops
            position_mgr.update_on_candle_close(
                symbol=symbol,
                close_price=row["close"],
                atr=signal_engine._last_signal.atr if signal_engine._last_signal else (row["close"] * 0.005),
                regime=signal_engine.regime_detector.current_regime.value if signal_engine.regime_detector else "TRENDING"
            )
            
            if signal:
                signals_total += 1
                if signal.council_approved:
                    signals_approved += 1
                    order = await position_mgr.open_position(
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
                    if order and order.status.value == "FILLED":
                        trades_taken += 1
        
        # Close remaining
        positions = await broker.get_positions()
        for pos in positions:
            if pos.side.value != "FLAT":
                await position_mgr.close_position(pos.symbol)
        
        # Results
        final_balance = await broker.get_balance()
        final_equity = final_balance.total_equity
        pnl = final_equity - 10000
        pnl_pct = (pnl / 10000) * 100
        
        trades = broker.get_trade_history()
        
        print(f"  Signals generated: {signals_total}")
        print(f"  Signals approved:  {signals_approved}")
        print(f"  Trades executed:   {trades_taken}")
        print(f"  Final Balance:     ${final_equity:.2f}")
        print(f"  Net P&L:           ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        if trades:
            wins = sum(1 for t in trades if t.pnl > 0)
            losses = sum(1 for t in trades if t.pnl < 0)
            total = wins + losses
            if total > 0:
                print(f"  Win Rate:          {wins/total:.1%}")


if __name__ == "__main__":
    asyncio.run(run_diagnostic())
