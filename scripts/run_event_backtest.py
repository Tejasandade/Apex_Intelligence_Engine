"""
Apex Intelligence Engine V5 — Event-Driven Backtest Simulator
=============================================================
Simulates the live environment accurately using historical data.
"""

import asyncio
import sys
import time
import argparse
import yaml
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.core.config import DATA_DIR
from src.core.logging import get_logger
from src.engine.signal_engine import LiveSignalEngine
from src.execution.paper_broker import PaperBroker
from src.execution.position_manager import PositionManager
from src.models.xgboost_model import ApexXGBoostModel
from src.features.store import FeatureStore
from tqdm import tqdm

logger = get_logger("apex.scripts.event_backtest")

async def main():
    parser = argparse.ArgumentParser(description="Apex V5 Event-Driven Backtest")
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--bars", type=int, default=1000, help="Number of recent 1m bars to simulate")
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital")
    parser.add_argument("--asymmetric", action="store_true", help="Enable Asymmetric Risk Mode")
    parser.add_argument("--council-threshold", type=float, default=0.60,
                        help="Consensus threshold for council approval (0 to 1)")
    parser.add_argument("--scalp", action="store_true",
                        help="Enable Scalp Mode (high frequency, lower thresholds, loose rules)")
    args = parser.parse_args()

    symbol = args.symbol.lower()
    total_bars = args.bars
    market_type = "crypto"

    print(f"\n[SIMULATOR] Starting Event-Driven Backtest for {symbol}")
    print(f"[SIMULATOR] Capital: ${args.capital}")
    print(f"[SIMULATOR] Asymmetric Mode: {args.asymmetric}")
    print(f"[SIMULATOR] Total Bars: {total_bars}")

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    data_path = DATA_DIR / "historical" / "crypto" / f"{symbol}.parquet"
    if not data_path.exists():
        print(f"[ERROR] No historical data found at {data_path}. Run normal backtester first to download.")
        return

    df = pd.read_parquet(data_path)
    if len(df) > total_bars:
        df = df.tail(total_bars).reset_index(drop=True)
        
    print(f"[SIMULATOR] Loaded {len(df)} historical bars.")

    # ── 2. Setup Models ───────────────────────────────────────────────────────
    print("[SIMULATOR] Loading Base AI Models...")
    models = {}
    feature_store = FeatureStore(market_type)
    MODELS_DIR = DATA_DIR / "models"
    
    # NEW REGIMES + BAGGING
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
                print(f"[SIMULATOR] Loaded {model_name}")
    
    if not models:
        print("[ERROR] No models found. Train models first.")
        return

    # ── 3. Setup Components ───────────────────────────────────────────────────
    broker = PaperBroker(initial_balance=args.capital, maker_fee_bps=0.0, taker_fee_bps=0.0, slippage_bps=1.0)
    await broker.connect()

    config_path = Path("configs/models.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        tm_config = config.get("trade_management", {})
        
    breakeven = tm_config.get("breakeven", 1.5)
    trail_trigger = tm_config.get("trail_trigger", 2.0)
    trail_mult = tm_config.get("trail_mult", 1.5)
    
    dynamic_breakeven = {"TRENDING": breakeven, "RANGING": breakeven, "VOLATILE": breakeven, "QUIET": breakeven}
    dynamic_trailing = {"TRENDING": trail_trigger, "RANGING": trail_trigger, "VOLATILE": trail_trigger, "QUIET": trail_trigger}
    dynamic_mults = {"TRENDING": trail_mult, "RANGING": trail_mult, "VOLATILE": trail_mult, "QUIET": trail_mult}

    position_mgr = PositionManager(
        broker=broker,
        max_positions=3,
        max_drawdown_pct=100.0,  # Prevent kill switch in backtest
        daily_loss_limit=args.capital * 0.50,
        enable_scale_out=False,
        max_daily_trades=50,
        enable_trailing_stop=True,
        dynamic_breakeven_triggers=dynamic_breakeven,
        dynamic_trailing_triggers=dynamic_trailing,
        dynamic_trail_multipliers=dynamic_mults
    )
    await position_mgr.initialize()

    signal_engine = LiveSignalEngine(
        symbol=symbol,
        market_type=market_type,
        models=models,
        conviction_threshold=0.58,
        risk_per_trade=50.0,  # Match live runner configuration
        enable_ensemble=True,
        enable_monitoring=True,
        enable_council=True,
        enable_walk_forward=False,
        max_concurrent=1,
        consensus_threshold=0.50,
    )
    
    print("[SIMULATOR] Warming up engines (60 bars)...")
    warmup_df = df.iloc[:60]
    signal_engine.warm_up(warmup_df)
    
    # ── 4. Run Simulation ─────────────────────────────────────────────────────
    print("[SIMULATOR] Commencing Tick-by-Tick Simulation...\n")
    
    signals_generated = 0
    trades_taken = 0
    
    # We simulate tick-by-tick by processing each candle
    current_date = None
    loop = asyncio.get_event_loop()
    for idx in tqdm(range(60, len(df)), desc="Simulating Time"):
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
        
        # In a real tick environment, price moves within the candle.
        # We will simulate high/low for trailing stops by calling update_price.
        # First, pass the High of the candle (to trigger limits/stops)
        broker.update_price(symbol, row["high"], current_time)
        await position_mgr.update_price(symbol, row["high"], current_time)
        # Then pass the Low of the candle
        broker.update_price(symbol, row["low"], current_time)
        await position_mgr.update_price(symbol, row["low"], current_time)
        # Then pass the Close
        broker.update_price(symbol, row["close"], current_time)
        await position_mgr.update_price(symbol, row["close"], current_time)
        
        # Sync open position count
        positions = await broker.get_positions()
        active_positions = [p for p in positions if p.side.value != "FLAT"]
        signal_engine.set_open_positions(len(active_positions))
        
        # Feed candle to engine
        signal = await signal_engine.process_candle(row)
        
        # Ratchet trailing stops
        position_mgr.update_on_candle_close(
            symbol=symbol,
            close_price=row["close"],
            atr=signal_engine._last_signal.atr if signal_engine._last_signal else (row["close"] * 0.005),
            regime=signal_engine.regime_detector.current_regime.value if signal_engine.regime_detector else "TRENDING"
        )
        
        if signal:
            signals_generated += 1
            if signal.council_approved:
                # Execute Trade
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
            else:
                print(f"[{row_date}] Signal {signal.direction} blocked by Council. Score: {signal.council_score}. Reason: {signal.member_predictions.get('council_summary', 'N/A')}")
                
    # ── 5. Output Report ──────────────────────────────────────────────────────
    print("\n========================================================")
    print("  SIMULATION COMPLETE")
    print("========================================================")
    
    # ── Close all open positions before printing stats ────────────────────────
    print("[SIMULATOR] Closing any remaining open positions...")
    positions = await broker.get_positions()
    for pos in positions:
        if pos.side.value != "FLAT":
            await position_mgr.close_position(pos.symbol)

    final_balance = await broker.get_balance()
    final_equity = final_balance.total_equity
    pnl = final_equity - args.capital
    pnl_pct = (pnl / args.capital) * 100
    
    trades = broker.get_trade_history()
    
    # Group partial and full closes by signal_id to get true per-trade metrics
    from collections import defaultdict
    trades_by_signal = defaultdict(list)
    for t in trades:
        trades_by_signal[t.signal_id].append(t)
        
    true_pnls = []
    true_durations = []
    for sig_id, trade_group in trades_by_signal.items():
        net_pnl = sum(t.pnl for t in trade_group)
        true_pnls.append(net_pnl)
        
        # Duration is from first entry to last exit
        first_entry = min(t.entry_time for t in trade_group if t.entry_time) if any(t.entry_time for t in trade_group) else None
        last_exit = max(t.exit_time for t in trade_group if t.exit_time) if any(t.exit_time for t in trade_group) else None
        if first_entry and last_exit:
            duration = (last_exit - first_entry).total_seconds()
            true_durations.append(duration)
            
    wins = [p for p in true_pnls if p > 0]
    losses = [p for p in true_pnls if p <= 0]
    
    win_rate = (len(wins) / len(true_pnls) * 100) if true_pnls else 0.0
    
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    
    # ── Max Drawdown Calculation ──────────────────────────────────────────
    peak_equity = args.capital
    max_drawdown = 0.0
    max_drawdown_pct = 0.0
    running_equity = args.capital
    # We must calculate drawdown sequentially as events occurred
    for t in trades:
        running_equity += t.pnl
        if running_equity > peak_equity:
            peak_equity = running_equity
        drawdown = peak_equity - running_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            max_drawdown_pct = (drawdown / peak_equity) * 100.0
    
    # ── Average Hold Time ─────────────────────────────────────────────────
    avg_hold_seconds = sum(true_durations) / len(true_durations) if true_durations else 0.0
    avg_hold_minutes = avg_hold_seconds / 60.0
    
    print(f"  Final Balance:     ${final_equity:.2f}")
    print(f"  Net P&L:           ${pnl:.2f} ({pnl_pct:.2f}%)")
    print(f"  Total Signals:     {signals_generated}")
    print(f"  Trades Executed:   {len(true_pnls)} (Completed Cycles)")
    print(f"  Win Rate:          {win_rate:.1f}%")
    print(f"  Average Win:       ${avg_win:.2f}")
    print(f"  Average Loss:      ${avg_loss:.2f}")
    print(f"  Expectancy/Trade:  ${expectancy:.2f}")
    print(f"  Max Drawdown:      ${max_drawdown:.2f} ({max_drawdown_pct:.1f}%)")
    print(f"  Avg Hold Time:     {avg_hold_minutes:.1f} minutes ({avg_hold_seconds:.0f}s)")
    print("========================================================\n")
    
    # Save the Pyramiding history state to disk so LiveRunner can use it!
    position_mgr.save_state()
    print("[SIMULATOR] Trade history state saved successfully.")
    
if __name__ == "__main__":
    asyncio.run(main())
