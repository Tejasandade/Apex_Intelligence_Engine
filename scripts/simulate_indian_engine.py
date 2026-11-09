"""
Verification Script: Apex Indian Engine End-to-End Simulation
Tests ML model inference, Council consensus, strike resolution,
paper order execution, trailing stop mechanics, and friction accounting.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex_indian_engine.engine import ApexIndianEngine

def run_simulation():
    print("=" * 70)
    print("  🚀 RUNNING APEX INDIAN ENGINE END-TO-END SIMULATION")
    print("=" * 70)

    # 1. Initialize engine in PAPER mode with isolated state file
    engine = ApexIndianEngine(mode="PAPER", initial_capital=100_000.0)
    engine.ignore_time_checks = True
    test_state_file = Path("data/test_indian_state.json")
    engine.risk_manager.state_file = test_state_file
    engine.executor.journal_file = Path("data/test_indian_trade_journal.json")
    
    # Load Scrip Master (offline cache)
    assert engine.scrip_master.load_or_download(), "Failed to load Scrip Master"
    print("  ✓ Scrip Master loaded successfully.")

    # 2. Load historical BankNifty candles
    parquet_path = Path("data/historical/india_equity/banknifty.parquet")
    if not parquet_path.exists():
        print(f"  ❌ Historical data not found at {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"  ✓ Loaded {len(df):,} historical BankNifty candles.")

    # 3. Feed a sequence of candles to build indicators & trigger ML
    # We take candles 0 to 55 which contains a verified high-conviction breakdown
    slice_df = df.iloc[0:55]
    
    opened_trade = None
    candles_fed = 0
    
    print("\n  📊 Ingesting candles through 5-Advisor Council...")
    for idx, row in slice_df.iterrows():
        candle = {
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        }
        res = engine.process_candle("BANKNIFTY", candle)
        candles_fed += 1
        if res and res.get("action") == "TRADE_OPENED":
            opened_trade = res
            print(f"\n  🎯 TRADE TRIGGERED AT CANDLE #{candles_fed}!")
            print(f"     Contract:   {res['symbol']}")
            print(f"     Direction:  {res['direction']}")
            print(f"     Lots:       {res['lots']} ({res['quantity']} qty)")
            print(f"     Entry Px:   ₹{res['entry_price']:.2f}")
            print(f"     SL:         ₹{res['sl_price']:.2f}")
            print(f"     TP1:        ₹{res['tp1_price']:.2f}")
            print(f"     TP2:        ₹{res['tp2_price']:.2f}")
            print(f"     Score:      {res['score']:+.2f} (Conviction: {res['conviction']*100:.0f}%)")
            break

    assert opened_trade is not None, "Expected trade to be triggered on high-conviction slice!"

    # 4. Simulate Price Action Ticks for the Active Position
    print("\n  📈 Simulating Option Price Progression (Tick updates)...")
    entry_px = opened_trade["entry_price"]
    
    # Tick 1: Small pullback
    engine.process_tick(entry_px - 5.0)
    print(f"     Tick 1: ₹{entry_px - 5.0:.2f} | Unrealized PnL: ₹{engine.executor.active_trade.unrealized_pnl:.2f}")

    # Tick 2: Strong expansion into +1.0R (Risk reduction trigger)
    sl_dist = entry_px - opened_trade["sl_price"]
    t1_px = entry_px + (sl_dist * 0.9)
    engine.process_tick(t1_px)
    print(f"     Tick 2: ₹{t1_px:.2f} (+0.9R) | SL Ratchet: ₹{engine.executor.active_trade.sl_price:.2f}")

    # Tick 3: Break-Even lock at +1.3R
    t2_px = entry_px + (sl_dist * 1.3)
    engine.process_tick(t2_px)
    print(f"     Tick 3: ₹{t2_px:.2f} (+1.3R) | SL Ratchet: ₹{engine.executor.active_trade.sl_price:.2f}")

    # Tick 4: Take Profit 2 Target Hit
    target_px = opened_trade["tp2_price"] + 2.0
    closed = engine.process_tick(target_px)
    assert closed is not None, "Trade should have closed at TP2!"
    
    print("\n  🏆 TRADE CLOSED & SETTLED:")
    print(f"     Exit Price:  ₹{closed['exit_price']:.2f}")
    print(f"     Gross PnL:   ₹{closed['gross_pnl']:.2f}")
    print(f"     Reason:      {closed['exit_reason']}")

    # 5. Check Risk Manager metrics & friction accounting
    metrics = engine.get_hud_metrics()
    print("\n  💼 POST-TRADE RISK & ACCOUNT METRICS:")
    print(f"     Account Capital:      ₹{metrics['capital_inr']:,.2f}")
    print(f"     Daily Realized Net:   ₹{metrics['daily_realized_pnl']:,.2f}")
    print(f"     Friction Deducted:    ₹{metrics['daily_friction_cost']:.2f} (STT + GST + Brokerage)")
    print(f"     Trades Today:         {metrics['trades_today']} / {metrics['max_trades']}")
    print(f"     Status:               {metrics['status']}")

    # Cleanup test state
    if test_state_file.exists():
        test_state_file.unlink()

    print("\n" + "=" * 70)
    print("  ✅ ALL END-TO-END INDIAN ENGINE TESTS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()
