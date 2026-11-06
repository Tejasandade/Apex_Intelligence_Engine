"""
Apex Intelligence Engine — MT5 Position Controller
Close or inspect open MT5 positions.
Usage:
  python scripts/close_mt5_position.py         (Lists open positions)
  python scripts/close_mt5_position.py close   (Closes open BTC position and banks profit)
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv

load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 python package not installed.")
    sys.exit(1)

if not mt5.initialize():
    print("❌ Failed to initialize MT5:", mt5.last_error())
    sys.exit(1)

from src.execution.mt5_broker import MT5Broker

broker = MT5Broker()

positions = mt5.positions_get()
print("=" * 60)
print(f"  ACTIVE MT5 POSITIONS: {len(positions) if positions else 0}")
print("=" * 60)

if not positions:
    print("  No open positions found in MT5.")
    mt5.shutdown()
    sys.exit(0)

for p in positions:
    p_type = "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL"
    print(f"  Ticket:       {p.ticket}")
    print(f"  Symbol:       {p.symbol}")
    print(f"  Type:         {p_type}")
    print(f"  Volume:       {p.volume}")
    print(f"  Open Price:   ${p.price_open:,.2f}")
    print(f"  Current:      ${p.price_current:,.2f}")
    print(f"  Stop Loss:    ${p.sl:,.2f}")
    print(f"  Take Profit:  ${p.tp:,.2f}")
    print(f"  Floating PnL: ${p.profit:+,.2f} USD")
    print("-" * 60)

if len(sys.argv) > 1 and sys.argv[1].lower() == "close":
    print("⚡ Closing active positions...")
    for p in positions:
        ok = broker._close_position_by_ticket_sync(p.ticket)
        if ok:
            print(f"✅ Successfully closed ticket {p.ticket}!")
        else:
            print(f"❌ Failed to close ticket {p.ticket}")

mt5.shutdown()
