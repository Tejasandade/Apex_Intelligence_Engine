"""
Apex Indian Engine — Standalone Runner
======================================
Usage:
    python -m apex_indian_engine.run_indian [--mode paper|live] [--capital 100000]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

load_dotenv(os.path.join(WORKSPACE_ROOT, ".env"))

from apex_indian_engine.engine import ApexIndianEngine


def main():
    parser = argparse.ArgumentParser(description="Apex Indian Options Trading Engine")
    parser.add_argument("--mode", choices=["paper", "live"], default="paper", help="Execution mode (default: paper)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital in INR (default: ₹1,00,000)")
    args = parser.parse_args()

    engine = ApexIndianEngine(mode=args.mode.upper(), initial_capital=args.capital)
    if not engine.initialize():
        print("❌ Engine initialization failed. Check credentials and scrip master.")
        sys.exit(1)

    print("\n  🚀 Apex Indian Engine is active and monitoring NSE sessions...")
    print(f"  Mode: {engine.config.mode} | Capital: ₹{engine.risk_manager.capital_inr:,.2f}")
    print("  Press Ctrl+C to safely pause and exit.\n")

    try:
        asyncio.run(engine.run_loop())
    except KeyboardInterrupt:
        print("\n  🛑 Apex Indian Engine safely stopped by user.")


if __name__ == "__main__":
    main()

