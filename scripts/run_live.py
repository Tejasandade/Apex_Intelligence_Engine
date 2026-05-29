"""
Apex Intelligence Engine V5 — Live Runner Script
===================================================
Entry point for running the live trading engine and WebSocket dashboard server.

Usage:
    python -m scripts.run_live --symbol btcusdt
    python -m scripts.run_live --symbol btcusdt --paper-mode
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.live_runner import LiveRunner
from src.core.logging import get_logger

logger = get_logger("apex.scripts.run_live")

async def main():
    parser = argparse.ArgumentParser(description="Apex Intelligence Engine V5 - Live Runner")
    parser.add_argument("--symbol", type=str, action="append", help="Symbols to trade (e.g. btcusdt)")
    parser.add_argument("--market", type=str, default="crypto", help="Market type (crypto, india)")
    parser.add_argument("--live", action="store_true", help="Run with real money (disables paper mode)")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard WebSocket port")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard server")
    
    args = parser.parse_args()
    
    symbols = args.symbol or ["btcusdt"]
    paper_mode = not args.live
    enable_dashboard = not args.no_dashboard
    
    print(f"Starting Apex Engine...")
    print(f"Symbols: {symbols}")
    print(f"Mode: {'PAPER' if paper_mode else 'LIVE'}")
    print(f"Dashboard: {'Enabled (Port ' + str(args.port) + ')' if enable_dashboard else 'Disabled'}")
    
    runner = LiveRunner(
        symbols=symbols,
        market_type=args.market,
        paper_mode=paper_mode,
        dashboard_port=args.port,
        enable_dashboard=enable_dashboard
    )
    
    await runner.setup()
    
    try:
        await runner.run()
    except KeyboardInterrupt:
        print("\nShutting down by user request...")
    except Exception as e:
        logger.error("runner_crashed", error=str(e))
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
