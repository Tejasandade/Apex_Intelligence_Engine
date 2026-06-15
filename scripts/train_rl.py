"""
Apex Intelligence Engine V5 — RL Meta-Controller Trainer
==========================================================
Trains the PPO Agent using historical paper trading logs.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import DATA_DIR
from src.core.logging import get_logger
from src.models.rl_agent import RLMetaController

logger = get_logger("apex.scripts.train_rl")


def train_rl(symbol: str, market: str, steps: int):
    # Load historical signal logs from paper trading
    log_path = DATA_DIR / "logs" / "signals.jsonl"
    
    historical_trades = []
    
    if log_path.exists():
        with open(log_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("symbol") == symbol:
                        # pnl_pct is already inside data from backtester!
                        historical_trades.append(data)
                except Exception:
                    pass
                    
    print(f"Loaded {len(historical_trades)} historical trades for RL Training.")
    
    if len(historical_trades) < 100:
        print("Not enough historical trades to train RL! Need at least 100.")
        print("Please run the Backtester or Live Paper Trader for longer.")
        return

    rl = RLMetaController(market_type=market, symbol=symbol)
    
    print(f"Training PPO Meta-Controller for {steps} timesteps...")
    rl.train(historical_trades=historical_trades, total_timesteps=steps)
    
    print("RL Agent trained and saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--market", type=str, default="crypto")
    parser.add_argument("--steps", type=int, default=100000)
    args = parser.parse_args()
    
    train_rl(args.symbol, args.market, args.steps)
