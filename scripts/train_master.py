"""
Apex Intelligence Engine V5 — Master Training Pipeline
======================================================
Executes the full Institutional Phase 2 Training Pipeline:
1. Optuna Hyperparameter Tuning (with Purged CV)
2. XGBoost Ensemble Training
3. Time-Series Transformer Training
4. Simulated Trade Generation (Backtest)
5. PPO RL Meta-Controller Training
"""

import argparse
import subprocess
import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

def run_step(name: str, cmd: list[str]) -> bool:
    print("\n" + "=" * 60)
    print(f"STARTING PHASE: {name}")
    print("=" * 60)
    
    # We set PYTHONPATH to the root directory to avoid ModuleNotFoundErrors
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    
    try:
        # Launch as a subprocess to ensure clean RAM after each phase
        result = subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
        print(f"\n[SUCCESS] PHASE COMPLETE: {name}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] PHASE FAILED: {name} (Exit code {e.returncode})\n")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--market", type=str, default="crypto")
    parser.add_argument("--trials", type=int, default=15, help="Optuna trials for XGBoost")
    args = parser.parse_args()

    python_exe = sys.executable

    # Phase 1: Tune XGBoost Hyperparameters
    if not run_step("XGBoost Hyperparameter Tuning", [
        python_exe, "-u", "scripts/tune_models.py",
        "--symbol", args.symbol,
        "--market", args.market,
        "--trials", str(args.trials)
    ]):
        return

    # Phase 2: Train XGBoost Ensemble
    if not run_step("XGBoost Ensemble Training", [
        python_exe, "-u", "scripts/train_all.py",
        "--symbol", args.symbol,
        "--market", args.market
    ]):
        return

    # Phase 3: Train Time-Series Transformer
    if not run_step("Time-Series Transformer Training", [
        python_exe, "-u", "scripts/train_transformer.py",
        "--symbol", args.symbol,
        "--market", args.market
    ]):
        return

    # Phase 4: Generate RL Trading Data (via Backtester)
    if not run_step("RL Trade Data Generation", [
        python_exe, "-u", "scripts/run_backtest.py",
        "--symbol", args.symbol,
        "--market", args.market,
        "--bars", "10000"  # We don't need the full dataset, just enough trades
    ]):
        return

    # Phase 5: Train PPO RL Meta-Controller
    if not run_step("RL Meta-Controller Training", [
        python_exe, "-u", "scripts/train_rl.py",
        "--symbol", args.symbol,
        "--market", args.market,
        "--steps", "50000"
    ]):
        return
        
    print("\n" + "=" * 60)
    print("APEX INTELLIGENCE ENGINE: ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
