"""
Apex Intelligence Engine V5 — TRUE Out-of-Sample Backtest
===========================================================
The CORRECT way to validate a trading strategy:

1. Split data into TRAIN (first 60%) and TEST (last 40%)
2. Train model ONLY on TRAIN data
3. Backtest ONLY on TEST data (model has NEVER seen this)
4. This gives REALISTIC performance expectations

The previous run_backtest.py trained and tested on the same data.
This script fixes that critical flaw.

Usage:
    python -m scripts.run_backtest_oos --symbol btcusdt
    python -m scripts.run_backtest_oos --symbol btcusdt --cost-sensitivity
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtester.cost_model import COST_MODELS
from src.backtester.engine import BacktestEngine
from src.backtester.reports import (
    generate_text_report,
    print_cost_sensitivity_summary,
    save_report,
)
from src.core.config import DATA_DIR, get_model_params
from src.core.logging import get_logger
from src.data.providers.binance import BinanceHistoricalProvider
from src.data.validation.validator import DataValidator
from src.features.store import FeatureStore
from src.models.training.trainer import UnifiedTrainer

logger = get_logger("apex.scripts.backtest_oos")

# ── Split ratios ────────────────────────────────────────────────────────────
TRAIN_PCT = 0.60  # First 60% for training
TEST_PCT = 0.40   # Last 40% for testing (out-of-sample)


async def run_oos_backtest(
    symbol: str = "btcusdt",
    market_type: str = "crypto",
    total_bars: int = 50000,
    cost_sensitivity: bool = False,
) -> None:
    """
    Run a TRUE out-of-sample backtest.

    The model is trained ONLY on the first 60% of data.
    The backtest runs ONLY on the last 40% — data the model has NEVER seen.
    """

    # ── Step 1: Load Data ────────────────────────────────────────────────────
    data_path = DATA_DIR / "historical" / "crypto" / f"{symbol.lower()}.parquet"

    if data_path.exists():
        logger.info("loading_cached_data", path=str(data_path))
        data = pd.read_parquet(data_path)
    else:
        logger.info("downloading_data", symbol=symbol, bars=total_bars)
        provider = BinanceHistoricalProvider()
        await provider.connect()
        data = await provider.fetch_historical_candles(
            symbol=symbol.upper(), interval="1m", limit=total_bars
        )
        await provider.disconnect()

        if data.empty:
            logger.error("no_data_available", symbol=symbol)
            return

        validator = DataValidator()
        data = validator.validate_dataframe(data)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(data_path, index=False)

    n_total = len(data)
    split_idx = int(n_total * TRAIN_PCT)

    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"  OUT-OF-SAMPLE BACKTEST — {symbol.upper()}")
    print(f"{'=' * 60}")
    print(f"  Total bars:    {n_total:,}")
    print(f"  Train set:     {len(train_data):,} bars (first {TRAIN_PCT:.0%})")
    print(f"  Test set:      {len(test_data):,} bars (last {TEST_PCT:.0%})")
    print(f"  Train period:  {pd.to_datetime(train_data['timestamp'].iloc[0], unit='ms')} "
          f"-> {pd.to_datetime(train_data['timestamp'].iloc[-1], unit='ms')}")
    print(f"  Test period:   {pd.to_datetime(test_data['timestamp'].iloc[0], unit='ms')} "
          f"-> {pd.to_datetime(test_data['timestamp'].iloc[-1], unit='ms')}")
    print(f"{'=' * 60}\n")

    # ── Step 2: Train ONLY on train data ─────────────────────────────────────
    logger.info(
        "oos_training",
        symbol=symbol,
        train_bars=len(train_data),
        test_bars=len(test_data),
    )
    print("  [1/4] Training model on TRAIN set only...")

    trainer = UnifiedTrainer(market_type=market_type, symbol=symbol)
    training_results = await trainer.run(data=train_data, save_models=True)

    model = trainer.get_model("trending")
    if model is None:
        model = trainer.get_model("ranging")
    if model is None:
        model = trainer.get_model("all")
    if model is None:
        logger.error("no_model_trained", results=training_results)
        print("  ERROR: No model could be trained. Insufficient data.")
        return

    # Print training summary
    for regime, result in training_results.items():
        if isinstance(result, dict) and "cv_metrics" in result:
            cv = result["cv_metrics"]
            print(f"  Model [{regime}]: CV Accuracy={cv['accuracy']:.4f}, "
                  f"LogLoss={cv['log_loss']:.4f}, Precision={cv['precision']:.4f}")

    print(f"  [2/4] Model trained: {model.model_name}")
    print()

    # ── Step 3: Backtest ONLY on test data ───────────────────────────────────
    print("  [3/4] Backtesting on UNSEEN test data...")

    feature_store = FeatureStore(market_type)
    model_params = get_model_params(market_type)
    labeling_cfg = model_params.get("labeling", {})

    engine = BacktestEngine(
        initial_capital=10_000.0,
        conviction_threshold=0.60,
        position_size_pct=0.02,
        default_rr=2.0,
        max_holding_bars=labeling_cfg.get("max_holding_bars", 20),
        max_concurrent_positions=1,
        warmup_bars=60,
    )

    if cost_sensitivity:
        results = await engine.run_cost_sensitivity(
            symbol=symbol,
            data=test_data,
            model=model,
            feature_store=feature_store,
            market_type=market_type,
        )

        print("\n  [4/4] Results:\n")
        summary = print_cost_sensitivity_summary(results)
        print(summary)

        for name, result in results.items():
            save_report(result, f"oos_backtest_{symbol}_{name}.txt")
            report_text = generate_text_report(result)
            print(report_text)

    else:
        result = await engine.run(
            symbol=symbol,
            data=test_data,
            model=model,
            feature_store=feature_store,
            cost_model=COST_MODELS["expected"],
            market_type=market_type,
        )

        print("\n  [4/4] Results:\n")
        report_text = generate_text_report(result)
        print(report_text)
        save_report(result, f"oos_backtest_{symbol}_expected.txt")

        if result.quality_gate_passed:
            print("\n  QUALITY GATE: PASSED — Strategy has edge on UNSEEN data.\n")
        else:
            print("\n  QUALITY GATE: FAILED — Strategy needs improvement.\n")
            for metric, (actual, required) in result.quality_gate_failures.items():
                print(f"    {metric}: {actual:.4f} (required: {required:.4f})")

    # ── Feature Importance ───────────────────────────────────────────────────
    importance = model.get_feature_importance()
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\n  TOP 10 FEATURE IMPORTANCE:")
        print("  " + "-" * 45)
        for feat, imp in sorted_imp[:10]:
            bar = "#" * int(imp * 200)
            print(f"  {feat:30s} {imp:.4f} {bar}")
        print()


async def main():
    parser = argparse.ArgumentParser(
        description="Run TRUE out-of-sample backtest (train/test split)"
    )
    parser.add_argument("--symbol", type=str, default="btcusdt")
    parser.add_argument("--market", type=str, default="crypto",
                        choices=["crypto", "india_equity"])
    parser.add_argument("--bars", type=int, default=50000)
    parser.add_argument("--cost-sensitivity", action="store_true")

    args = parser.parse_args()
    await run_oos_backtest(
        symbol=args.symbol,
        market_type=args.market,
        total_bars=args.bars,
        cost_sensitivity=args.cost_sensitivity,
    )


if __name__ == "__main__":
    asyncio.run(main())
