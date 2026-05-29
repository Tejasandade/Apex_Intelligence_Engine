"""
Apex Intelligence Engine V5 — Backtest Runner
===============================================
End-to-end backtest: download data → train model → run backtest → report.

Usage:
    python -m scripts.run_backtest
    python -m scripts.run_backtest --symbol btcusdt
    python -m scripts.run_backtest --symbol btcusdt --cost-sensitivity
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtester.cost_model import COST_MODELS
from src.backtester.engine import BacktestEngine
from src.backtester.reports import (
    generate_text_report,
    print_cost_sensitivity_summary,
    save_report,
)
from src.core.config import DATA_DIR, get_market, get_model_params
from src.core.logging import get_logger
from src.data.providers.binance import BinanceHistoricalProvider
from src.data.validation.validator import DataValidator
from src.features.store import FeatureStore
from src.models.training.trainer import UnifiedTrainer

logger = get_logger("apex.scripts.run_backtest")


async def run_full_backtest(
    symbol: str = "btcusdt",
    market_type: str = "crypto",
    total_bars: int = 50000,
    cost_sensitivity: bool = False,
) -> None:
    """
    Run the complete backtest pipeline:
    1. Download/load historical data
    2. Train model with purged CV
    3. Run backtest with cost modeling
    4. Generate and save report
    """

    # ── Step 1: Load or Download Data ────────────────────────────────────────
    data_path = DATA_DIR / "historical" / "crypto" / f"{symbol.lower()}.parquet"

    if data_path.exists():
        import pandas as pd

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

        # Validate and cache
        validator = DataValidator()
        data = validator.validate_dataframe(data)

        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(data_path, index=False)

    logger.info("data_ready", rows=len(data))

    # ── Step 2: Train Model ──────────────────────────────────────────────────
    logger.info("training_model", symbol=symbol, market_type=market_type)
    trainer = UnifiedTrainer(market_type=market_type, symbol=symbol)
    training_results = await trainer.run(data=data, save_models=True)

    # Get the trained model (use trending model for backtest)
    model = trainer.get_model("trending")
    if model is None:
        model = trainer.get_model("ranging")
    if model is None:
        model = trainer.get_model("all")
    if model is None:
        logger.error("no_model_trained", results=training_results)
        return

    logger.info("model_trained", model=model.model_name)

    # ── Step 3: Run Backtest ─────────────────────────────────────────────────
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
        # Run across all cost scenarios
        results = await engine.run_cost_sensitivity(
            symbol=symbol,
            data=data,
            model=model,
            feature_store=feature_store,
            market_type=market_type,
        )

        # Print summary
        summary = print_cost_sensitivity_summary(results)
        print(summary)

        # Save individual reports
        for name, result in results.items():
            save_report(result)
            report_text = generate_text_report(result)
            print(report_text)

    else:
        # Single run with expected costs
        result = await engine.run(
            symbol=symbol,
            data=data,
            model=model,
            feature_store=feature_store,
            cost_model=COST_MODELS["expected"],
            market_type=market_type,
        )

        # Print and save report
        report_text = generate_text_report(result)
        print(report_text)
        save_report(result)

        # Print verdict
        if result.quality_gate_passed:
            print("\n  [PASS] QUALITY GATE: PASSED — Strategy has a proven edge.\n")
        else:
            print("\n  [FAIL] QUALITY GATE: FAILED — Strategy needs improvement.\n")
            for metric, (actual, required) in result.quality_gate_failures.items():
                print(f"    • {metric}: {actual:.4f} (required: {required:.4f})")

    # ── Step 4: Feature Importance ───────────────────────────────────────────
    importance = model.get_feature_importance()
    if importance:
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\n  TOP 10 FEATURE IMPORTANCE:")
        print("  " + "-" * 45)
        for feat, imp in sorted_imp[:10]:
            bar = "█" * int(imp * 200)
            print(f"  {feat:30s} {imp:.4f} {bar}")
        print()


async def main():
    parser = argparse.ArgumentParser(description="Run Apex backtest pipeline")
    parser.add_argument(
        "--symbol",
        type=str,
        default="btcusdt",
        help="Symbol to backtest (default: btcusdt)",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="crypto",
        choices=["crypto", "india_equity"],
        help="Market type",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=50000,
        help="Number of 1-minute bars (default: 50000)",
    )
    parser.add_argument(
        "--cost-sensitivity",
        action="store_true",
        help="Run cost sensitivity analysis across all scenarios",
    )

    args = parser.parse_args()
    await run_full_backtest(
        symbol=args.symbol,
        market_type=args.market,
        total_bars=args.bars,
        cost_sensitivity=args.cost_sensitivity,
    )


if __name__ == "__main__":
    asyncio.run(main())
