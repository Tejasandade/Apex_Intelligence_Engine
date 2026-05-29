"""
Apex Intelligence Engine V5 — Backtest Engine
===============================================
Event-driven backtesting engine that replays historical data bar-by-bar,
runs the full feature → model → signal → execution pipeline, and tracks
portfolio performance.

Critically: NO lookahead bias. Each bar only sees data up to that point.

Usage:
    engine = BacktestEngine(cost_model=CostModel(commission_bps=2.0, slippage_bps=1.0))
    result = await engine.run(
        symbol="BTCUSDT",
        data=historical_df,
        model=trained_model,
        feature_store=FeatureStore("crypto"),
    )
    print(result.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.cost_model import CostModel, COST_MODELS
from src.backtester.metrics import PerformanceMetrics, compute_metrics, check_quality_gates
from src.backtester.portfolio import PortfolioTracker
from src.core.logging import get_logger

logger = get_logger("apex.backtester.engine")


@dataclass
class BacktestResult:
    """Complete backtest output."""

    symbol: str
    market_type: str
    metrics: PerformanceMetrics
    equity_curve: pd.Series
    trades: list[dict]
    quality_gate_passed: bool
    quality_gate_failures: dict[str, tuple[float, float]]
    cost_model_name: str
    config: dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    Sequential event-driven backtesting engine.

    Replays historical data bar-by-bar:
    1. For each bar, compute features using data UP TO that bar (no lookahead)
    2. Feed features to the model for prediction
    3. Apply signal thresholds to decide BUY/SELL/HOLD
    4. Execute trades through the portfolio tracker (with cost model)
    5. Track equity curve and compute performance metrics

    The engine supports running multiple cost scenarios to test sensitivity.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        conviction_threshold: float = 0.60,
        position_size_pct: float = 0.02,
        default_rr: float = 2.0,
        max_holding_bars: int = 20,
        max_concurrent_positions: int = 1,
        scale_out_at_rr: float = 1.0,
        warmup_bars: int = 60,
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting portfolio value.
            conviction_threshold: Minimum model probability to trigger a trade.
            position_size_pct: Position size as fraction of capital (risk per trade).
            default_rr: Default risk:reward ratio for stop/target calculation.
            max_holding_bars: Maximum bars before time-based exit.
            max_concurrent_positions: Max simultaneous open positions.
            scale_out_at_rr: R:R level to trigger partial profit taking.
            warmup_bars: Number of initial bars to skip (for indicator warmup).
        """
        self.initial_capital = initial_capital
        self.conviction_threshold = conviction_threshold
        self.position_size_pct = position_size_pct
        self.default_rr = default_rr
        self.max_holding_bars = max_holding_bars
        self.max_concurrent_positions = max_concurrent_positions
        self.scale_out_at_rr = scale_out_at_rr
        self.warmup_bars = warmup_bars

    async def run(
        self,
        symbol: str,
        data: pd.DataFrame,
        model: Any,  # BaseModel — uses .predict() method
        feature_store: Any,  # FeatureStore
        cost_model: CostModel | None = None,
        market_type: str = "crypto",
    ) -> BacktestResult:
        """
        Run a full backtest on historical data.

        Args:
            symbol: Trading symbol.
            data: Historical OHLCV DataFrame.
            model: Trained model with a .predict(features_df) -> float method.
            feature_store: FeatureStore instance for feature computation.
            cost_model: Transaction cost model (defaults to "expected").
            market_type: Market type for feature engineering.

        Returns:
            BacktestResult with full metrics and trade history.
        """
        cost = cost_model or COST_MODELS["expected"]
        portfolio = PortfolioTracker(
            initial_capital=self.initial_capital,
            cost_model=cost,
            max_concurrent_positions=self.max_concurrent_positions,
            scale_out_at_rr=self.scale_out_at_rr,
        )

        n_bars = len(data)
        if n_bars < self.warmup_bars + 10:
            logger.error(
                "backtest_insufficient_data",
                symbol=symbol,
                bars=n_bars,
                required=self.warmup_bars + 10,
            )
            return self._empty_result(symbol, market_type, cost)

        logger.info(
            "backtest_started",
            symbol=symbol,
            bars=n_bars,
            warmup=self.warmup_bars,
            capital=self.initial_capital,
            cost_model=f"{cost.commission_bps}+{cost.slippage_bps} bps",
        )

        # Pre-compute all features in one vectorised pass (efficient)
        all_features = feature_store.build_features(data)

        signals_generated = 0
        trades_attempted = 0

        # Bar-by-bar simulation (starting after warmup)
        for i in range(self.warmup_bars, n_bars):
            bar = data.iloc[i]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            # Step 1: Update portfolio with current bar (check stops, targets, expiry)
            portfolio.update_bar(bar_index=i, high=high, low=low, close=close)

            # Step 2: Get model prediction using features UP TO this bar (no lookahead)
            if i >= len(all_features):
                continue

            feature_row = all_features.iloc[[i]]

            # Skip if features are all zeros (insufficient data for indicators)
            if feature_row.iloc[0].abs().sum() < 1e-6:
                continue

            try:
                probability = model.predict(feature_row)
            except Exception as e:
                logger.debug("backtest_predict_error", bar=i, error=str(e))
                continue

            # Step 3: Signal decision
            if probability >= self.conviction_threshold and portfolio.can_open_position():
                signals_generated += 1
                atr = float(all_features.iloc[i].get("ATR", 0.0))

                if atr <= 0:
                    continue

                # FIXED-FRACTION sizing: risk a fixed % of INITIAL capital
                # (not current equity — prevents runaway compounding)
                risk_amount = self.initial_capital * self.position_size_pct
                stop_distance = atr * 1.5  # 1.5x ATR stop
                quantity = risk_amount / stop_distance if stop_distance > 0 else 0

                # Cap max notional at 10x initial capital (leverage limit)
                max_notional = self.initial_capital * 10.0
                max_qty = max_notional / close if close > 0 else 0
                quantity = min(quantity, max_qty)

                if quantity <= 0:
                    continue

                stop_loss = close - stop_distance
                take_profit = close + (stop_distance * self.default_rr)

                pos = portfolio.open_position(
                    symbol=symbol,
                    side="BUY",
                    price=close,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    bar_index=i,
                    max_holding_bars=self.max_holding_bars,
                )

                if pos is not None:
                    trades_attempted += 1

            elif probability <= (1.0 - self.conviction_threshold) and portfolio.can_open_position():
                signals_generated += 1
                atr = float(all_features.iloc[i].get("ATR", 0.0))

                if atr <= 0:
                    continue

                risk_amount = self.initial_capital * self.position_size_pct
                stop_distance = atr * 1.5
                quantity = risk_amount / stop_distance if stop_distance > 0 else 0

                max_notional = self.initial_capital * 10.0
                max_qty = max_notional / close if close > 0 else 0
                quantity = min(quantity, max_qty)

                if quantity <= 0:
                    continue

                stop_loss = close + stop_distance
                take_profit = close - (stop_distance * self.default_rr)

                pos = portfolio.open_position(
                    symbol=symbol,
                    side="SELL",
                    price=close,
                    quantity=quantity,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    bar_index=i,
                    max_holding_bars=self.max_holding_bars,
                )

                if pos is not None:
                    trades_attempted += 1

        # Compute metrics
        equity_curve = portfolio.get_equity_curve()
        trade_list = portfolio.get_trade_list()
        metrics = compute_metrics(equity_curve, trade_list)

        # Check quality gates
        passed, failures = check_quality_gates(metrics)

        cost_name = "custom"
        for name, cm in COST_MODELS.items():
            if cm.commission_bps == cost.commission_bps and cm.slippage_bps == cost.slippage_bps:
                cost_name = name
                break

        logger.info(
            "backtest_completed",
            symbol=symbol,
            total_bars=n_bars,
            signals=signals_generated,
            trades_executed=len(trade_list),
            sharpe=f"{metrics.sharpe_ratio:.2f}",
            max_dd=f"{metrics.max_drawdown_pct:.1f}%",
            total_return=f"{metrics.total_return_pct:.1f}%",
            win_rate=f"{metrics.win_rate:.1%}",
            profit_factor=f"{metrics.profit_factor:.2f}",
            quality_gate="PASS" if passed else "FAIL",
        )

        return BacktestResult(
            symbol=symbol,
            market_type=market_type,
            metrics=metrics,
            equity_curve=equity_curve,
            trades=trade_list,
            quality_gate_passed=passed,
            quality_gate_failures=failures,
            cost_model_name=cost_name,
            config={
                "initial_capital": self.initial_capital,
                "conviction_threshold": self.conviction_threshold,
                "position_size_pct": self.position_size_pct,
                "default_rr": self.default_rr,
                "max_holding_bars": self.max_holding_bars,
                "warmup_bars": self.warmup_bars,
            },
        )

    async def run_cost_sensitivity(
        self,
        symbol: str,
        data: pd.DataFrame,
        model: Any,
        feature_store: Any,
        market_type: str = "crypto",
    ) -> dict[str, BacktestResult]:
        """
        Run the backtest across all cost model scenarios.
        The strategy must be profitable at WORST-CASE costs.

        Returns:
            Dict mapping cost model name to BacktestResult.
        """
        results: dict[str, BacktestResult] = {}

        for name, cost_model in COST_MODELS.items():
            logger.info("cost_sensitivity_run", scenario=name, cost=f"{cost_model.round_trip_cost_pct:.4%}")
            result = await self.run(
                symbol=symbol,
                data=data,
                model=model,
                feature_store=feature_store,
                cost_model=cost_model,
                market_type=market_type,
            )
            results[name] = result

        # Summary
        logger.info("cost_sensitivity_summary")
        for name, result in results.items():
            logger.info(
                f"  {name:15s} | Sharpe: {result.metrics.sharpe_ratio:6.2f} | "
                f"Return: {result.metrics.total_return_pct:7.1f}% | "
                f"MaxDD: {result.metrics.max_drawdown_pct:5.1f}% | "
                f"Gate: {'PASS' if result.quality_gate_passed else 'FAIL'}"
            )

        return results

    def _empty_result(self, symbol: str, market_type: str, cost: CostModel) -> BacktestResult:
        """Return an empty result for insufficient data."""
        return BacktestResult(
            symbol=symbol,
            market_type=market_type,
            metrics=PerformanceMetrics(),
            equity_curve=pd.Series(dtype=float),
            trades=[],
            quality_gate_passed=False,
            quality_gate_failures={"data": (0, 1)},
            cost_model_name="N/A",
        )
