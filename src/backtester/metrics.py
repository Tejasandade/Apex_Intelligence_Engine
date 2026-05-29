"""
Apex Intelligence Engine V5 — Performance Metrics
===================================================
Quantitative performance metrics for backtesting and live monitoring.
Every metric a quant fund uses to evaluate strategy viability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.core.logging import get_logger

logger = get_logger("apex.backtester.metrics")


@dataclass
class PerformanceMetrics:
    """Complete performance metrics for a backtest or live trading period."""

    # Returns
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_bars: int = 0
    avg_drawdown_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_rr_realized: float = 0.0
    expectancy: float = 0.0  # Average $ per trade
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Holding periods
    avg_holding_bars: float = 0.0
    avg_winning_holding_bars: float = 0.0
    avg_losing_holding_bars: float = 0.0

    # Consistency
    monthly_returns: dict[str, float] = field(default_factory=dict)
    positive_months: int = 0
    negative_months: int = 0


def compute_metrics(
    equity_curve: pd.Series,
    trades: list[dict],
    risk_free_rate: float = 0.0,
    bars_per_year: int = 525_600,  # 1-minute bars in a year
) -> PerformanceMetrics:
    """
    Compute comprehensive performance metrics from an equity curve and trade list.

    Args:
        equity_curve: Series indexed by bar number with portfolio value.
        trades: List of trade dicts with keys: pnl, pnl_pct, bars_held, side.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 0).
        bars_per_year: Number of bars per year for annualization.

    Returns:
        PerformanceMetrics dataclass with all computed values.
    """
    metrics = PerformanceMetrics()

    if equity_curve.empty or len(trades) == 0:
        return metrics

    # ── Returns ──────────────────────────────────────────────────────────────
    initial_value = equity_curve.iloc[0]
    final_value = equity_curve.iloc[-1]
    total_bars = len(equity_curve)

    metrics.total_return_pct = (
        (final_value - initial_value) / initial_value * 100
        if initial_value > 0
        else 0.0
    )

    # Annualized return — only meaningful with 90+ days of data
    years = total_bars / bars_per_year
    if years >= 0.25 and initial_value > 0:
        # Only annualize if we have at least ~3 months of data
        raw_annual = ((final_value / initial_value) ** (1.0 / years) - 1.0) * 100
        metrics.annualized_return_pct = min(raw_annual, 999999.0)  # Cap display
    elif years > 0 and initial_value > 0:
        # For shorter periods, extrapolate linearly (more conservative)
        daily_return = metrics.total_return_pct / (years * 365)
        metrics.annualized_return_pct = daily_return * 365

    # ── Risk Metrics (Sharpe, Sortino, Calmar) ──────────────────────────────
    returns = equity_curve.pct_change().dropna()

    if len(returns) > 1 and returns.std() > 0:
        excess_return = returns.mean() - (risk_free_rate / bars_per_year)
        metrics.sharpe_ratio = (
            excess_return / returns.std() * np.sqrt(bars_per_year)
        )

        # Sortino: only penalize downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics.sortino_ratio = (
                excess_return / downside_returns.std() * np.sqrt(bars_per_year)
            )

    # ── Drawdown Analysis ────────────────────────────────────────────────────
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    metrics.max_drawdown_pct = abs(float(drawdown.min())) * 100

    # Max drawdown duration
    is_in_dd = drawdown < 0
    dd_groups = (~is_in_dd).cumsum()
    if is_in_dd.any():
        dd_lengths = is_in_dd.groupby(dd_groups).sum()
        metrics.max_drawdown_duration_bars = int(dd_lengths.max())

    # Average drawdown
    dd_values = drawdown[drawdown < 0]
    if len(dd_values) > 0:
        metrics.avg_drawdown_pct = abs(float(dd_values.mean())) * 100

    # Calmar ratio
    if metrics.max_drawdown_pct > 0:
        metrics.calmar_ratio = metrics.annualized_return_pct / metrics.max_drawdown_pct

    # ── Trade Statistics ─────────────────────────────────────────────────────
    metrics.total_trades = len(trades)
    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t.get("pnl_pct", 0.0) for t in trades]
    bars_held = [t.get("bars_held", 0) for t in trades]

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    metrics.winning_trades = len(winners)
    metrics.losing_trades = len(losers)
    metrics.win_rate = len(winners) / len(pnls) if pnls else 0.0

    # Profit factor
    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    metrics.profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    # Average win/loss
    win_pcts = [p for p in pnl_pcts if p > 0]
    loss_pcts = [p for p in pnl_pcts if p <= 0]
    metrics.avg_win_pct = np.mean(win_pcts) * 100 if win_pcts else 0.0
    metrics.avg_loss_pct = np.mean(loss_pcts) * 100 if loss_pcts else 0.0

    # Average R:R realized
    if metrics.avg_loss_pct != 0:
        metrics.avg_rr_realized = abs(metrics.avg_win_pct / metrics.avg_loss_pct)

    # Expectancy
    metrics.expectancy = np.mean(pnls) if pnls else 0.0

    # Extremes
    metrics.largest_win = max(pnls) if pnls else 0.0
    metrics.largest_loss = min(pnls) if pnls else 0.0

    # Holding periods
    metrics.avg_holding_bars = np.mean(bars_held) if bars_held else 0.0

    winning_bars = [b for p, b in zip(pnls, bars_held) if p > 0]
    losing_bars = [b for p, b in zip(pnls, bars_held) if p <= 0]
    metrics.avg_winning_holding_bars = (
        np.mean(winning_bars) if winning_bars else 0.0
    )
    metrics.avg_losing_holding_bars = (
        np.mean(losing_bars) if losing_bars else 0.0
    )

    return metrics


def check_quality_gates(
    metrics: PerformanceMetrics,
    min_sharpe: float = 1.5,
    max_drawdown_pct: float = 15.0,
    min_win_rate: float = 0.40,
    min_profit_factor: float = 1.3,
    min_trades: int = 200,
) -> tuple[bool, dict[str, tuple[float, float]]]:
    """
    Check if performance metrics pass the quality gates.

    Args:
        metrics: Computed performance metrics.
        min_sharpe: Minimum required Sharpe ratio.
        max_drawdown_pct: Maximum allowed drawdown percentage.
        min_win_rate: Minimum required win rate.
        min_profit_factor: Minimum required profit factor.
        min_trades: Minimum required number of trades.

    Returns:
        Tuple of (passed: bool, failures: dict mapping metric name to (actual, required)).
    """
    failures: dict[str, tuple[float, float]] = {}

    if metrics.sharpe_ratio < min_sharpe:
        failures["sharpe_ratio"] = (metrics.sharpe_ratio, min_sharpe)

    if metrics.max_drawdown_pct > max_drawdown_pct:
        failures["max_drawdown_pct"] = (metrics.max_drawdown_pct, max_drawdown_pct)

    if metrics.win_rate < min_win_rate:
        failures["win_rate"] = (metrics.win_rate, min_win_rate)

    if metrics.profit_factor < min_profit_factor:
        failures["profit_factor"] = (metrics.profit_factor, min_profit_factor)

    if metrics.total_trades < min_trades:
        failures["total_trades"] = (float(metrics.total_trades), float(min_trades))

    passed = len(failures) == 0

    if passed:
        logger.info(
            "quality_gates_passed",
            sharpe=f"{metrics.sharpe_ratio:.2f}",
            max_dd=f"{metrics.max_drawdown_pct:.1f}%",
            win_rate=f"{metrics.win_rate:.1%}",
            profit_factor=f"{metrics.profit_factor:.2f}",
            trades=metrics.total_trades,
        )
    else:
        logger.warning(
            "quality_gates_failed",
            failures={k: f"{v[0]:.4f} (required: {v[1]:.4f})" for k, v in failures.items()},
        )

    return passed, failures
