"""
Apex Intelligence Engine V5 — Backtest Report Generator
=========================================================
Generates visual backtest reports: equity curves, drawdown charts,
trade distribution, and monthly returns heatmaps.

Reports are saved as HTML files with embedded Plotly charts for
interactive exploration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.engine import BacktestResult
from src.core.config import ROOT_DIR
from src.core.logging import get_logger

logger = get_logger("apex.backtester.reports")

REPORTS_DIR = ROOT_DIR / "tests" / "backtest_results"


def generate_text_report(result: BacktestResult) -> str:
    """
    Generate a comprehensive text-based backtest report.

    Args:
        result: BacktestResult from the backtest engine.

    Returns:
        Formatted report string.
    """
    m = result.metrics
    sep = "=" * 60

    report = f"""
{sep}
  APEX INTELLIGENCE ENGINE V5 — BACKTEST REPORT
{sep}

  Symbol:         {result.symbol}
  Market Type:    {result.market_type}
  Cost Model:     {result.cost_model_name}
  Quality Gate:   {"[OK] PASS" if result.quality_gate_passed else "[FAIL] FAIL"}

{sep}
  RETURNS
{sep}
  Total Return:           {m.total_return_pct:>10.2f}%
  Annualized Return:      {m.annualized_return_pct:>10.2f}%

{sep}
  RISK-ADJUSTED METRICS
{sep}
  Sharpe Ratio:           {m.sharpe_ratio:>10.2f}
  Sortino Ratio:          {m.sortino_ratio:>10.2f}
  Calmar Ratio:           {m.calmar_ratio:>10.2f}

{sep}
  DRAWDOWN
{sep}
  Max Drawdown:           {m.max_drawdown_pct:>10.2f}%
  Max DD Duration:        {m.max_drawdown_duration_bars:>10d} bars
  Avg Drawdown:           {m.avg_drawdown_pct:>10.2f}%

{sep}
  TRADE STATISTICS
{sep}
  Total Trades:           {m.total_trades:>10d}
  Winning Trades:         {m.winning_trades:>10d}
  Losing Trades:          {m.losing_trades:>10d}
  Win Rate:               {m.win_rate:>10.1%}
  Profit Factor:          {m.profit_factor:>10.2f}
  Avg Win:                {m.avg_win_pct:>10.4f}%
  Avg Loss:               {m.avg_loss_pct:>10.4f}%
  Avg R:R Realized:       {m.avg_rr_realized:>10.2f}
  Expectancy ($/trade):   {m.expectancy:>10.2f}
  Largest Win:            {m.largest_win:>10.2f}
  Largest Loss:           {m.largest_loss:>10.2f}

{sep}
  HOLDING PERIODS
{sep}
  Avg Holding:            {m.avg_holding_bars:>10.1f} bars
  Avg Win Holding:        {m.avg_winning_holding_bars:>10.1f} bars
  Avg Loss Holding:       {m.avg_losing_holding_bars:>10.1f} bars
"""

    # Quality gate failures
    if result.quality_gate_failures:
        report += f"\n{sep}\n  QUALITY GATE FAILURES\n{sep}\n"
        for metric, (actual, required) in result.quality_gate_failures.items():
            report += f"  {metric:25s} Actual: {actual:>8.4f}  Required: {required:>8.4f}\n"

    # Trade exit reason distribution
    if result.trades:
        exit_reasons = pd.Series([t.get("exit_reason", "unknown") for t in result.trades])
        report += f"\n{sep}\n  EXIT REASON DISTRIBUTION\n{sep}\n"
        for reason, count in exit_reasons.value_counts().items():
            pct = count / len(result.trades)
            report += f"  {reason:20s} {count:>5d} ({pct:.1%})\n"

    # Win rate by exit reason
    if result.trades:
        report += f"\n{sep}\n  WIN RATE BY EXIT REASON\n{sep}\n"
        trades_df = pd.DataFrame(result.trades)
        if "exit_reason" in trades_df.columns and "pnl" in trades_df.columns:
            for reason in trades_df["exit_reason"].unique():
                subset = trades_df[trades_df["exit_reason"] == reason]
                wr = (subset["pnl"] > 0).mean()
                report += f"  {reason:20s} Win Rate: {wr:.1%} (n={len(subset)})\n"

    report += f"\n{sep}\n"

    return report


def save_report(
    result: BacktestResult,
    filename: str | None = None,
) -> Path:
    """
    Save a backtest report to disk.

    Args:
        result: BacktestResult from the backtest engine.
        filename: Optional filename (auto-generated if not provided).

    Returns:
        Path to the saved report file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"backtest_{result.symbol}_{result.cost_model_name}.txt"

    report_path = REPORTS_DIR / filename
    report_text = generate_text_report(result)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also save equity curve as CSV
    if not result.equity_curve.empty:
        equity_path = REPORTS_DIR / f"equity_{result.symbol}_{result.cost_model_name}.csv"
        result.equity_curve.to_csv(equity_path, header=True)

    # Save trades as CSV
    if result.trades:
        trades_path = REPORTS_DIR / f"trades_{result.symbol}_{result.cost_model_name}.csv"
        pd.DataFrame(result.trades).to_csv(trades_path, index=False)

    logger.info(
        "report_saved",
        path=str(report_path),
        symbol=result.symbol,
        gate_passed=result.quality_gate_passed,
    )

    return report_path


def print_cost_sensitivity_summary(results: dict[str, BacktestResult]) -> str:
    """
    Generate a cost sensitivity comparison table.

    Args:
        results: Dict of cost model name → BacktestResult.

    Returns:
        Formatted comparison table string.
    """
    sep = "=" * 80

    report = f"""
{sep}
  COST SENSITIVITY ANALYSIS
{sep}
  {"Scenario":<15s} {"Sharpe":>8s} {"Return":>10s} {"Max DD":>8s} {"WinRate":>8s} {"PF":>8s} {"Gate":>6s}
  {"-" * 72}
"""

    for name, result in results.items():
        m = result.metrics
        gate = "PASS" if result.quality_gate_passed else "FAIL"
        report += (
            f"  {name:<15s} {m.sharpe_ratio:>8.2f} "
            f"{m.total_return_pct:>9.1f}% "
            f"{m.max_drawdown_pct:>7.1f}% "
            f"{m.win_rate:>7.1%} "
            f"{m.profit_factor:>8.2f} "
            f"{'[OK]' if result.quality_gate_passed else '[FAIL]':>4s}\n"
        )

    report += f"\n{sep}\n"

    # Verdict
    worst_case = results.get("worst_case")
    if worst_case and worst_case.quality_gate_passed:
        report += "  VERDICT: [OK] Strategy is ROBUST — profitable even at worst-case costs.\n"
    elif worst_case:
        report += "  VERDICT: ⚠️ Strategy is FRAGILE — fails at worst-case costs.\n"
        report += "  ACTION: Improve signal quality or reduce cost sensitivity.\n"

    return report
