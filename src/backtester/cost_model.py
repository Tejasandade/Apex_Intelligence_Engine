"""
Apex Intelligence Engine V5 — Transaction Cost Model
=====================================================
Models the real-world costs of executing trades:
- Commission fees (maker/taker)
- Slippage (market impact)
- Spread costs

Backtests must prove profitability at WORST-CASE costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.logging import get_logger

logger = get_logger("apex.backtester.cost_model")


@dataclass
class CostModel:
    """
    Transaction cost model for backtesting.

    Costs are in basis points (bps). 1 bps = 0.01%.
    Total cost per round-trip = 2 × (commission + slippage).
    """

    commission_bps: float = 2.0     # Exchange fee per side
    slippage_bps: float = 1.0       # Market impact per side

    @property
    def one_way_cost_pct(self) -> float:
        """Total one-way cost as a decimal fraction."""
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    @property
    def round_trip_cost_pct(self) -> float:
        """Total round-trip cost (entry + exit) as a decimal fraction."""
        return 2.0 * self.one_way_cost_pct

    def apply_entry_cost(self, price: float) -> float:
        """
        Adjust entry price for slippage + commission (buy higher, sell lower).
        For a BUY: effective price is HIGHER.
        """
        return price * (1.0 + self.one_way_cost_pct)

    def apply_exit_cost(self, price: float) -> float:
        """
        Adjust exit price for slippage + commission (sell lower, buy higher).
        For a SELL to close: effective price is LOWER.
        """
        return price * (1.0 - self.one_way_cost_pct)

    def compute_net_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        side: str = "BUY",
    ) -> float:
        """
        Compute net PnL after all costs.

        Args:
            entry_price: Raw entry price.
            exit_price: Raw exit price.
            quantity: Position size in base currency units.
            side: "BUY" for long, "SELL" for short.

        Returns:
            Net profit/loss after costs.
        """
        effective_entry = self.apply_entry_cost(entry_price)
        effective_exit = self.apply_exit_cost(exit_price)

        if side == "BUY":
            gross_pnl = (effective_exit - effective_entry) * quantity
        else:
            gross_pnl = (effective_entry - effective_exit) * quantity

        return gross_pnl


# ── Preset Cost Models ──────────────────────────────────────────────────────
COST_MODELS = {
    "optimistic": CostModel(commission_bps=2.0, slippage_bps=0.5),
    "expected": CostModel(commission_bps=2.0, slippage_bps=1.0),
    "conservative": CostModel(commission_bps=4.0, slippage_bps=2.0),
    "worst_case": CostModel(commission_bps=4.0, slippage_bps=3.0),
}
