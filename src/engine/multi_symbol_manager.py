"""
# APEX v8.0 — Track C
Apex Intelligence Engine v8.0 — Multi-Symbol Manager
======================================================
Manages N independent symbol instances with capital isolation,
exposure cap enforcement, and correlation-based direction filters.

Design:
  - Each symbol gets its own LiveSignalEngine + candle buffers + portfolio
  - Equal capital split across symbols (3 symbols → 33% each)
  - Total exposure across ALL symbols capped at max_total_exposure_pct
  - Correlation filter blocks same-direction trades on correlated pairs
"""

from __future__ import annotations

import time
import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.core.logging import get_logger
from src.engine.signal_engine import LiveSignalEngine, TradingSignal

logger = get_logger("apex.engine.multi_symbol")

# APEX v8.0 — Track C: Correlated pair definitions
# Pairs that share strong positive correlation and should not both be open
# in the same direction simultaneously.
CORRELATED_PAIRS: list[tuple[str, str]] = [
    ("btcusdt", "ethusdt"),
]


def _are_correlated(sym_a: str, sym_b: str) -> bool:
    """Return True if two symbols are defined as correlated."""
    a, b = sym_a.lower(), sym_b.lower()
    for x, y in CORRELATED_PAIRS:
        if (a == x and b == y) or (a == y and b == x):
            return True
    return False


@dataclass
class SymbolState:
    """
    Per-symbol state container.
    Holds an independent portfolio, account slice, signal engine,
    and candle buffers for one traded symbol.
    """

    symbol: str
    # Portfolio — mirrors apex_v7.py's portfolio dict
    portfolio: dict = field(default_factory=lambda: {
        "position": "FLAT",
        "direction": None,
        "entry_price": 0.0,
        "entry_time": 0.0,
        "sl_price": 0.0,
        "sl_distance": 0.0,
        "tp1_price": 0.0,
        "tp2_price": 0.0,
        "notional": 0.0,
        "initial_notional": 0.0,
        "unrealized_pnl": 0.0,
        "banked_pnl": 0.0,
        "phase": 0,
        "signal_id": "",
        "prob": 0.0,
        "council_score": 0.0,
        "regime_entry": "",
        "source": "",
        "highest_fav": 0.0,
    })
    # Capital allocated to this symbol (a slice of total_capital)
    account_slice: dict = field(default_factory=lambda: {
        "balance": 0.0,
        "peak": 0.0,
        "drawdown_pct": 0.0,
        "net_pnl": 0.0,
        "total": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "best": 0.0,
        "worst": 0.0,
        "total_wins_usd": 0.0,
        "total_losses_usd": 0.0,
        "consecutive_losses": 0,
        "trades_today": 0,
        "today_date": "",
        "last_trade_time": 0.0,
        "last_loss_time": 0.0,
        "halted": False,
        "halt_reason": "",
        "halt_until": 0.0,
    })
    # ML signal engine (independent per-symbol instance)
    signal_engine: LiveSignalEngine | None = None
    # Candle buffers
    candles_5m: deque = field(default_factory=lambda: deque(maxlen=210))
    candles_1h: deque = field(default_factory=lambda: deque(maxlen=168))
    # Live model probabilities (updated every 5m candle close)
    live_probs: dict = field(default_factory=lambda: {
        "long": 0.5, "short": 0.5, "updated": False
    })
    # Last signal emitted by the engine for this symbol
    last_signal: TradingSignal | None = None
    # Current notional open in this symbol (tracked externally for exposure calc)
    _open_notional: float = 0.0

    def __post_init__(self):
        # Ensure today_date is initialised
        self.account_slice["today_date"] = datetime.datetime.now().strftime("%Y-%m-%d")

    @property
    def is_open(self) -> bool:
        return self.portfolio["position"] == "OPEN"

    @property
    def open_direction(self) -> str | None:
        """'LONG', 'SHORT', or None."""
        if self.is_open:
            return self.portfolio.get("direction")
        return None


class MultiSymbolManager:
    """
    # APEX v8.0 — Track C
    Manages N symbols independently with capital isolation.

    Key design:
      - Each symbol gets its own LiveSignalEngine instance
      - Each symbol has an independent portfolio + partial capital allocation
      - Total exposure across symbols capped at max_total_exposure_pct
      - Correlation filter: won't open same-direction trades on correlated pairs
    """

    def __init__(
        self,
        symbols: list[str],
        total_capital: float,
        max_total_exposure_pct: float = 0.40,
    ):
        """
        Args:
            symbols: List of symbol keys (e.g. ["btcusdt", "ethusdt"]).
            total_capital: Total capital pool to split across all symbols.
            max_total_exposure_pct: Maximum fraction of total_capital that may
                                    be simultaneously at risk across ALL symbols.
        """
        if not symbols:
            raise ValueError("MultiSymbolManager requires at least one symbol.")

        self.symbols = [s.lower() for s in symbols]
        self.total_capital = total_capital
        self.max_total_exposure_pct = max_total_exposure_pct

        # Per-symbol capital slice
        self._capital_per_symbol: float = total_capital / len(self.symbols)

        # Build SymbolState for every symbol
        self._states: dict[str, SymbolState] = {}
        for sym in self.symbols:
            state = SymbolState(symbol=sym)
            alloc = self.get_capital_for_symbol(sym)
            state.account_slice["balance"] = alloc
            state.account_slice["peak"] = alloc
            self._states[sym] = state
            logger.info(
                "symbol_state_created",
                symbol=sym,
                capital=alloc,
            )

        logger.info(
            "multi_symbol_manager_ready",
            symbols=self.symbols,
            total_capital=total_capital,
            max_exposure_pct=max_total_exposure_pct,
            capital_per_symbol=self._capital_per_symbol,
        )

    # ── Capital Allocation ────────────────────────────────────────────────────

    def get_capital_for_symbol(self, symbol: str) -> float:
        """
        Equal capital split across symbols.
        e.g. 3 symbols → 33.3% each of total_capital.
        """
        return round(self.total_capital / len(self.symbols), 2)

    # ── Position Gate ─────────────────────────────────────────────────────────

    def can_open_position(self, symbol: str, direction: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason).

        Blocks if:
        1. Total exposure would exceed max_total_exposure_pct after this trade.
        2. Another correlated symbol already has a same-direction trade open.
        """
        sym = symbol.lower()
        dir_upper = direction.upper()

        # 1. Exposure cap
        current_exp = self.total_exposure_pct()
        # We approximate a new position as one full capital_per_symbol slice.
        additional = self._capital_per_symbol / self.total_capital
        if (current_exp + additional) > self.max_total_exposure_pct:
            msg = (
                f"Exposure cap: {current_exp:.1%} + {additional:.1%} "
                f"> {self.max_total_exposure_pct:.1%} max"
            )
            logger.info(
                "position_blocked_exposure",
                symbol=sym,
                direction=dir_upper,
                current=f"{current_exp:.2%}",
                cap=f"{self.max_total_exposure_pct:.2%}",
            )
            return False, msg

        # 2. Correlation filter
        for other_sym, state in self._states.items():
            if other_sym == sym:
                continue
            if not _are_correlated(sym, other_sym):
                continue
            if state.is_open and state.open_direction == dir_upper:
                msg = (
                    f"Correlated pair block: {other_sym.upper()} already "
                    f"{dir_upper} — skipping {sym.upper()}"
                )
                logger.info(
                    "position_blocked_correlation",
                    symbol=sym,
                    correlated_with=other_sym,
                    direction=dir_upper,
                )
                return False, msg

        return True, "CLEAR"

    # ── Bookkeeping ───────────────────────────────────────────────────────────

    def register_open(self, symbol: str, direction: str, notional: float) -> None:
        """Call when a position is opened for a symbol."""
        sym = symbol.lower()
        if sym in self._states:
            self._states[sym]._open_notional = notional
            logger.debug(
                "position_registered_open",
                symbol=sym,
                direction=direction,
                notional=notional,
            )

    def register_close(self, symbol: str, pnl: float) -> None:
        """Call when a position is closed for a symbol."""
        sym = symbol.lower()
        if sym in self._states:
            self._states[sym]._open_notional = 0.0
            logger.debug(
                "position_registered_close",
                symbol=sym,
                pnl=pnl,
            )

    # ── Metrics ───────────────────────────────────────────────────────────────

    def total_exposure_pct(self) -> float:
        """
        Returns current total exposure as a fraction of total_capital.
        Computed as sum(open_notionals) / total_capital.
        """
        total_open = sum(s._open_notional for s in self._states.values())
        return total_open / self.total_capital if self.total_capital > 0 else 0.0

    def get_symbol_state(self, symbol: str) -> SymbolState:
        """Return the SymbolState for a given symbol key."""
        sym = symbol.lower()
        if sym not in self._states:
            raise KeyError(
                f"Unknown symbol '{symbol}'. "
                f"Available: {list(self._states.keys())}"
            )
        return self._states[sym]

    def all_states(self) -> dict[str, SymbolState]:
        """Return all symbol states."""
        return dict(self._states)

    def attach_signal_engine(
        self, symbol: str, engine: LiveSignalEngine
    ) -> None:
        """Attach a pre-built LiveSignalEngine to a SymbolState."""
        sym = symbol.lower()
        if sym not in self._states:
            raise KeyError(f"Unknown symbol '{symbol}'.")
        self._states[sym].signal_engine = engine
        logger.info("signal_engine_attached", symbol=sym)

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of all symbol states."""
        result: dict[str, Any] = {
            "total_capital": self.total_capital,
            "exposure_pct": round(self.total_exposure_pct(), 4),
            "max_exposure_pct": self.max_total_exposure_pct,
            "symbols": {},
        }
        for sym, state in self._states.items():
            result["symbols"][sym] = {
                "position": state.portfolio["position"],
                "direction": state.portfolio.get("direction"),
                "unrealized_pnl": state.portfolio.get("unrealized_pnl", 0.0),
                "balance": state.account_slice.get("balance", 0.0),
                "net_pnl": state.account_slice.get("net_pnl", 0.0),
                "open_notional": state._open_notional,
                "last_signal": (
                    {
                        "direction": state.last_signal.direction,
                        "conviction": state.last_signal.conviction,
                        "regime": state.last_signal.regime,
                    }
                    if state.last_signal
                    else None
                ),
            }
        return result
