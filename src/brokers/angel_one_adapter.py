"""
Angel One Dry-Run Broker Adapter
=================================
Implements BaseBrokerAdapter for the Angel One SmartAPI platform.

SAFETY CONTRACT
---------------
live_trading_enabled is HARDCODED to False. This adapter will NEVER
place a real order regardless of what is passed in at runtime.
All execute_order / close_position calls are logged and return a
mock DRY_RUN response so the rest of the engine can operate normally.
"""

import os
import random
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from loguru import logger

from src.brokers.base_adapter import BaseBrokerAdapter

IST = ZoneInfo("Asia/Kolkata")


class AngelOneAdapter(BaseBrokerAdapter):
    """
    Paper-trading adapter for Angel One SmartAPI.

    Connects to Angel One for market data and account info, but all
    order execution is simulated — no real capital is risked.
    """

    # ── Safety hard-lock ──────────────────────────────────────────────────────
    live_trading_enabled: bool = False   # MUST NEVER be set to True here

    def __init__(
        self,
        paper_balance: float = 100_000.0,   # INR starting balance
    ):
        super().__init__(mode="DRY_RUN", paper_balance=paper_balance)

        self.api_key   = os.getenv("ANGEL_API_KEY", "")
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "")
        self.pin       = os.getenv("ANGEL_PIN", "")
        self.totp_key  = os.getenv("ANGEL_TOTP_KEY", "")

        self._smart   = None          # SmartConnect client (set on connect)
        self._session = None          # session token cache

        self._fill_counter = 0        # monotonic fill ID for mock responses

        logger.info(
            "AngelOneAdapter created | mode=DRY_RUN | paper_balance={:,.0f} INR | "
            "live_trading_enabled={}",
            self.paper_balance,
            self.live_trading_enabled,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self):
        """Authenticate with Angel One SmartConnect (read-only / data access)."""
        try:
            from SmartApi import SmartConnect
            import pyotp
        except ImportError:
            logger.warning(
                "smartapi-python or pyotp not installed. "
                "AngelOneAdapter running in offline mode."
            )
            return

        try:
            totp_val = pyotp.TOTP(self.totp_key).now() if self.totp_key else ""
            self._smart = SmartConnect(api_key=self.api_key)
            data = self._smart.generateSession(self.client_id, self.pin, totp_val)
            if data.get("status"):
                self._session = data["data"]
                logger.success(
                    "AngelOneAdapter connected as {} (DRY_RUN \u2014 no live orders)",
                    self.client_id,
                )
            else:
                logger.error("AngelOneAdapter auth failed: {}", data.get("message"))
                self._smart = None
        except Exception as exc:
            logger.error("AngelOneAdapter connect error: {}", exc)
            self._smart = None

    async def disconnect(self):
        if self._smart and self._session:
            try:
                self._smart.terminateSession(self.client_id)
                logger.info("AngelOneAdapter session terminated.")
            except Exception:
                pass
        self._smart = None
        self._session = None

    # ── Account ───────────────────────────────────────────────────────────────

    async def get_account_balance(self) -> float:
        """
        Returns the paper balance (DRY_RUN).
        In a live adapter this would call SmartAPI RMS / fund details.
        """
        return self.paper_balance

    # ── Order Execution (DRY_RUN ONLY) ───────────────────────────────────────

    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        **kwargs,
    ) -> Optional[dict]:
        """
        SAFETY: live_trading_enabled is False — this NEVER places a real order.
        Logs the intended execution and returns a mock DRY_RUN fill response.

        Slippage model (Epic 26):
        - BUY  → fill at best_ask + random [0.01%, 0.05%] penalty  (cross the spread aggressively)
        - SELL → fill at best_bid - random [0.01%, 0.05%] penalty  (accept the bid, minus impact)
        Falls back to the provided limit_price / price kwarg when LOB data is unavailable.
        """
        if self.live_trading_enabled:
            # Defensive guard — should never be reachable
            raise RuntimeError(
                "AngelOneAdapter: live_trading_enabled must remain False. "
                "Do not enable real trading through this adapter."
            )

        self._fill_counter += 1
        fill_id = f"DRYRUN-AO-{self._fill_counter:06d}"

        # ── Spread-Crossing Slippage Simulation ─────────────────────────────
        best_bid   = float(kwargs.get("best_bid", 0.0))
        best_ask   = float(kwargs.get("best_ask", 0.0))
        slip_pct   = random.uniform(0.0001, 0.0005)   # 0.01% – 0.05%

        side_upper = side.upper()
        if side_upper == "BUY" and best_ask > 0:
            simulated_price = round(best_ask * (1.0 + slip_pct), 4)
        elif side_upper == "SELL" and best_bid > 0:
            simulated_price = round(best_bid * (1.0 - slip_pct), 4)
        else:
            # Fallback: use limit_price / price kwarg, no slippage applied
            simulated_price = float(kwargs.get("limit_price", kwargs.get("price", 0.0)))
        # ────────────────────────────────────────────────────────────────────

        logger.info(
            "[DRY_RUN] AngelOneAdapter | {} {} {} @ {:.4f} | order_type={} | slip={:.4%} | fill_id={}",
            side_upper, quantity, symbol.upper(),
            simulated_price, order_type, slip_pct, fill_id,
        )

        return {
            "broker_status":    "DRY_RUN",
            "adapter":          "AngelOneAdapter",
            "fill_id":          fill_id,
            "symbol":           symbol.upper(),
            "side":             side_upper,
            "quantity":         quantity,
            "order_type":       order_type,
            "simulated_price":  simulated_price,
            "slippage_pct":     round(slip_pct * 100, 4),
            "timestamp":        datetime.now(IST).isoformat(),
            "live_executed":    False,
        }

    async def get_open_positions(self, symbol: str) -> list[dict]:
        """Returns empty list in DRY_RUN (no real positions)."""
        return []

    async def set_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float,
        **kwargs,
    ) -> Optional[dict]:
        """Logs the intended trailing stop and returns a mock response."""
        logger.info(
            "[DRY_RUN] AngelOneAdapter trailing stop | {} {} {} | callback={:.2f}%",
            side.upper(), quantity, symbol.upper(), callback_rate,
        )
        return {
            "broker_status":   "DRY_RUN",
            "adapter":         "AngelOneAdapter",
            "symbol":          symbol.upper(),
            "trailing_stop":   True,
            "callback_rate":   callback_rate,
            "live_executed":   False,
        }
