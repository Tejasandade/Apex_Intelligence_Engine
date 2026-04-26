import os
from datetime import datetime, timedelta

from loguru import logger


TRADE_COOLDOWN_MINUTES = 5


class RiskManager:
    """
    Governs capital, drawdown protection, and whether signals require
    manual approval or can flow through an autonomous execution path.
    """

    def __init__(
        self,
        total_capital: float = 10000.0,
        max_risk_per_trade: float = 0.02,
        daily_drawdown_limit: float = 0.05,
        professional_trader_enabled: bool = True,
    ):
        self.total_capital = total_capital
        self.starting_capital = total_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.daily_drawdown_limit = daily_drawdown_limit
        self.professional_trader_enabled = professional_trader_enabled
        self.circuit_breaker_tripped = False
        self.price_gap_alert_pct = float(os.getenv("APEX_PRICE_GAP_ALERT_PCT", "0.0025"))
        self.force_price_gap_override = (
            os.getenv("APEX_FORCE_PRICE_GAP_OVERRIDE", "True").lower() in {"true", "1", "yes"}
        )
        self.trade_allocation = 0.0
        # Per-symbol cooldown registry: symbol -> datetime when cooldown expires
        self._cooldown_until: dict[str, datetime] = {}
        logger.info(
            "Initialized RiskManager with ${:.2f} capital, {:.1f}% max risk, mode={}.",
            self.total_capital,
            self.max_risk_per_trade * 100,
            self.execution_mode,
        )

    @property
    def autonomous_mode(self) -> bool:
        return self.professional_trader_enabled

    @autonomous_mode.setter
    def autonomous_mode(self, enabled: bool):
        self.professional_trader_enabled = enabled

    @property
    def execution_mode(self) -> str:
        return "AUTONOMOUS" if self.autonomous_mode else "MANUAL_APPROVAL"

    @property
    def requires_manual_approval(self) -> bool:
        return not self.autonomous_mode

    def set_professional_trader_mode(self, enabled: bool) -> str:
        self.autonomous_mode = enabled
        logger.info("Professional Trader mode set to {}", self.execution_mode)
        return self.execution_mode

    # ------------------------------------------------------------------
    # Cooldown & Multi-Trade Helpers
    # ------------------------------------------------------------------

    def _get_confidence_tier(self, probability: float) -> int:
        if probability > 0.80 or probability < 0.20:
            return 3
        if probability > 0.65 or probability < 0.35:
            return 2
        if probability > 0.55 or probability < 0.45:
            return 1
        return 0

    def record_trade_closed(self, symbol: str) -> None:
        """Arms a post-trade cooldown for *symbol* to prevent revenge trading."""
        expires_at = datetime.utcnow() + timedelta(minutes=TRADE_COOLDOWN_MINUTES)
        self._cooldown_until[symbol] = expires_at
        logger.info(
            "Trade cooldown armed for {} | no new entries until {} UTC ({}m window)",
            symbol,
            expires_at.strftime("%H:%M:%S"),
            TRADE_COOLDOWN_MINUTES,
        )

    def is_in_cooldown(self, symbol: str) -> bool:
        """Returns True while the symbol is inside its post-trade cooldown window."""
        expires_at = self._cooldown_until.get(symbol)
        if expires_at is None:
            return False
        return datetime.utcnow() < expires_at

    def cooldown_remaining_seconds(self, symbol: str) -> int:
        """Returns the integer seconds remaining in the cooldown, or 0 if not active."""
        expires_at = self._cooldown_until.get(symbol)
        if expires_at is None:
            return 0
        remaining = (expires_at - datetime.utcnow()).total_seconds()
        return max(0, int(remaining))

    def update_capital(self, total_capital: float):
        self.total_capital = total_capital
        logger.info("RiskManager capital updated to ${:.2f}", self.total_capital)

    def check_circuit_breaker(self, current_balance: float):
        """
        Trips the circuit breaker if the daily drawdown limit is exceeded.
        """
        self.total_capital = current_balance
        if self.circuit_breaker_tripped:
            return

        drawdown = (self.starting_capital - current_balance) / self.starting_capital
        if drawdown >= self.daily_drawdown_limit:
            self.circuit_breaker_tripped = True
            logger.warning("CRITICAL: Daily drawdown limit reached. Trading halted.")

    def evaluate_trade_signal(
        self,
        symbol: str,
        bullish_probability: float,
        signal_price: float | None = None,
        market_price: float | None = None,
        active_trades: dict | None = None,
    ) -> dict:
        """
        Evaluates a model signal and reports whether it can route directly
        or must wait for manual confirmation.

        Guards (evaluated in order):
        1. ACTIVE_TRADE_BLOCK  — a position for this symbol is already open.
        2. COOLDOWN_BLOCK      — 5-minute post-exit window has not elapsed.
        3. CIRCUIT_BREAKER     — daily drawdown limit exceeded.
        """
        price_gap = 0.0
        price_gap_pct = 0.0
        price_gap_override = False
        if (
            signal_price is not None
            and market_price is not None
            and float(signal_price) > 0
            and float(market_price) > 0
        ):
            signal_price = float(signal_price)
            market_price = float(market_price)
            price_gap = round(market_price - signal_price, 4)
            price_gap_pct = round(abs(price_gap) / market_price, 6)
            price_gap_override = (
                price_gap_pct >= self.price_gap_alert_pct and self.force_price_gap_override
            )
            log_fn = logger.warning if price_gap_pct >= self.price_gap_alert_pct else logger.info
            log_fn(
                "Price Gap | {} | signal_price={:.4f} | market_price={:.4f} | gap={:.4f} ({:.4%}) | override={}",
                symbol,
                signal_price,
                market_price,
                price_gap,
                price_gap_pct,
                price_gap_override,
            )

        # ── Guard 1: Block new signals based on active trades & tiers ────────
        confidence_tier = self._get_confidence_tier(bullish_probability)
        if active_trades:
            active_symbol_trades = [
                t for t in active_trades.values()
                if t.get("symbol") == symbol and t.get("is_active", True)
            ]
            
            if len(active_symbol_trades) >= 3:
                logger.debug("Signal suppressed for {} — max trades (3) reached.", symbol)
                return {
                    "symbol": symbol,
                    "action": "HOLD",
                    "probability": bullish_probability,
                    "signal_price": signal_price,
                    "market_price": market_price,
                    "price_gap": price_gap,
                    "price_gap_pct": price_gap_pct,
                    "price_gap_override": price_gap_override,
                    "requires_manual_approval": False,
                    "execution_mode": "ACTIVE_TRADE_BLOCK",
                    "block_reason": "MAX_TRADES",
                    "confidence_tier": confidence_tier,
                }

            if confidence_tier > 0:
                occupied_tiers = [t.get("confidence_tier", 1) for t in active_symbol_trades]
                if confidence_tier in occupied_tiers:
                    logger.debug(
                        "Signal suppressed for {} — active trade with tier {} already open.",
                        symbol, confidence_tier
                    )
                    return {
                        "symbol": symbol,
                        "action": "HOLD",
                        "probability": bullish_probability,
                        "signal_price": signal_price,
                        "market_price": market_price,
                        "price_gap": price_gap,
                        "price_gap_pct": price_gap_pct,
                        "price_gap_override": price_gap_override,
                        "requires_manual_approval": False,
                        "execution_mode": "ACTIVE_TRADE_BLOCK",
                        "block_reason": f"TIER_{confidence_tier}_OCCUPIED",
                        "confidence_tier": confidence_tier,
                    }

        # ── Guard 2: Block new signals during the post-trade cooldown ─────────
        if self.is_in_cooldown(symbol):
            remaining = self.cooldown_remaining_seconds(symbol)
            logger.debug(
                "Signal suppressed for {} — cooldown active ({} s remaining).",
                symbol,
                remaining,
            )
            return {
                "symbol": symbol,
                "action": "HOLD",
                "probability": bullish_probability,
                "signal_price": signal_price,
                "market_price": market_price,
                "price_gap": price_gap,
                "price_gap_pct": price_gap_pct,
                "price_gap_override": price_gap_override,
                "requires_manual_approval": False,
                "execution_mode": "COOLDOWN_BLOCK",
                "block_reason": f"COOLDOWN:{remaining}s",
                "confidence_tier": confidence_tier,
            }

        # ── Guard 3: Circuit-breaker (daily drawdown limit) ───────────────────
        if self.circuit_breaker_tripped:
            return {
                "symbol": symbol,
                "action": "HOLD",
                "probability": bullish_probability,
                "signal_price": signal_price,
                "market_price": market_price,
                "price_gap": price_gap,
                "price_gap_pct": price_gap_pct,
                "price_gap_override": price_gap_override,
                "requires_manual_approval": True,
                "execution_mode": "CIRCUIT_BREAKER",
                "confidence_tier": confidence_tier,
            }

        logger.debug(
            f"Evaluating signal for {symbol}: Bullish Prob = {bullish_probability}"
        )

        action = "HOLD"
        if bullish_probability > 0.65:
            action = "BUY"
        elif bullish_probability < 0.35:
            action = "SELL"

        return {
            "symbol": symbol,
            "action": action,
            "probability": bullish_probability,
            "signal_price": signal_price,
            "market_price": market_price,
            "price_gap": price_gap,
            "price_gap_pct": price_gap_pct,
            "price_gap_override": price_gap_override,
            "requires_manual_approval": self.requires_manual_approval and action != "HOLD",
            "execution_mode": self.execution_mode,
            "confidence_tier": confidence_tier,
        }

    def calculate_position_size(
        self,
        symbol: str,
        action: str,
        confidence: float,
        reward_risk_ratio: float = 1.5,
        kelly_fraction: float = 0.5,
        high_confidence_confluence: bool = False,
    ) -> float:
        """
        Calculates the exact position size based on current capital, risk limits,
        and signal confidence using a fractional Kelly criterion.
        """
        if action == "HOLD":
            return 0.0

        if self.trade_allocation > 0:
            final_allocation = min(self.trade_allocation, self.total_capital)
            logger.debug(f"Using fixed user trade allocation: ${final_allocation:.2f}")
            return round(final_allocation, 2)

        logger.debug(
            f"Calculating position size for {action} on {symbol} (Confidence: {confidence})"
        )

        probability = confidence
        kelly_pct = probability - ((1.0 - probability) / reward_risk_ratio)
        if kelly_pct <= 0:
            logger.info("Kelly percentage is negative (no statistical edge). Aborting trade.")
            return 0.0

        confluence_multiplier = 1.35 if high_confidence_confluence else 1.0
        adjusted_kelly_fraction = kelly_fraction * confluence_multiplier
        raw_allocation = self.total_capital * (kelly_pct * adjusted_kelly_fraction)
        max_allowed_risk = self.total_capital * self.max_risk_per_trade * confluence_multiplier
        final_allocation = min(raw_allocation, max_allowed_risk)

        logger.debug(
            "Kelly %: {:.2f} | Kelly Fraction: {:.2f} | Confluence: {} | Raw Allocation: ${:.2f} | Capped Allocation: ${:.2f}",
            kelly_pct,
            adjusted_kelly_fraction,
            high_confidence_confluence,
            raw_allocation,
            final_allocation,
        )

        return round(final_allocation, 2)


if __name__ == "__main__":
    risk_manager = RiskManager(
        total_capital=10000.0,
        max_risk_per_trade=0.05,
        daily_drawdown_limit=0.05,
    )

    risk_manager.check_circuit_breaker(9400.0)

    signal = risk_manager.evaluate_trade_signal("BTCUSDT", 0.85)
    size = risk_manager.calculate_position_size(
        signal["symbol"],
        signal["action"],
        signal["probability"],
        reward_risk_ratio=1.5,
        kelly_fraction=0.5,
    )

    logger.info(
        "Final Decision: {} {} with size ${} ({})",
        signal["action"],
        signal["symbol"],
        size,
        signal["execution_mode"],
    )
