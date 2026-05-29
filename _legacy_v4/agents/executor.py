import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.brokers.base_adapter import BaseBrokerAdapter
from src.brokers.binance_adapter import BinanceAdapter
from dotenv import load_dotenv
from loguru import logger

load_dotenv()


class TradeExecutor:
    """
    The Trade Execution Integration module.
    Responsible for securely connecting to the exchange API, monitoring websockets,
    translating signals into orders, and persisting the active trade blotter.
    """

    def __init__(self, live_trading_enabled: bool = False, broker_adapter: Optional[BaseBrokerAdapter] = None):
        self.live_trading_enabled = live_trading_enabled
        self._state_lock = asyncio.Lock()

        mode = "LIVE" if live_trading_enabled else "DRY_RUN"
        self.broker = broker_adapter or BinanceAdapter(mode=mode)

        root_dir = Path(__file__).resolve().parents[2]
        self.state_file = Path(
            os.getenv(
                "APEX_ACTIVE_TRADES_FILE",
                str(root_dir / "data" / "runtime" / "active_trades.json"),
            )
        )
        self.active_trades: dict[str, dict] = self._load_active_trades()

        if self.live_trading_enabled:
            logger.warning("LIVE TRADING IS ENABLED. Real orders will be executed on the exchange.")
        else:
            logger.info("TradeExecutor initialized in DRY RUN mode. No real orders will be placed.")

    def _normalize_trade_record(self, order_id: str, trade: dict) -> dict:
        normalized = dict(trade)
        status = str(normalized.get("status", "OPEN")).upper()
        normalized["order_id"] = normalized.get("order_id", order_id)
        normalized["status"] = status
        normalized["is_active"] = bool(normalized.get("is_active", status != "CLOSED"))
        normalized["broker_status"] = normalized.get("broker_status", "DRY_RUN")
        normalized["exit_reason"] = normalized.get("exit_reason")
        normalized["scaled_out"] = bool(normalized.get("scaled_out", False))
        normalized["breakeven_triggered"] = bool(normalized.get("breakeven_triggered", False))
        return normalized

    async def connect(self):
        """Initializes the connection to the broker API."""
        await self.broker.connect()

    async def disconnect(self):
        """Gracefully closes the connection."""
        await self.broker.disconnect()

    def _load_active_trades(self) -> dict[str, dict]:
        try:
            if self.state_file.exists():
                with self.state_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    return {
                        order_id: self._normalize_trade_record(order_id, trade)
                        for order_id, trade in payload.items()
                        if isinstance(trade, dict)
                    }
        except Exception as exc:
            logger.warning(f"Failed to load active trade state: {exc}")
        return {}

    async def _persist_active_trades(self):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with self.state_file.open("w", encoding="utf-8") as handle:
                json.dump(self.active_trades, handle, indent=2)
        except Exception as exc:
            logger.error(f"Failed to persist active trades: {exc}")

    def list_active_trades(self, include_inactive: bool = False) -> list[dict]:
        trades = self.active_trades.values()
        if not include_inactive:
            trades = [trade for trade in trades if trade.get("is_active", True)]

        return sorted(
            trades,
            key=lambda trade: trade.get("opened_at", ""),
            reverse=True,
        )

    def sync_live_prices(self, symbol: str, current_price: float):
        current_price = float(current_price)
        if current_price <= 0:
            return

        for trade in self.active_trades.values():
            if not trade.get("is_active", True):
                continue
            if trade.get("symbol") != symbol:
                continue

            entry_price = float(trade.get("entry_price", 0.0))
            side = trade.get("side", "BUY").upper()
            callback_rate = float(trade.get("callback_rate", 1.0))

            trade["current_price"] = round(current_price, 4)
            if entry_price > 0:
                quantity = float(trade.get("quantity", 0.0))
                if side == "BUY":
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    pnl_value = (current_price - entry_price) * quantity
                    trade["peak_price"] = round(
                        max(float(trade.get("peak_price", entry_price)), current_price),
                        4,
                    )
                    trailing_stop = float(trade.get("peak_price", current_price)) * (1 - callback_rate / 100)
                else:
                    pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    pnl_value = (entry_price - current_price) * quantity
                    trade["trough_price"] = round(
                        min(float(trade.get("trough_price", entry_price)), current_price),
                        4,
                    )
                    trailing_stop = float(trade.get("trough_price", current_price)) * (1 + callback_rate / 100)

                if entry_price < (current_price * 0.5):
                    trade["broker_status"] = "CORRUPTED"
                    pnl_pct = 0.0
                    pnl_value = 0.0

                trade["pnl_pct"] = round(pnl_pct, 4)
                trade["pnl_value"] = round(pnl_value, 2)
                trade["trailing_stop_level"] = round(trailing_stop, 4)

    async def execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_id: Optional[str] = None,
        entry_price: Optional[float] = None,
        trailing_stop_level: Optional[float] = None,
        callback_rate: float = 1.0,
        confidence_tier: int = 1,
        order_type: str = "MARKET",
        best_bid: float = 0.0,
        best_ask: float = 0.0,
    ):
        """
        Executes a market or limit order on the configured broker.
        If live_trading_enabled is False, this is safely mocked as a DRY RUN.

        best_bid / best_ask are forwarded to the broker adapter for realistic
        spread-crossing slippage simulation (Epic 26).
        """
        side = side.upper()
        if side not in ["BUY", "SELL"]:
            logger.error(f"Invalid order side: {side}")
            return None

        order_id = order_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")

        if not self.live_trading_enabled:
            logger.info(
                "[DRY RUN] Would execute {} {} order for {} {} "
                "| bid={:.4f} | ask={:.4f}",
                side, order_type, quantity, symbol, best_bid, best_ask,
            )
            # Route through broker adapter so slippage simulation fires
            broker_result = await self.broker.execute_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                entry_price=entry_price,
                best_bid=best_bid,
                best_ask=best_ask,
            )
            fill_price = entry_price
            if broker_result:
                fill_price = broker_result.get("simulated_price") or broker_result.get("avgPrice") or entry_price
            await self._register_active_trade(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=fill_price,
                trailing_stop_level=trailing_stop_level,
                callback_rate=callback_rate,
                broker_status="DRY_RUN",
                confidence_tier=confidence_tier,
            )
            return {
                "status": "DRY_RUN",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "orderId": order_id,
                "order_type": order_type,
                "simulated_price": fill_price,
            }

        response = await self.broker.execute_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            entry_price=entry_price,
            best_bid=best_bid,
            best_ask=best_ask,
        )
        if response:
            logger.success(f"Order executed successfully: {response.get('orderId')}")
            await self._register_active_trade(
                order_id=str(response.get("orderId", order_id)),
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price or float(response.get("avgPrice", 0.0) or 0.0),
                trailing_stop_level=trailing_stop_level,
                callback_rate=callback_rate,
                broker_status="LIVE",
                confidence_tier=confidence_tier,
            )
        return response
    async def _register_active_trade(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: Optional[float],
        trailing_stop_level: Optional[float],
        callback_rate: float,
        broker_status: str,
        confidence_tier: int,
    ):
        async with self._state_lock:
            normalized_entry = round(float(entry_price or 0.0), 4)
            self.active_trades[order_id] = {
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "status": "OPEN",
                "is_active": True,
                "quantity": round(float(quantity), 6),
                "entry_price": normalized_entry,
                "current_price": normalized_entry,
                "pnl_pct": 0.0,
                "trailing_stop_level": round(float(trailing_stop_level or 0.0), 4),
                "callback_rate": round(float(callback_rate), 4),
                "broker_status": broker_status,
                "confidence_tier": confidence_tier,
                "exit_reason": None,
                "closing_requested_at": None,
                "closed_at": None,
                "opened_at": datetime.utcnow().isoformat(),
                "peak_price": normalized_entry,
                "trough_price": normalized_entry if normalized_entry > 0 else 0.0,
            }
            await self._persist_active_trades()

    def _resolve_exit_reason(
        self,
        current_probability: float,
        structure_break_signal: float,
    ) -> Optional[str]:
        if float(structure_break_signal) == -1.0:
            return "BEARISH_CHOCH"
        if float(current_probability) < 0.40:
            return "PROBABILITY_DROP"
        return None

    async def check_exit_conditions(
        self,
        symbol: str,
        current_probability: float,
        structure_break_signal: float,
        current_price: Optional[float] = None,
    ) -> list[dict]:
        """
        Marks active LONG positions for autonomous square-off when the model
        loses conviction or structure flips bearish. Also manages dynamic scaling and trailing.
        """
        await self.purge_stale_dry_run_trades()
        
        normalized_price = round(float(current_price or 0.0), 4)

        async with self._state_lock:
            for order_id, trade in self.active_trades.items():
                if not trade.get("is_active", True) or trade.get("symbol") != symbol:
                    continue

                entry_price = float(trade.get("entry_price", 0.0))
                original_stop_loss = float(trade.get("trailing_stop_level", 0.0))
                side = trade.get("side", "").upper()
                
                if normalized_price > 0:
                    trade["current_price"] = normalized_price

                # Scale-Out & Breakeven Logic (The 1:1 Trigger)
                if not trade.get("scaled_out", False) and normalized_price > 0 and entry_price > 0:
                    rr_multiple = 0.0
                    profit = 0.0
                    risk = 0.0
                    
                    if side == "BUY":
                        profit = normalized_price - entry_price
                        risk = entry_price - original_stop_loss
                    elif side == "SELL":
                        profit = entry_price - normalized_price
                        risk = original_stop_loss - entry_price
                        
                    if risk > 0:
                        rr_multiple = profit / risk
                        
                    if rr_multiple >= 1.0:
                        half_qty = round(float(trade.get("quantity", 0.0)) * 0.5, 3)
                        if half_qty > 0:
                            close_side = "SELL" if side == "BUY" else "BUY"
                            
                            logger.info(f"1:1 Target Hit! Scaling out 50% ({half_qty}) and trailing SL to Breakeven.")
                            
                            if self.live_trading_enabled:
                                await self.broker.execute_order(
                                    symbol=trade["symbol"],
                                    side=close_side,
                                    quantity=half_qty,
                                    order_type="MARKET"
                                )
                            else:
                                logger.info(f"[DRY RUN] Scaled out 50% via {close_side} MARKET for {half_qty}")

                            trade["quantity"] = float(trade.get("quantity", 0.0)) - half_qty
                            trade["trailing_stop_level"] = entry_price
                            trade["scaled_out"] = True
                            trade["breakeven_triggered"] = True
                            
            await self._persist_active_trades()
            
        exit_reason = self._resolve_exit_reason(current_probability, structure_break_signal)
        if exit_reason is None:
            return []

        exit_candidates = []
        marked_at = datetime.utcnow().isoformat()

        async with self._state_lock:
            for order_id, trade in self.active_trades.items():
                if not trade.get("is_active", True):
                    continue
                if trade.get("symbol") != symbol:
                    continue
                if trade.get("side", "").upper() != "BUY":
                    continue

                trade["status"] = "SQUARING OFF"
                trade["exit_reason"] = exit_reason
                trade["closing_requested_at"] = marked_at
                trade["last_probability"] = round(float(current_probability), 4)
                trade["last_structure_break_signal"] = float(structure_break_signal)
                if normalized_price > 0:
                    trade["current_price"] = normalized_price

                exit_candidates.append(
                    {
                        "order_id": order_id,
                        "symbol": trade["symbol"],
                        "side": trade["side"],
                        "quantity": float(trade.get("quantity", 0.0)),
                        "exit_reason": exit_reason,
                        "current_probability": round(float(current_probability), 4),
                        "structure_break_signal": float(structure_break_signal),
                        "current_price": float(trade.get("current_price", normalized_price)),
                    }
                )

            if exit_candidates:
                await self._persist_active_trades()

        if exit_candidates:
            logger.warning(
                "Autonomous exit triggered for {} active LONG trade(s) on {} | reason={} | probability={} | structure_break_signal={}",
                len(exit_candidates),
                symbol,
                exit_reason,
                round(float(current_probability), 4),
                float(structure_break_signal),
            )

        return exit_candidates

    async def close_trade(self, order_id: str, current_price: Optional[float] = None):
        return await self._close_trade_internal(
            order_id=order_id,
            current_price=current_price,
            exit_reason=None,
            triggered_by="manual_close",
        )

    async def _close_trade_internal(
        self,
        order_id: str,
        current_price: Optional[float],
        exit_reason: Optional[str],
        triggered_by: str,
    ):
        async with self._state_lock:
            trade = self.active_trades.get(order_id)
            if not trade or not trade.get("is_active", True):
                logger.warning(f"Active trade not found for close request: {order_id}")
                return None

            normalized_price = round(float(current_price or trade.get("current_price", 0.0)), 4)
            if trade.get("status") != "SQUARING OFF":
                trade["status"] = "SQUARING OFF"
                trade["closing_requested_at"] = datetime.utcnow().isoformat()
            if normalized_price > 0:
                trade["current_price"] = normalized_price
            if exit_reason:
                trade["exit_reason"] = exit_reason
            await self._persist_active_trades()

            side = trade.get("side", "BUY").upper()
            close_side = "SELL" if side == "BUY" else "BUY"
            symbol = trade["symbol"]
            quantity = float(trade.get("quantity", 0.0))

        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] Would close {symbol} order {order_id} via {close_side} MARKET for {quantity}")
            result = {"status": "DRY_RUN", "orderId": order_id}
        else:
            result = await self.broker.execute_order(
                symbol=symbol,
                side=close_side,
                quantity=quantity,
                order_type="MARKET"
            )
            if not result:
                await self._mark_exit_failed(order_id)
                logger.error(f"Failed to close trade {order_id} via broker.")
                return None

        async with self._state_lock:
            trade = self.active_trades.get(order_id)
            if not trade:
                logger.warning(f"Trade state disappeared before final close persistence: {order_id}")
                return None

            trade["closed_at"] = datetime.utcnow().isoformat()
            trade["current_price"] = round(float(current_price or trade.get("current_price", 0.0)), 4)
            trade["status"] = "CLOSED"
            trade["is_active"] = False
            trade["closed_by"] = triggered_by
            if exit_reason:
                trade["exit_reason"] = exit_reason
            closed_trade = dict(trade)
            await self._persist_active_trades()
            return {"result": result, "closed_trade": closed_trade}

    async def _mark_exit_failed(self, order_id: str):
        async with self._state_lock:
            trade = self.active_trades.get(order_id)
            if not trade or not trade.get("is_active", True):
                return
            trade["status"] = "EXIT_FAILED"
            await self._persist_active_trades()

    async def square_off_trade(
        self,
        order_id: str,
        current_price: Optional[float] = None,
        exit_reason: Optional[str] = None,
    ):
        return await self._close_trade_internal(
            order_id=order_id,
            current_price=current_price,
            exit_reason=exit_reason,
            triggered_by="autonomous_exit",
        )

    async def get_open_positions(self, symbol: str) -> list:
        """Fetches current active positions from Binance USD-M Futures or local persisted trades."""
        local_positions = [trade for trade in self.list_active_trades() if trade.get("symbol") == symbol]
        if local_positions:
            return local_positions

        return await self.broker.get_open_positions(symbol=symbol)

    async def clear_all_stale_trades(self):
        """Force-truncates the active trades local state file."""
        async with self._state_lock:
            self.active_trades.clear()
            await self._persist_active_trades()
            logger.warning("All stale trades have been force-cleared from local state.")

    async def purge_stale_dry_run_trades(self):
        """Removes DRY_RUN trades that have been open for more than 4 hours."""
        now = datetime.utcnow()
        purged = False
        async with self._state_lock:
            stale_orders = []
            for order_id, trade in self.active_trades.items():
                if trade.get("broker_status") == "DRY_RUN" and trade.get("is_active", True):
                    opened_at_str = trade.get("opened_at")
                    if opened_at_str:
                        try:
                            opened_at = datetime.fromisoformat(opened_at_str)
                            if (now - opened_at).total_seconds() > 4 * 3600:
                                stale_orders.append(order_id)
                        except ValueError:
                            pass
            
            for order_id in stale_orders:
                logger.warning(f"Purging stale DRY_RUN trade... order_id={order_id}")
                del self.active_trades[order_id]
                purged = True

            if purged:
                await self._persist_active_trades()

    async def set_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float = 1.0,
    ):
        """
        Sets a trailing stop market order to protect profits.
        Side should be the opposite of the open position.
        """
        side = side.upper()
        return await self.broker.set_trailing_stop(
            symbol=symbol,
            side=side,
            quantity=quantity,
            callback_rate=callback_rate
        )


if __name__ == "__main__":
    async def main():
        executor = TradeExecutor(live_trading_enabled=False)
        await executor.connect()
        await executor.execute_market_order(
            "BTCUSDT",
            "BUY",
            0.01,
            order_id="smoke",
            entry_price=70000.0,
            trailing_stop_level=69300.0,
        )
        await executor.disconnect()

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
