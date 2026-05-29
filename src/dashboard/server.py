"""
Apex Intelligence Engine V5 — Dashboard WebSocket Server
==========================================================
Broadcasts live engine events to the web dashboard via WebSocket.

The LiveRunner pushes events here, and the dashboard connects
to receive them in real-time.

Events broadcast:
    status   — System status (balance, P&L, regime, etc.)
    signal   — New signal generated (with council decision)
    council  — Council voting details
    ensemble — Ensemble prediction breakdown
    trade_open  — New position opened
    trade_close — Position closed (with P&L)
    tick     — Price tick update (unrealized P&L)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.core.logging import get_logger

logger = get_logger("apex.dashboard.server")

# Try importing websockets, gracefully handle if not installed
try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("websockets not installed — dashboard server disabled")


class DashboardServer:
    """
    WebSocket server that broadcasts live engine events to the dashboard.

    Usage:
        server = DashboardServer(port=8765)
        await server.start()
        server.broadcast_signal({...})  # Called from LiveRunner
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set = set()
        self._server = None
        self._running = False
        self._message_count = 0

    async def start(self) -> None:
        """Start the WebSocket server."""
        if not HAS_WEBSOCKETS:
            logger.warning("Cannot start dashboard server — websockets not installed")
            return

        self._running = True
        self._server = await serve(
            self._handle_client,
            self.host,
            self.port,
        )

        logger.info(
            "dashboard_server_started",
            host=self.host,
            port=self.port,
            url=f"ws://{self.host}:{self.port}",
        )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("dashboard_server_stopped", messages_sent=self._message_count)

    async def _handle_client(self, websocket) -> None:
        """Handle a new dashboard client connection."""
        self._clients.add(websocket)
        client_id = f"client_{len(self._clients)}"

        logger.info("dashboard_client_connected", client=client_id, total=len(self._clients))

        try:
            async for message in websocket:
                # Handle incoming messages from dashboard (future: commands)
                try:
                    data = json.loads(message)
                    await self._handle_command(data, websocket)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("dashboard_client_disconnected", total=len(self._clients))

    async def _handle_command(self, data: dict, websocket) -> None:
        """Handle commands from the dashboard (future feature)."""
        cmd = data.get("command")
        if cmd == "ping":
            await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))

    def _broadcast(self, msg_type: str, data: dict[str, Any]) -> None:
        """Broadcast a message to all connected dashboard clients."""
        if not self._clients:
            return

        message = json.dumps({"type": msg_type, "data": data, "ts": time.time()})
        self._message_count += 1

        # Fire and forget — don't block the trading engine
        disconnected = set()
        for client in self._clients:
            try:
                asyncio.create_task(client.send(message))
            except Exception:
                disconnected.add(client)

        self._clients -= disconnected

    # ── Broadcast Methods (called by LiveRunner) ─────────────────────────────

    def broadcast_status(
        self,
        balance: float = 0,
        total_pnl: float = 0,
        total_trades: int = 0,
        winning_trades: int = 0,
        drawdown_pct: float = 0,
        regime: str = "",
        adx: float = 0,
        signals: int = 0,
        candles: int = 0,
        cooldown_remaining: int = 0,
        trailing_stops: dict = None,
    ) -> None:
        """Broadcast system status update."""
        if trailing_stops is None:
            trailing_stops = {}
            
        self._broadcast("status", {
            "balance": balance,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "drawdown_pct": drawdown_pct,
            "regime": regime,
            "adx": adx,
            "signals": signals,
            "candles": candles,
            "cooldown_remaining": cooldown_remaining,
            "trailing_stops": trailing_stops,
        })

    def broadcast_ensemble(self, trending: float, ranging: float, agreement: float) -> None:
        """Broadcast underlying ensemble model probabilities."""
        self._broadcast("ensemble", {
            "trending": trending,
            "ranging": ranging,
            "agreement": agreement,
        })

    def broadcast_signal(
        self,
        direction: str,
        symbol: str,
        price: float,
        conviction: float,
        council_score: float,
        blocked: bool = False,
        reason: str = "",
    ) -> None:
        """Broadcast a new signal (approved or blocked)."""
        self._broadcast("signal", {
            "direction": direction,
            "symbol": symbol,
            "price": price,
            "conviction": conviction,
            "council_score": council_score,
            "blocked": blocked,
            "reason": reason,
        })

    def broadcast_council(
        self,
        votes: dict[str, dict],
        consensus: float,
        approved: bool,
        summary: str = "",
    ) -> None:
        """Broadcast council voting details."""
        self._broadcast("council", {
            "votes": votes,
            "consensus": consensus,
            "approved": approved,
            "summary": summary,
        })

    def broadcast_ensemble(
        self,
        trending: float,
        ranging: float,
        agreement: float,
    ) -> None:
        """Broadcast ensemble prediction breakdown."""
        self._broadcast("ensemble", {
            "trending": trending,
            "ranging": ranging,
            "agreement": agreement,
        })

    def broadcast_trade_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Broadcast a new position opened."""
        self._broadcast("trade_open", {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "current_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        })

    def broadcast_trade_close(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        duration_seconds: float,
    ) -> None:
        """Broadcast a closed trade with P&L."""
        self._broadcast("trade_close", {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "duration_seconds": duration_seconds,
        })

    def broadcast_trail_update(self, update: dict) -> None:
        """Broadcast trailing stop update."""
        self._broadcast("trail_update", update)

    def broadcast_tick(self, unrealized_pnl: float, current_price: float = 0.0) -> None:
        """Broadcast unrealized P&L update and current price."""
        self._broadcast("tick", {
            "unrealized_pnl": unrealized_pnl,
            "current_price": current_price
        })

    @property
    def client_count(self) -> int:
        return len(self._clients)
