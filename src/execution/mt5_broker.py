"""
Apex Intelligence Engine — MetaTrader 5 (MT5) Broker
===================================================
Executes live and demo orders via MetaTrader 5 Python API.
Designed for XM Global and other MT5 brokers (BTCUSD, Gold, Forex, CFDs).
"""

import os
import asyncio
from typing import Optional, Dict, Any
from src.core.logging import get_logger

logger = get_logger("apex.execution.mt5")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


class MT5Broker:
    """
    Broker implementation for MetaTrader 5 (XM Global, FTMO, etc.).
    Supports automated order placement with SL/TP, position management, and balance queries.
    """

    DEFAULT_PATHS = [
        r"C:\Program Files\XM Global MT5\terminal64.exe",
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\XM Global MT5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
    ]

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        magic: int = 777007
    ):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed. Run 'pip install MetaTrader5'.")

        self.login = int(login) if login else int(os.getenv("MT5_LOGIN", "0"))
        self.password = password or os.getenv("MT5_PASSWORD", "")
        self.server = server or os.getenv("MT5_SERVER", "")
        self.magic = magic

        # Auto-detect terminal path if not provided
        self.path = path or os.getenv("MT5_PATH")
        if not self.path:
            for p in self.DEFAULT_PATHS:
                if os.path.exists(p):
                    self.path = p
                    break

        self._initialized = False

    def _ensure_connected(self) -> bool:
        """Connect or reconnect to MT5 terminal."""
        if self._initialized:
            # Check if still connected
            term_info = mt5.terminal_info()
            if term_info and term_info.connected:
                return True

        init_kwargs: Dict[str, Any] = {}
        if self.path and os.path.exists(self.path):
            init_kwargs["path"] = self.path
        if self.login and self.login > 0:
            init_kwargs["login"] = self.login
        if self.password:
            init_kwargs["password"] = self.password
        if self.server:
            init_kwargs["server"] = self.server

        logger.info("mt5_connecting", server=self.server, login=self.login, path=self.path)
        ok = mt5.initialize(**init_kwargs)
        if not ok:
            err = mt5.last_error()
            logger.error("mt5_initialize_failed", error=err)
            self._initialized = False
            return False

        self._initialized = True
        acc = mt5.account_info()
        if acc:
            logger.info(
                "mt5_connected",
                login=acc.login,
                server=acc.server,
                balance=acc.balance,
                equity=acc.equity,
                currency=acc.currency
            )
        return True

    async def validate_account(self) -> dict:
        """Async validation of MT5 account connection and balance."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._validate_sync)

    def _validate_sync(self) -> dict:
        if not self._ensure_connected():
            err = mt5.last_error()
            return {
                "valid": False,
                "balance": 0.0,
                "error": f"MT5 connection failed: {err}"
            }
        acc = mt5.account_info()
        if not acc:
            return {
                "valid": False,
                "balance": 0.0,
                "error": "MT5 account_info is None"
            }
        return {
            "valid": True,
            "balance": float(acc.balance),
            "equity": float(acc.equity),
            "currency": acc.currency,
            "server": acc.server,
            "login": acc.login,
            "error": None
        }

    def resolve_symbol(self, target: str = "BTC") -> Optional[str]:
        """
        Find the exact symbol name used by the broker (e.g. BTCUSD#, BTCUSD, etc.).
        """
        if not self._ensure_connected():
            return None

        # Check explicit env override first
        env_sym = os.getenv("MT5_SYMBOL")
        if env_sym:
            info = mt5.symbol_info(env_sym)
            if info is not None:
                mt5.symbol_select(env_sym, True)
                return env_sym

        t_up = target.upper()

        # Priority list for Bitcoin
        if "BTC" in t_up:
            preferred = ["BTCUSD#", "BTCUSD", "BTCUSD.i", "BTCUSDm", "BTCUSDT"]
            for p in preferred:
                info = mt5.symbol_info(p)
                if info is not None:
                    mt5.symbol_select(p, True)
                    return p

        # Priority list for Gold
        if "GOLD" in t_up or "XAU" in t_up:
            preferred = ["GOLD#", "GOLD", "XAUUSD#", "XAUUSD", "GOLD.i#", "GOLD.i"]
            for p in preferred:
                info = mt5.symbol_info(p)
                if info is not None:
                    mt5.symbol_select(p, True)
                    return p

        # Direct exact match
        info = mt5.symbol_info(target)
        if info is not None:
            mt5.symbol_select(target, True)
            return target

        # Fallback search
        clean_target = t_up.replace("USDT", "USD")
        symbols = mt5.symbols_get()
        if not symbols:
            return None

        for s in symbols:
            name = s.name.upper()
            if clean_target in name and not name.startswith("G") and not name.startswith("F"):
                mt5.symbol_select(s.name, True)
                return s.name

        return None

    async def get_wallet_balance(self, coin: str = "USD") -> float:
        """Fetch current balance."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_balance_sync)

    def _get_balance_sync(self) -> float:
        if not self._ensure_connected():
            return 0.0
        acc = mt5.account_info()
        return float(acc.balance) if acc else 0.0

    async def place_order_with_stops(
        self,
        symbol: str,
        side: str,
        qty: float,
        take_profit: float,
        stop_loss: float,
        sl_distance: Optional[float] = None,
        tp_distance: Optional[float] = None
    ) -> Optional[str]:
        """Place market order with Stop Loss and Take Profit attached."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._place_order_sync,
            symbol, side, qty, take_profit, stop_loss
        )

    def _place_order_sync(
        self,
        symbol: str,
        side: str,
        qty: float,
        take_profit: float,
        stop_loss: float
    ) -> Optional[str]:
        if not self._ensure_connected():
            return None

        actual_symbol = self.resolve_symbol(symbol) or symbol
        sym_info = mt5.symbol_info(actual_symbol)
        if not sym_info:
            logger.error("mt5_symbol_not_found", symbol=actual_symbol)
            return None

        if not sym_info.visible:
            mt5.symbol_select(actual_symbol, True)

        tick = mt5.symbol_info_tick(actual_symbol)
        if not tick:
            logger.error("mt5_no_tick_data", symbol=actual_symbol)
            return None

        is_buy = side.upper() in ["BUY", "LONG"]
        order_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
        price = tick.ask if is_buy else tick.bid

        # Check volume constraints
        vol = max(sym_info.volume_min, min(float(qty), sym_info.volume_max))
        # Round to volume step
        step = sym_info.volume_step
        if step > 0:
            vol = round(vol / step) * step
            vol = round(vol, 4)

        # Detect supported filling mode
        filling = mt5.ORDER_FILLING_IOC
        if hasattr(sym_info, "filling_mode"):
            if sym_info.filling_mode & 2:
                filling = mt5.ORDER_FILLING_IOC
            elif sym_info.filling_mode & 1:
                filling = mt5.ORDER_FILLING_FOK
            else:
                filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": actual_symbol,
            "volume": vol,
            "type": order_type,
            "price": price,
            "sl": float(stop_loss),
            "tp": float(take_profit),
            "deviation": 20,
            "magic": self.magic,
            "comment": "Apex v7.1",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        logger.info("mt5_sending_order", request=request)
        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            logger.error("mt5_order_send_failed", error=err)
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error("mt5_order_rejected", retcode=result.retcode, comment=result.comment)
            return None

        order_id = str(result.order)
        logger.info(
            "mt5_order_placed_success",
            order_id=order_id,
            symbol=actual_symbol,
            side=side,
            volume=vol,
            price=price,
            sl=stop_loss,
            tp=take_profit
        )
        return order_id

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False
    ) -> Optional[str]:
        """Close or hedge positions via market order."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._close_position_sync,
            symbol, side, qty
        )

    def _close_position_sync(self, symbol: str, side: str, qty: float) -> Optional[str]:
        if not self._ensure_connected():
            return None

        actual_symbol = self.resolve_symbol(symbol) or symbol
        positions = mt5.positions_get(symbol=actual_symbol)
        if not positions:
            logger.warn("mt5_no_open_positions_to_close", symbol=actual_symbol)
            return None

        is_close_buy = side.upper() in ["BUY", "LONG"]
        order_type = mt5.ORDER_TYPE_BUY if is_close_buy else mt5.ORDER_TYPE_SELL

        # Close matching tickets
        closed_tickets = []
        for pos in positions:
            tick = mt5.symbol_info_tick(actual_symbol)
            price = tick.ask if is_close_buy else tick.bid
            vol = min(pos.volume, float(qty))

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": actual_symbol,
                "volume": vol,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": self.magic,
                "comment": "Apex close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            res = mt5.order_send(request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                closed_tickets.append(str(res.order))

        return ",".join(closed_tickets) if closed_tickets else None

    async def close(self):
        """Shutdown MT5 terminal connection."""
        if MT5_AVAILABLE and self._initialized:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, mt5.shutdown)
            self._initialized = False

