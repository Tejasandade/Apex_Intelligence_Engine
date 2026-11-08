"""
Apex Intelligence Engine — MetaTrader 5 (MT5) Broker
===================================================
Executes live and demo orders via MetaTrader 5 Python API.
Designed for XM Global and other MT5 brokers (BTCUSD, Gold, Forex, CFDs).
"""

import os
import time
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

        # Priority list for Bitcoin (supports all prop firm brokers: XM, Eightcap, FTMO, ThinkMarkets)
        if "BTC" in t_up:
            preferred = [
                "BTCUSD#", "BTCUSD", "BTCUSD.i", "BTCUSDm", "BTCUSDT", "BTCUSDT#",
                "BTCUSD.pro", "BTCUSD.m", "BTCUSD+", "BTCUSD_i", "BTC_USD"
            ]
            for p in preferred:
                info = mt5.symbol_info(p)
                if info is not None:
                    mt5.symbol_select(p, True)
                    return p

        # Priority list for Ethereum
        if "ETH" in t_up:
            preferred = [
                "ETHUSD#", "ETHUSD", "ETHUSD.i", "ETHUSDm", "ETHUSDT", "ETHUSDT#",
                "ETHUSD.pro", "ETHUSD.m", "ETHUSD+", "ETHUSD_i", "ETH_USD"
            ]
            for p in preferred:
                info = mt5.symbol_info(p)
                if info is not None:
                    mt5.symbol_select(p, True)
                    return p

        # Priority list for Gold (XAU / PAXG)
        if "GOLD" in t_up or "XAU" in t_up or "PAXG" in t_up:
            preferred = [
                "GOLD.i#", "GOLD#", "GOLD", "XAUUSD#", "XAUUSD", "GOLD.i",
                "XAUUSD.pro", "XAUUSD.m", "XAUUSD+", "XAUUSD_i", "XAUUSDm"
            ]
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

        # Fallback search across available broker symbols
        clean_target = t_up.replace("USDT", "USD")
        symbols = mt5.symbols_get()
        if not symbols:
            return None

        is_gold_search = ("GOLD" in t_up or "XAU" in t_up or "PAXG" in t_up)
        for s in symbols:
            name = s.name.upper()
            if is_gold_search:
                if "XAU" in name or "GOLD" in name:
                    mt5.symbol_select(s.name, True)
                    return s.name
            else:
                if clean_target in name:
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
            symbol, side, qty, take_profit, stop_loss, sl_distance, tp_distance
        )

    def _place_order_sync(
        self,
        symbol: str,
        side: str,
        qty: float,
        take_profit: float,
        stop_loss: float,
        sl_distance: Optional[float] = None,
        tp_distance: Optional[float] = None
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

        # ── Spread Optimizer & Fill Protection ──
        # Measure real broker spread (Ask - Bid)
        spread = tick.ask - tick.bid
        act_up = actual_symbol.upper()
        if "BTC" in act_up:
            max_spread_usd = 35.00
        elif "ETH" in act_up:
            max_spread_usd = 2.50
        elif "GOLD" in act_up or "XAU" in act_up or "PAXG" in act_up:
            max_spread_usd = 0.80
        else:
            max_spread_usd = float(sym_info.point * 150)
        
        wait_start = time.time()
        max_wait_secs = 6.0
        while spread > max_spread_usd and (time.time() - wait_start) < max_wait_secs:
            logger.warn(
                "mt5_spread_elevated_waiting",
                symbol=actual_symbol,
                spread=round(spread, 2),
                limit=max_spread_usd,
                wait_time=round(time.time() - wait_start, 1)
            )
            time.sleep(0.3)
            tick = mt5.symbol_info_tick(actual_symbol)
            if tick:
                spread = tick.ask - tick.bid
                price = tick.ask if is_buy else tick.bid

        if spread > max_spread_usd:
            logger.error(
                "mt5_order_aborted_spread_too_wide",
                symbol=actual_symbol,
                spread=round(spread, 2),
                limit=max_spread_usd
            )
            print(f"  ⚠️ [MT5] Entry Aborted: Spread elevated (${spread:.2f} > ${max_spread_usd:.2f} limit). Fill protection activated to prevent retail slippage.")
            return None

        # Adapt SL and TP to MT5 execution price using exact dollar distances
        digits = getattr(sym_info, "digits", 2)
        if sl_distance and float(sl_distance) > 0:
            target_sl = round(price - float(sl_distance), digits) if is_buy else round(price + float(sl_distance), digits)
        else:
            target_sl = round(float(stop_loss), digits)

        if tp_distance and float(tp_distance) > 0:
            target_tp = round(price + float(tp_distance), digits) if is_buy else round(price - float(tp_distance), digits)
        else:
            target_tp = round(float(take_profit), digits)

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
            "sl": float(target_sl),
            "tp": float(target_tp),
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
        fill_price = float(result.price) if getattr(result, "price", 0) > 0 else price

        # If fill price slipped from quoted price, recalibrate stops to exact fill price
        if sl_distance and float(sl_distance) > 0 and abs(fill_price - price) >= 1.0:
            exact_sl = round(fill_price - float(sl_distance), digits) if is_buy else round(fill_price + float(sl_distance), digits)
            exact_tp = round(fill_price + float(tp_distance), digits) if (tp_distance and is_buy) else (round(fill_price - float(tp_distance), digits) if tp_distance else target_tp)
            logger.info("mt5_fill_anchoring_stops", fill_price=fill_price, quote_price=price, exact_sl=exact_sl, exact_tp=exact_tp)
            self._modify_position_stops_sync(result.order, exact_sl, exact_tp)

        logger.info(
            "mt5_order_placed_success",
            order_id=order_id,
            symbol=actual_symbol,
            side=side,
            volume=vol,
            price=fill_price,
            sl=target_sl,
            tp=target_tp
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

    # =========================================================================
    # ADVANCED POSITION MANAGEMENT & SYNCHRONIZATION
    # =========================================================================

    async def get_open_position(self, ticket: Optional[int] = None, symbol: Optional[str] = None) -> Optional[dict]:
        """Fetch real-time details of an open position by ticket or symbol."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_open_position_sync, ticket, symbol)

    def _get_open_position_sync(self, ticket: Optional[int] = None, symbol: Optional[str] = None) -> Optional[dict]:
        if not self._ensure_connected():
            return None

        positions = None
        if ticket:
            positions = mt5.positions_get(ticket=int(ticket))
        elif symbol:
            act_sym = self.resolve_symbol(symbol) or symbol
            positions = mt5.positions_get(symbol=act_sym)
        else:
            positions = mt5.positions_get()

        if not positions:
            return None

        p = positions[0]
        return {
            "ticket": p.ticket,
            "symbol": p.symbol,
            "type": "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": p.volume,
            "price_open": p.price_open,
            "price_current": p.price_current,
            "sl": p.sl,
            "tp": p.tp,
            "profit": p.profit,
            "time": p.time,
        }

    async def modify_position_stops(self, ticket: int, new_sl: float, new_tp: Optional[float] = None) -> bool:
        """Modify Stop Loss and Take Profit of an active position in MT5."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._modify_position_stops_sync, ticket, new_sl, new_tp)

    def _modify_position_stops_sync(self, ticket: int, new_sl: float, new_tp: Optional[float] = None) -> bool:
        if not self._ensure_connected():
            return False

        pos = mt5.positions_get(ticket=int(ticket))
        if not pos:
            logger.warn("mt5_cannot_modify_stops_pos_not_found", ticket=ticket)
            return False

        p = pos[0]
        tp_val = float(new_tp) if new_tp is not None else float(p.tp)
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": p.ticket,
            "symbol": p.symbol,
            "sl": float(new_sl),
            "tp": tp_val,
        }

        res = mt5.order_send(request)
        if res is None:
            err = mt5.last_error()
            logger.error("mt5_modify_stops_failed", error=err, ticket=ticket)
            return False

        if res.retcode != mt5.TRADE_RETCODE_DONE:
            logger.warn("mt5_modify_stops_rejected", retcode=res.retcode, comment=res.comment, ticket=ticket)
            return False

        logger.info("mt5_stops_modified_success", ticket=ticket, sl=new_sl, tp=tp_val)
        return True

    async def close_position_by_ticket(self, ticket: int) -> bool:
        """Close an exact position by ticket at market price."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._close_position_by_ticket_sync, ticket, None)

    async def partial_close_position(self, ticket: int, volume: float) -> bool:
        """Partially close an open position by ticket, banking profit on that portion."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._close_position_by_ticket_sync, ticket, volume)

    def _close_position_by_ticket_sync(self, ticket: int, volume: Optional[float] = None) -> bool:
        if not self._ensure_connected():
            return False

        pos = mt5.positions_get(ticket=int(ticket))
        if not pos:
            logger.warn("mt5_close_ticket_not_found", ticket=ticket)
            return False

        p = pos[0]
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(p.symbol)
        if not tick:
            logger.error("mt5_close_no_tick_data", symbol=p.symbol)
            return False

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        sym_info = mt5.symbol_info(p.symbol)
        close_vol = p.volume
        if volume is not None and 0.0 < float(volume) < p.volume:
            step = getattr(sym_info, "volume_step", 0.01) if sym_info else 0.01
            min_v = getattr(sym_info, "volume_min", 0.01) if sym_info else 0.01
            close_vol = max(min_v, round(float(volume) / step) * step)
            close_vol = round(close_vol, 4)
            if (p.volume - close_vol) < min_v:
                close_vol = p.volume

        filling = mt5.ORDER_FILLING_IOC
        if sym_info and hasattr(sym_info, "filling_mode"):
            if sym_info.filling_mode & 2:
                filling = mt5.ORDER_FILLING_IOC
            elif sym_info.filling_mode & 1:
                filling = mt5.ORDER_FILLING_FOK
            else:
                filling = mt5.ORDER_FILLING_RETURN

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": p.ticket,
            "symbol": p.symbol,
            "volume": close_vol,
            "type": close_type,
            "price": price,
            "deviation": 20,
            "magic": p.magic,
            "comment": "Apex Partial Exit" if close_vol < p.volume else "Apex Exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        res = mt5.order_send(request)
        if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
            err = res.comment if res else str(mt5.last_error())
            logger.error("mt5_close_failed", ticket=ticket, error=err)
            return False

        logger.info("mt5_position_closed_success", ticket=ticket, volume=close_vol, exit_price=price)
        return True

    async def get_closed_deal(self, ticket: int) -> Optional[dict]:
        """Fetch exit deal details from history for a closed position."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_closed_deal_sync, ticket)

    def _get_closed_deal_sync(self, ticket: int) -> Optional[dict]:
        if not self._ensure_connected():
            return None

        deals = mt5.history_deals_get(position=int(ticket))
        if not deals:
            return None

        # Look for the exit deal (entry == 1, DEAL_ENTRY_OUT)
        for d in reversed(deals):
            if d.entry == 1:
                return {
                    "deal_ticket": d.ticket,
                    "order": d.order,
                    "price": d.price,
                    "profit": d.profit,
                    "commission": d.commission,
                    "swap": d.swap,
                    "comment": d.comment,
                    "time": d.time,
                }

        # Fallback to last deal
        last_d = deals[-1]
        return {
            "deal_ticket": last_d.ticket,
            "order": last_d.order,
            "price": last_d.price,
            "profit": last_d.profit,
            "comment": last_d.comment,
            "time": last_d.time,
        }

    async def close(self):
        """Shutdown MT5 terminal connection."""
        if MT5_AVAILABLE and self._initialized:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, mt5.shutdown)
            self._initialized = False

