"""
Apex Intelligence Engine V5 - Bybit V5 Broker
================================================
Handles live and paper execution via the Bybit V5 API.
Optimized for ultra-low latency derivatives execution (Futures/Perps).

Bybit V5 Features used:
- Order Placement (Linear Perpetual)
- Position Management
- Balance Queries
- Signature Generation via HMAC-SHA256

Note: Set use_testnet=True to trade with play money.
"""

import time
import json
import hmac
import hashlib
from typing import Dict, Any, Optional

import httpx

from src.core.logging import get_logger

logger = get_logger("apex.execution.bybit")


class BybitBroker:
    """
    Async Broker for executing trades on Bybit V5 API.
    Designed for zero-latency, stateless execution.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        use_testnet: bool = True,
        recv_window: int = 20000
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_testnet = use_testnet
        self.recv_window = str(recv_window)
        self._time_offset_ms: int = 0  # corrected via sync_time()
        
        # Bybit Testnet uses https://api-testnet.bybit.com (testnet.bybit.com)
        # Bybit Mainnet Demo Trading uses https://api-demo.bybit.com
        import os as _os
        _use_demo = _os.getenv("BYBIT_USE_DEMO", "false").lower() == "true"
        if use_testnet and _use_demo:
            self.base_url = "https://api-demo.bybit.com"
        elif use_testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"
        
        # We use HTTPX AsyncClient for fast, non-blocking requests
        self._client = httpx.AsyncClient(base_url=self.base_url)
        
        logger.info(
            "bybit_broker_initialized",
            mode="TESTNET" if use_testnet else "MAINNET"
        )
    

    async def sync_time(self) -> None:
        """Fetch Bybit server time and calculate clock offset."""
        try:
            resp = await self._client.get(f"{self.base_url}/v5/market/time")
            data = resp.json()
            server_ms = int(data["result"]["timeNano"]) // 1_000_000
            local_ms  = int(time.time() * 1000)
            self._time_offset_ms = server_ms - local_ms
            import structlog as _sl
            _sl.get_logger().info("bybit_time_synced", offset_ms=self._time_offset_ms)
        except Exception as e:
            # fallback: ignore, recv_window=20000 is usually enough
            pass

    async def validate_account(self) -> dict:
        """Fetch unified trading account balance to verify API key validity."""
        try:
            await self.sync_time()
            res = await self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
            ret_code = res.get("retCode", 0)
            ret_msg = res.get("retMsg", "")
            if ret_code != 0:
                return {"valid": False, "balance": 0.0, "error": f"Bybit error {ret_code}: {ret_msg}"}
            try:
                bal = float(res["result"]["list"][0]["coin"][0]["walletBalance"])
                return {"valid": True, "balance": bal, "error": None}
            except Exception:
                return {"valid": True, "balance": 0.0, "error": None}
        except Exception as e:
            return {"valid": False, "balance": 0.0, "error": str(e)}

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()
        
    def _generate_signature(self, timestamp: str, payload: str) -> str:
        """
        Generate HMAC-SHA256 signature for Bybit V5.
        param_str = timestamp + api_key + recv_window + payload
        """
        param_str = timestamp + self.api_key + self.recv_window + payload
        return hmac.new(
            bytes(self.api_secret, "utf-8"),
            param_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
    def _get_headers(self, payload: str) -> Dict[str, str]:
        """Construct required headers for Bybit V5 private endpoints."""
        timestamp = str(int(time.time() * 1000) + self._time_offset_ms)
        signature = self._generate_signature(timestamp, payload)
        
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": self.recv_window,
            "Content-Type": "application/json"
        }
    
    async def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute HTTP request with authentication."""
        payload_str = json.dumps(data) if data else ""
        headers = self._get_headers(payload_str)
        
        start_time = time.perf_counter()
        
        try:
            if method.upper() == "GET":
                # For GET, payload_str should be URL-encoded query params.
                # Since V5 requires it in the signature string, we handle it slightly differently.
                # To keep it simple, we use query string for payload
                query_str = "&".join([f"{k}={v}" for k, v in (data or {}).items()])
                headers = self._get_headers(query_str)
                url = f"{endpoint}?{query_str}" if query_str else endpoint
                response = await self._client.get(url, headers=headers)
            else:
                response = await self._client.post(endpoint, headers=headers, content=payload_str)
            
            response.raise_for_status()
            result = response.json()
            
            latency = (time.perf_counter() - start_time) * 1000
            
            if result.get("retCode") != 0:
                logger.error(
                    "bybit_api_error",
                    endpoint=endpoint,
                    code=result.get("retCode"),
                    msg=result.get("retMsg"),
                    latency_ms=round(latency, 2)
                )
                return result
                
            logger.debug("bybit_api_success", endpoint=endpoint, latency_ms=round(latency, 2))
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error("bybit_http_error", status_code=e.response.status_code, response=e.response.text)
            raise e
        except Exception as e:
            logger.error("bybit_request_failed", error=str(e))
            raise e

    # =========================================================================
    # CORE TRADING METHODS
    # =========================================================================

    async def get_wallet_balance(self, coin: str = "USDT") -> float:
        """Fetch unified trading account balance."""
        res = await self._request("GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": coin})
        try:
            balance = res["result"]["list"][0]["coin"][0]["walletBalance"]
            return float(balance)
        except (KeyError, IndexError, TypeError):
            return 0.0

    async def place_market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> Optional[str]:
        """
        Place a Market Order for a Linear Perpetual contract.
        
        Args:
            symbol: e.g., "BTCUSDT"
            side: "Buy" or "Sell"
            qty: Quantity in base asset (e.g., 0.01 BTC)
            reduce_only: True if closing a position.
            
        Returns:
            Order ID if successful, None otherwise.
        """
        payload = {
            "category": "linear",
            "symbol": symbol.upper(),
            "side": side.capitalize(), # "Buy" or "Sell"
            "orderType": "Market",
            "qty": str(qty),
            "reduceOnly": reduce_only,
            "timeInForce": "IOC", # Immediate Or Cancel
        }
        
        res = await self._request("POST", "/v5/order/create", payload)
        
        if res.get("retCode") == 0:
            order_id = res["result"]["orderId"]
            logger.info("order_placed", symbol=symbol, side=side, qty=qty, order_id=order_id)
            return order_id
        
        return None
        
    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get current open position for a symbol.
        Returns empty dict if no position exists.
        """
        res = await self._request("GET", "/v5/position/list", {"category": "linear", "symbol": symbol.upper()})
        
        if res.get("retCode") == 0 and res["result"]["list"]:
            pos = res["result"]["list"][0]
            if float(pos["size"]) > 0:
                return {
                    "symbol": pos["symbol"],
                    "side": pos["side"],
                    "size": float(pos["size"]),
                    "entry_price": float(pos["avgPrice"]),
                    "unrealized_pnl": float(pos["unrealisedPnl"]),
                    "leverage": float(pos["leverage"])
                }
        return {}

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set the account leverage for a specific symbol on Linear Perpetuals.
        Bybit requires buying and selling leverage to be the same in one-way mode.
        """
        payload = {
            "category": "linear",
            "symbol": symbol.upper(),
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage)
        }
        
        res = await self._request("POST", "/v5/position/set-leverage", payload)
        
        # retCode 0 = Success. retCode 110043 = Leverage not modified (already set)
        if res.get("retCode") == 0 or res.get("retCode") == 110043:
            logger.info("leverage_set_success", symbol=symbol, leverage=leverage)
            return True
            
        logger.error("leverage_set_failed", symbol=symbol, response=res)
        return False

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
        """
        Place a Market Entry Order with Take-Profit and Stop-Loss orders attached.
        Automatically adapts to Bybit Testnet price drift if testnet sandbox diverges from live index.
        """
        target_sl = float(stop_loss)
        target_tp = float(take_profit)
        side_clean = side.capitalize()

        # Check Bybit's current execution price to ensure SL/TP brackets are valid
        try:
            ticker_res = await self._request("GET", "/v5/market/tickers", {"category": "linear", "symbol": symbol.upper()})
            if ticker_res.get("retCode") == 0 and ticker_res.get("result", {}).get("list"):
                bybit_ltp = float(ticker_res["result"]["list"][0].get("lastPrice", 0))
                if bybit_ltp > 0:
                    # Calculate distances
                    sl_dist = float(sl_distance) if sl_distance else abs(bybit_ltp - target_sl)
                    tp_dist = float(tp_distance) if tp_distance else abs(target_tp - bybit_ltp)
                    
                    if side_clean == "Buy":
                        # For LONG: SL must be below entry, TP must be above entry
                        target_sl = round(bybit_ltp - sl_dist, 1)
                        target_tp = round(bybit_ltp + tp_dist, 1)
                    else:
                        # For SHORT: SL must be above entry, TP must be below entry
                        target_sl = round(bybit_ltp + sl_dist, 1)
                        target_tp = round(bybit_ltp - tp_dist, 1)
        except Exception as _pe:
            logger.debug("bybit_price_adapt_error", error=str(_pe))

        payload = {
            "category": "linear",
            "symbol": symbol.upper(),
            "side": side_clean,
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "IOC",
            "takeProfit": str(target_tp),
            "stopLoss": str(target_sl),
            "tpslMode": "Full", # TP/SL covers the entire position
            "tpOrderType": "Market",
            "slOrderType": "Market"
        }
        
        res = await self._request("POST", "/v5/order/create", payload)
        
        if res.get("retCode") == 0:
            order_id = res["result"]["orderId"]
            logger.info(
                "order_placed_with_stops", 
                symbol=symbol, side=side, qty=qty, 
                tp=take_profit, sl=stop_loss, order_id=order_id
            )
            return order_id
            
        return None

    # =========================================================================
    # OPTIONS TRADING METHODS
    # =========================================================================
    
    async def get_option_chain(self, base_coin: str = "BTC") -> list[dict[str, Any]]:
        """
        Fetch all available options for the base coin (e.g., BTC, ETH).
        """
        res = await self._request("GET", "/v5/market/instruments-info", {
            "category": "option",
            "baseCoin": base_coin.upper()
        })
        
        if res.get("retCode") == 0:
            return res["result"]["list"]
        return []

    async def get_option_ticker(self, symbol: str) -> dict[str, float]:
        """
        Fetch the current mark price and Greeks for an option.
        """
        res = await self._request("GET", "/v5/market/tickers", {
            "category": "option",
            "symbol": symbol.upper()
        })
        
        if res.get("retCode") == 0 and res["result"]["list"]:
            data = res["result"]["list"][0]
            return {
                "markPrice": float(data.get("markPrice", 0)),
                "ask1Price": float(data.get("ask1Price", 0)),
                "bid1Price": float(data.get("bid1Price", 0)),
                "delta": float(data.get("delta", 0)),
                "gamma": float(data.get("gamma", 0)),
                "theta": float(data.get("theta", 0)),
                "vega": float(data.get("vega", 0)),
                "iv": float(data.get("markIv", 0))
            }
        return {}

    async def place_option_order(
        self, 
        symbol: str, 
        side: str, 
        qty: float, 
        price: Optional[float] = None
    ) -> Optional[str]:
        """
        Buy or Sell an option contract. Options are generally traded via limit orders
        due to wide spreads, but market orders are supported.
        """
        payload = {
            "category": "option",
            "symbol": symbol.upper(),
            "side": side.capitalize(),
            "orderType": "Limit" if price else "Market",
            "qty": str(qty)
        }
        
        if price:
            payload["price"] = str(price)
            
        res = await self._request("POST", "/v5/order/create", payload)
        
        if res.get("retCode") == 0:
            order_id = res["result"]["orderId"]
            logger.info("option_order_placed", symbol=symbol, side=side, qty=qty, order_id=order_id)
            return order_id
            
        return None
