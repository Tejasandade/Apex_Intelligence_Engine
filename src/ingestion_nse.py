import asyncio
import json
import os
import sys
import time
import pyotp
from dotenv import load_dotenv
from loguru import logger

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from src.data.db import DatabaseManager

load_dotenv()

ANGEL_API_KEY = os.getenv("ANGEL_API_KEY")
ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID")
ANGEL_PIN = os.getenv("ANGEL_PIN")
ANGEL_TOTP_KEY = os.getenv("ANGEL_TOTP_KEY")

db_manager = DatabaseManager()

CANDLE_INTERVAL_MS = 60_000

class CandleWarmupTracker:
    def __init__(self):
        from collections import deque
        self.current_candles: dict[str, dict] = {}
        self.completed_candles: dict[str, deque] = {}
        self.last_logged_counts: dict[str, int] = {}

    def apply_trade(self, symbol: str, price: float, volume: float, ts_ms: int) -> tuple[list[dict], dict | None]:
        normalized_symbol = symbol.lower()
        if normalized_symbol not in self.completed_candles:
            from collections import deque
            self.completed_candles[normalized_symbol] = deque(maxlen=20)

        bucket_start = (ts_ms // CANDLE_INTERVAL_MS) * CANDLE_INTERVAL_MS

        current_candle = self.current_candles.get(normalized_symbol)
        finalized_candle = None

        if current_candle is None or int(current_candle["open_time"]) != bucket_start:
            if current_candle is not None:
                finalized_candle = dict(current_candle)
                self.completed_candles[normalized_symbol].append(finalized_candle)

            current_candle = {
                "symbol": normalized_symbol,
                "open_time": bucket_start,
                "close_time": bucket_start + CANDLE_INTERVAL_MS - 1,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
            self.current_candles[normalized_symbol] = current_candle
        else:
            current_candle["high"] = max(float(current_candle["high"]), price)
            current_candle["low"] = min(float(current_candle["low"]), price)
            current_candle["close"] = price
            current_candle["volume"] = round(float(current_candle["volume"]) + volume, 8)

        candles = list(self.completed_candles[normalized_symbol])
        candles.append(dict(self.current_candles[normalized_symbol]))
        return candles, finalized_candle

candle_tracker = CandleWarmupTracker()

class NSEIngestionEngine:
    """
    Ingestion engine for the Indian Market (NSE) using Angel One SmartAPI.
    """
    def __init__(self):
        self.api_key = ANGEL_API_KEY
        self.client_id = ANGEL_CLIENT_ID
        self.pin = ANGEL_PIN
        self.totp_key = ANGEL_TOTP_KEY
        self.feed_token = None
        self.sws = None
        self.smart_connect = None
        self.loop = None

    async def connect_db(self):
        await db_manager.connect()

    def _authenticate(self):
        if not all([self.api_key, self.client_id, self.pin, self.totp_key]):
            logger.error("Angel One credentials missing. Please check .env file.")
            raise ValueError("Missing Angel One credentials")

        self.smart_connect = SmartConnect(api_key=self.api_key)
        
        totp = pyotp.TOTP(self.totp_key).now()
        logger.info(f"Authenticating with Angel One for client ID {self.client_id}...")
        data = self.smart_connect.generateSession(self.client_id, self.pin, totp)
        
        if data['status']:
            self.feed_token = self.smart_connect.getfeedToken()
            logger.info("Successfully authenticated and obtained feedToken.")
        else:
            logger.error(f"Authentication failed: {data}")
            raise Exception("Angel One Authentication Failed")

    def _on_message(self, ws, message):
        """
        Callback when a message is received from the websocket.
        Expected formatting: {"p": float(ltp), "v": float(volume), "t": int(timestamp)}.
        We will publish this to Redis `market:ticks:BANKNIFTY`.
        """
        logger.debug(f"Raw message received: {message}")
        try:
            ticks = message if isinstance(message, list) else [message]
            for tick in ticks:
                if not isinstance(tick, dict):
                    continue

                ltp = float(tick.get("last_traded_price", 0)) / 100.0 if tick.get("last_traded_price") else 0.0
                volume = float(tick.get("volume_trade_for_the_day", 0))
                
                ts_ms = int(time.time() * 1000)
                
                if ltp > 0:
                    payload = {"p": ltp, "v": volume, "t": ts_ms}
                    candles, _ = candle_tracker.apply_trade("BANKNIFTY", ltp, volume, ts_ms)
                    
                    if self.loop and self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            db_manager.redis.publish("market:ticks:BANKNIFTY", json.dumps(payload)),
                            self.loop
                        )
                        asyncio.run_coroutine_threadsafe(
                            db_manager.cache_market_candles("banknifty", candles),
                            self.loop
                        )
                    
                    logger.debug(f"Published NSE Tick & Updated Candles: {payload}")
                    
        except Exception as e:
            logger.error(f"Error parsing NSE tick: {e} | Raw message: {message}")

    def _on_open(self, ws):
        logger.info("Angel One SmartWebSocketV2 Opened. Subscribing to BANKNIFTY & HDFCBANK Spot...")
        # 26009 = BankNifty, 1333 = HDFCBANK, 3045 = SBIN
        token_list = [{"exchangeType": 1, "tokens": ["26009", "1333", "3045"]}]
        self.sws.subscribe("corrid", 1, token_list)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"Angel One SmartWebSocketV2 Closed. Code: {close_status_code}, Msg: {close_msg}")

    def _on_error(self, ws, error):
        logger.error(f"Angel One SmartWebSocketV2 Error: {error}")

    async def start(self):
        await self.connect_db()
        self.loop = asyncio.get_running_loop()
        
        self._authenticate()
        
        self.sws = SmartWebSocketV2(
            auth_token=self.smart_connect.access_token,
            api_key=self.api_key,
            client_code=self.client_id,
            feed_token=self.feed_token
        )
        
        self.sws.on_open = self._on_open
        self.sws.on_message = self._on_message
        self.sws.on_close = self._on_close
        self.sws.on_error = self._on_error
        
        logger.info("Starting NSE Ingestion WebSocket...")
        await self.loop.run_in_executor(None, self.sws.connect)


if __name__ == "__main__":
    logger.info("Starting NSE Ingestion Engine...")
    engine = NSEIngestionEngine()
    asyncio.run(engine.start())
