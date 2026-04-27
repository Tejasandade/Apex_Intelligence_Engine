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
        self.current_minute = -1
        self.current_candle = None

    async def _push_candle(self, candle: dict):
        key = "market:candles:1m:BANKNIFTY"
        try:
            await db_manager.redis_pool.lpush(key, json.dumps(candle))
            await db_manager.redis_pool.ltrim(key, 0, 99)
            logger.info(f"Pushed 1m candle to {key}: {candle}")
        except Exception as e:
            logger.error(f"Failed to push candle: {e}")

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

                ltp = tick.get("last_traded_price", 0) / 100.0
                volume = tick.get("volume_traded_today", 0)
                
                ts_ms = int(time.time() * 1000)
                
                if ltp > 0:
                    # L1 Firewall: Z-Score Anomaly Rejection
                    if self.current_candle and self.current_candle.get("close", 0) > 0:
                        last_price = self.current_candle["close"]
                        if abs(ltp - last_price) / last_price > 0.015:
                            logger.warning(f"[L1 FIREWALL] Dropped anomalous tick | Last Price: {last_price} | Anomalous Price: {ltp}")
                            return

                    payload = {"p": ltp, "v": volume, "t": ts_ms}
                    
                    if self.loop and self.loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            db_manager.redis.publish("market:ticks:BANKNIFTY", json.dumps(payload)),
                            self.loop
                        )
                        
                        now = datetime.now()
                        minute = now.minute
                        
                        if self.current_minute == -1:
                            self.current_minute = minute
                            self.current_candle = {
                                "symbol": "banknifty",
                                "open_time": ts_ms,
                                "close_time": ts_ms + 59999,
                                "open": ltp,
                                "high": ltp,
                                "low": ltp,
                                "close": ltp,
                                "volume": volume
                            }
                            
                        if minute != self.current_minute:
                            # Minute changed, push candle
                            asyncio.run_coroutine_threadsafe(
                                self._push_candle(self.current_candle),
                                self.loop
                            )
                            # Reset state
                            self.current_minute = minute
                            self.current_candle = {
                                "symbol": "banknifty",
                                "open_time": ts_ms,
                                "close_time": ts_ms + 59999,
                                "open": ltp,
                                "high": ltp,
                                "low": ltp,
                                "close": ltp,
                                "volume": volume
                            }
                        else:
                            # Update current candle
                            self.current_candle["high"] = max(self.current_candle["high"], ltp)
                            self.current_candle["low"] = min(self.current_candle["low"], ltp)
                            self.current_candle["close"] = ltp
                            self.current_candle["volume"] += volume
                    
                    logger.debug(f"Published NSE Tick: {payload}")
                    
        except Exception as e:
            logger.error(f"Error parsing NSE tick: {e} | Raw message: {message}")

    def _on_open(self, ws):
        logger.info("Angel One SmartWebSocketV2 Opened. Subscribing to BANKNIFTY & HDFCBANK Spot...")
        # 99926009 = BankNifty, 1333 = HDFCBANK, 3045 = SBIN
        token_list = [{"exchangeType": 1, "tokens": ["99926009", "1333", "3045"]}]
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
