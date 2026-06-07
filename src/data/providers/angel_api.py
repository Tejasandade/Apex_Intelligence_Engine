"""
Apex Intelligence Engine V5 — Angel One API Provider
======================================================
Provides authentication and historical data fetching via Angel One SmartAPI.
Fully automated login using TOTP via pyotp.
"""

import os
import time
from typing import Any
import pandas as pd
import pyotp
from datetime import datetime, timedelta
from dotenv import load_dotenv

from SmartApi import SmartConnect
from src.core.logging import get_logger

logger = get_logger("apex.data.angel_api")


class AngelApiProvider:
    """
    Manages connection to Angel One SmartAPI.
    """

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ANGEL_API_KEY")
        self.client_id = os.getenv("ANGEL_CLIENT_ID")
        self.pin = os.getenv("ANGEL_PIN")
        self.totp_secret = os.getenv("ANGEL_TOTP_KEY")

        if not all([self.api_key, self.client_id, self.pin, self.totp_secret]):
            raise ValueError("Missing Angel One credentials in .env file.")

        self.smartApi = SmartConnect(api_key=self.api_key)
        self.auth_token = None
        self.refresh_token = None
        self.feed_token = None

        # Angel One token mapping cache
        # Mapping symbol -> token. e.g. "BANKNIFTY" -> "26009"
        self._token_map: dict[str, str] = {}
        # Mapping symbol -> exchange. e.g. "BANKNIFTY" -> "NSE"
        self._exchange_map: dict[str, str] = {}

    def connect(self) -> bool:
        """Authenticate with Angel One using TOTP."""
        try:
            logger.info("angel_api_connecting", client_id=self.client_id)
            
            # Generate current TOTP
            totp = pyotp.TOTP(self.totp_secret).now()
            
            # Login
            data = self.smartApi.generateSession(self.client_id, self.pin, totp)
            
            if data['status'] == False:
                logger.error("angel_api_login_failed", message=data['message'])
                return False

            self.auth_token = data['data']['jwtToken']
            self.refresh_token = data['data']['refreshToken']
            self.feed_token = self.smartApi.getfeedToken()
            
            logger.info("angel_api_connected", message="Login successful")
            
            # We hardcode the most common indices to avoid downloading the massive scrip master every time
            # For options, we will need to search the scrip master, but for spot data this is enough.
            self._token_map["NIFTY"] = "99926000"
            self._exchange_map["NIFTY"] = "NSE"
            self._token_map["BANKNIFTY"] = "99926009"
            self._exchange_map["BANKNIFTY"] = "NSE"
            
            return True

        except Exception as e:
            logger.error("angel_api_connect_error", error=str(e))
            return False

    def fetch_historical_candles(
        self,
        symbol: str,
        interval: str = "ONE_MINUTE",
        limit: int = 1000,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical candles for a given symbol.
        
        Args:
            symbol: Symbol name (e.g., "BANKNIFTY")
            interval: Timeframe (ONE_MINUTE, THREE_MINUTE, FIVE_MINUTE, etc)
            limit: Maximum number of candles to return
            from_date: Start time
            to_date: End time
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self.auth_token:
            if not self.connect():
                return pd.DataFrame()

        symbol_upper = symbol.upper()
        if symbol_upper not in self._token_map:
            logger.error("angel_api_unknown_symbol", symbol=symbol)
            return pd.DataFrame()

        token = self._token_map[symbol_upper]
        exchange = self._exchange_map[symbol_upper]

        if to_date is None:
            to_date = datetime.now()
        if from_date is None:
            # Note: Angel API restricts history limits. ONE_MINUTE is often limited to 30 days.
            from_date = to_date - timedelta(days=1)

        try:
            historicParam = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"), 
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            }
            
            response = self.smartApi.getCandleData(historicParam)
            
            if response['status'] == False:
                logger.error("angel_api_fetch_failed", message=response['message'])
                return pd.DataFrame()
                
            data = response.get('data', [])
            if not data:
                return pd.DataFrame()
                
            # Angel One format: [timestamp, open, high, low, close, volume]
            # timestamp is ISO string e.g. "2024-05-10T09:15:00+05:30"
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to int (milliseconds since epoch) for internal consistency
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**6
            
            # Ensure types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            # Limit results
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
                
            return df

        except Exception as e:
            logger.error("angel_api_fetch_error", symbol=symbol, error=str(e))
            return pd.DataFrame()
