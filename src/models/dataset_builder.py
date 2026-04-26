import json

import pandas as pd
from loguru import logger

from src.data.db import (
    WARMUP_CANDLES_REQUIRED,
    DatabaseManager,
    build_book_key,
    build_candle_cache_key,
)


def compute_rsi(series: pd.Series, length: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0


def compute_ema(series: pd.Series, span: int) -> float:
    ema = series.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1]) if not ema.empty else float(series.iloc[-1])


def compute_macd(series: pd.Series):
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return (
        float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
        float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0.0,
    )


def compute_vwap(df: pd.DataFrame) -> float:
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    return float(vwap.iloc[-1]) if not vwap.empty and not pd.isna(vwap.iloc[-1]) else 0.0


def compute_atr(df: pd.DataFrame, length: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=length).mean()
    return float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0


def _normalize_gap(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return round(value / reference, 6)


def detect_fvg(df: pd.DataFrame) -> dict:
    if len(df) < 3:
        return {"fvg_signal": 0.0, "fvg_gap_pct": 0.0}

    candle_a = df.iloc[-3]
    candle_c = df.iloc[-1]
    reference = max(float(candle_c["close"]), 1.0)

    bullish_gap = float(candle_c["low"]) - float(candle_a["high"])
    bearish_gap = float(candle_a["low"]) - float(candle_c["high"])

    if bullish_gap > 0:
        return {
            "fvg_signal": 1.0,
            "fvg_gap_pct": _normalize_gap(bullish_gap, reference),
        }
    if bearish_gap > 0:
        return {
            "fvg_signal": -1.0,
            "fvg_gap_pct": _normalize_gap(bearish_gap, reference),
        }
    return {"fvg_signal": 0.0, "fvg_gap_pct": 0.0}


def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> dict:
    if len(df) < max(lookback + 1, 6):
        return {"liquidity_sweep_signal": 0.0, "liquidity_reclaim_strength": 0.0}

    current = df.iloc[-1]
    window = df.iloc[-(lookback + 1):-1]
    atr = compute_atr(df.tail(max(lookback, 20)))
    atr = atr if atr > 0 else max(float(current["close"]) * 0.001, 1.0)

    prior_high = float(window["high"].max())
    prior_low = float(window["low"].min())
    close_price = float(current["close"])
    high_price = float(current["high"])
    low_price = float(current["low"])

    if low_price < prior_low and close_price > prior_low:
        reclaim_strength = max(close_price - prior_low, 0.0) / atr
        return {
            "liquidity_sweep_signal": 1.0,
            "liquidity_reclaim_strength": round(reclaim_strength, 6),
        }
    if high_price > prior_high and close_price < prior_high:
        reclaim_strength = max(prior_high - close_price, 0.0) / atr
        return {
            "liquidity_sweep_signal": -1.0,
            "liquidity_reclaim_strength": round(reclaim_strength, 6),
        }
    return {"liquidity_sweep_signal": 0.0, "liquidity_reclaim_strength": 0.0}


def detect_structure_break(df: pd.DataFrame, lookback: int = 12) -> dict:
    if len(df) < max(lookback + 2, 8):
        return {"structure_break_signal": 0.0, "structure_break_strength": 0.0}

    current = df.iloc[-1]
    previous = df.iloc[-2]
    structure_window = df.iloc[-(lookback + 2):-2]

    prior_high = float(structure_window["high"].max())
    prior_low = float(structure_window["low"].min())
    atr = compute_atr(df.tail(max(lookback, 20)))
    atr = atr if atr > 0 else max(float(current["close"]) * 0.001, 1.0)
    trend_bias = float(previous["close"]) - float(structure_window["close"].mean())

    if float(current["close"]) > prior_high:
        break_strength = (float(current["close"]) - prior_high) / atr
        return {
            "structure_break_signal": 2.0 if trend_bias >= 0 else 1.0,
            "structure_break_strength": round(break_strength, 6),
        }
    if float(current["close"]) < prior_low:
        break_strength = (prior_low - float(current["close"])) / atr
        return {
            "structure_break_signal": -2.0 if trend_bias <= 0 else -1.0,
            "structure_break_strength": round(break_strength, 6),
        }
    return {"structure_break_signal": 0.0, "structure_break_strength": 0.0}


def build_structure_features(df: pd.DataFrame) -> dict:
    fvg = detect_fvg(df)
    sweep = detect_liquidity_sweep(df)
    structure = detect_structure_break(df)
    structural_confluence = (
        fvg["fvg_signal"] != 0.0
        and fvg["fvg_signal"] == sweep["liquidity_sweep_signal"]
    )

    return {
        **fvg,
        **sweep,
        **structure,
        "structural_confluence": 1.0 if structural_confluence else 0.0,
    }


def apply_signal_reference_to_latest_candle(
    klines_df: pd.DataFrame,
    signal_reference_price: float | None,
) -> pd.DataFrame:
    if klines_df.empty:
        return klines_df

    if signal_reference_price is None or signal_reference_price <= 0:
        return klines_df

    adjusted_df = klines_df.copy()
    for column in ("open", "high", "low", "close", "volume"):
        adjusted_df[column] = pd.to_numeric(adjusted_df[column], errors="coerce").astype(float)
    last_idx = adjusted_df.index[-1]
    adjusted_df.at[last_idx, "close"] = float(signal_reference_price)
    adjusted_df.at[last_idx, "high"] = max(float(adjusted_df.at[last_idx, "high"]), float(signal_reference_price))
    adjusted_df.at[last_idx, "low"] = min(float(adjusted_df.at[last_idx, "low"]), float(signal_reference_price))
    return adjusted_df


class DataFetcher:
    """
    Fetches raw data from TimescaleDB, Redis, and Binance REST API,
    and builds normalized feature datasets for the ML models.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def fetch_recent_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 60,
    ) -> pd.DataFrame:
        if not self.db_manager.redis_pool:
            logger.error("Redis pool not initialized.")
            return pd.DataFrame()

        try:
            payload = await self.db_manager.redis_pool.get(build_candle_cache_key(symbol, interval))
            if not payload:
                return pd.DataFrame()

            candles = json.loads(payload)
            if not isinstance(candles, list) or not candles:
                return pd.DataFrame()

            df = pd.DataFrame(candles)
            expected_columns = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
            missing_columns = [column for column in expected_columns if column not in df.columns]
            if missing_columns:
                logger.warning("Cached candle payload missing columns for {}: {}", symbol.upper(), missing_columns)
                return pd.DataFrame()

            for column in ["open", "high", "low", "close", "volume"]:
                df[column] = pd.to_numeric(df[column], errors="coerce")
            logger.debug("Fetched {} fresh {} candles from Redis for {}", len(df), interval, symbol)
            return df.tail(limit).reset_index(drop=True)
        except Exception as exc:
            logger.error(f"Failed to fetch cached klines: {exc}")
            return pd.DataFrame()

    async def fetch_warmup_status(self, symbol: str, interval: str = "1m") -> dict:
        try:
            return await self.db_manager.get_warmup_status(symbol, interval)
        except Exception as exc:
            logger.error(f"Failed to fetch warm-up status from Redis: {exc}")
            return {
                "symbol": symbol.upper(),
                "ready": False,
                "candle_count": 0,
                "required_candles": WARMUP_CANDLES_REQUIRED,
            }

    async def fetch_historical_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        if not self.db_manager.pg_pool:
            logger.error("Postgres pool not initialized.")
            return pd.DataFrame()

        query = """
            SELECT time, symbol, side, price, quantity
            FROM liquidations
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT $2
        """
        async with self.db_manager.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, limit)

        if not rows:
            return pd.DataFrame(columns=["time", "symbol", "side", "price", "quantity"])

        df = pd.DataFrame(rows, columns=["time", "symbol", "side", "price", "quantity"])
        return df.sort_values("time").reset_index(drop=True)

    async def fetch_current_lob(self, symbol: str) -> dict:
        if not self.db_manager.redis_pool:
            logger.error("Redis pool not initialized.")
            return {}

        candidates = [build_book_key(symbol), f"book:{symbol}", f"book:{symbol.lower()}"]
        payload = None
        for key in candidates:
            payload = await self.db_manager.redis_pool.get(key)
            if payload:
                break
        if not payload:
            return {}

        return json.loads(payload)

    async def fetch_signal_reference_price(self, symbol: str) -> float | None:
        lob = await self.fetch_current_lob(symbol)
        if not lob:
            return None

        bids = lob.get("bids", [])
        asks = lob.get("asks", [])
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            return round((best_bid + best_ask) / 2, 6)
        if bids:
            return round(float(bids[0][0]), 6)
        if asks:
            return round(float(asks[0][0]), 6)
        return None

    async def build_dataset(self, symbol: str) -> pd.DataFrame:
        warmup_status = await self.fetch_warmup_status(symbol)
        if not warmup_status.get("ready"):
            logger.info(
                "Warm-up gate active for {} | fresh_1m_candles={}/{}",
                symbol.upper(),
                warmup_status.get("candle_count", 0),
                warmup_status.get("required_candles", WARMUP_CANDLES_REQUIRED),
            )
            return pd.DataFrame()

        lob_data = await self.fetch_current_lob(symbol)
        best_bid = best_bid_qty = best_ask = best_ask_qty = 0.0
        if lob_data:
            bids = lob_data.get("bids", [])
            asks = lob_data.get("asks", [])
            if bids:
                best_bid = float(bids[0][0])
                best_bid_qty = float(bids[0][1])
            if asks:
                best_ask = float(asks[0][0])
                best_ask_qty = float(asks[0][1])

        klines_df = await self.fetch_recent_klines(symbol)
        liq_df = await self.fetch_historical_liquidations(symbol)
        macro_sentiment = await self.db_manager.fetch_latest_sentiment()
        # Derive signal reference from the LOB we already fetched (avoid double Redis call)
        if best_bid > 0 and best_ask > 0:
            signal_reference_price = round((best_bid + best_ask) / 2.0, 6)
        elif best_bid > 0:
            signal_reference_price = best_bid
        elif best_ask > 0:
            signal_reference_price = best_ask
        else:
            # Fall back to last candle close if LOB is unavailable
            signal_reference_price = None
            if not klines_df.empty and "close" in klines_df.columns:
                signal_reference_price = float(pd.to_numeric(klines_df["close"], errors="coerce").iloc[-1])
        current_close = float(signal_reference_price) if signal_reference_price is not None and signal_reference_price > 0 else None

        if not klines_df.empty and len(klines_df) >= WARMUP_CANDLES_REQUIRED:
            klines_df = apply_signal_reference_to_latest_candle(klines_df, current_close)
            close = klines_df["close"]
            rsi = compute_rsi(close, length=14)
            ema_14 = compute_ema(close, span=14)
            ema_50 = compute_ema(close, span=50)
            macd, macd_signal, macd_hist = compute_macd(close)
            vwap = compute_vwap(klines_df)
            atr = compute_atr(klines_df, length=14)

            latest = klines_df.iloc[-1]
            open_price = float(latest["open"])
            high_price = float(latest["high"])
            low_price = float(latest["low"])
            close_price = float(latest["close"])
            volume = float(latest["volume"])
            current_close = float(current_close or close_price)
            close_price = current_close

            direction = (klines_df["close"] - klines_df["open"]).apply(
                lambda value: 1 if value >= 0 else -1
            )
            cvd = float((klines_df["volume"] * direction).sum())
            structure_features = build_structure_features(klines_df)

            lob_src = "LOB" if (best_bid > 0 or best_ask > 0) else "CANDLE"
            logger.debug(
                "Fresh WebSocket candle set in use for {} | signal_reference={:.2f} [src={}] | best_bid={:.2f} | best_ask={:.2f} | RSI={:.1f} | VWAP={:.2f} | ATR={:.2f} | MACD={:.4f}",
                symbol.upper(),
                current_close,
                lob_src,
                best_bid,
                best_ask,
                rsi,
                vwap,
                atr,
                macd,
            )
        else:
            logger.warning("Insufficient fresh 1m candle data for {} after warm-up gate.", symbol.upper())
            rsi, ema_14, ema_50 = 50.0, 0.0, 0.0
            macd, macd_signal, macd_hist = 0.0, 0.0, 0.0
            vwap, atr = 0.0, 0.0
            close_price = float(current_close or 0.0)
            open_price = high_price = low_price = close_price
            volume, cvd = 0.0, 0.0
            current_close = close_price if close_price > 0 else None
            structure_features = {
                "fvg_signal": 0.0,
                "fvg_gap_pct": 0.0,
                "liquidity_sweep_signal": 0.0,
                "liquidity_reclaim_strength": 0.0,
                "structure_break_signal": 0.0,
                "structure_break_strength": 0.0,
                "structural_confluence": 0.0,
            }

        spread = (
            best_ask - best_bid
            if best_ask > 0 and best_bid > 0
            else (high_price - low_price) * 0.1
        )

        long_liq_vol = short_liq_vol = 0.0
        if not liq_df.empty:
            long_liq_vol = liq_df[liq_df["side"] == "SELL"]["quantity"].sum()
            short_liq_vol = liq_df[liq_df["side"] == "BUY"]["quantity"].sum()
        liq_imbalance = long_liq_vol - short_liq_vol

        feature_vector = {
            "symbol": symbol,
            "timestamp": pd.Timestamp.utcnow(),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "RSI": rsi,
            "EMA_14": ema_14,
            "EMA_50": ema_50,
            "MACD": macd,
            "MACD_signal": macd_signal,
            "MACD_hist": macd_hist,
            "VWAP": vwap,
            "ATR": atr,
            "spread": spread,
            "CVD": cvd,
            "best_bid": best_bid,
            "best_bid_qty": best_bid_qty,
            "best_ask": best_ask,
            "best_ask_qty": best_ask_qty,
            "recent_long_liq_vol": long_liq_vol,
            "recent_short_liq_vol": short_liq_vol,
            "liq_imbalance": liq_imbalance,
            "signal_price": float(current_close or close_price or 0.0),
            "market_price": float(current_close or close_price or 0.0),
            "price_gap": 0.0,
            **structure_features,
            "macro_sentiment_score": macro_sentiment,
        }

        return pd.DataFrame([feature_vector])
