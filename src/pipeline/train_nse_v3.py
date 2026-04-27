"""
Apex Intelligence Engine — NSE V3 Training Pipeline
====================================================
Downloads 60 days of 1-minute BANKNIFTY historical data from
Angel One SmartAPI, engineers session-aware features, labels targets
with an asymmetric R/R rule, trains ApexXGBoostModel, and saves
banknifty_model.json.

Run:
    python -m src.pipeline.train_nse_v3
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss, precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Auto-load .env so credentials are available without manual export
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except ImportError:
    pass  # python-dotenv optional; rely on pre-exported env vars

from src.models.quant_model import MODEL_FEATURE_COLUMNS, ApexXGBoostModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL_TOKEN    = "99926009"         # BankNifty index — confirmed working token
EXCHANGE        = "NSE"
INTERVAL        = "ONE_MINUTE"      # SmartAPI interval string
TOTAL_DAYS      = 60                # how many days of history to fetch
CHUNK_DAYS      = 15                # days per API request (Angel One limit ~30d)

IST             = ZoneInfo("Asia/Kolkata")
SESSION_OPEN    = (9, 15)           # 09:15 IST
SESSION_CLOSE   = (15, 30)          # 15:30 IST
DROP_OPEN_BARS  = 15                # skip first N 1m bars per session (gap noise)

TARGET_UP_PCT   = 0.0015
TARGET_DN_PCT   = 0.0010
LOOKAHEAD       = 15

RSI_LEN, ATR_LEN, BOS_WIN, SWEEP_WIN = 14, 14, 20, 20


# ---------------------------------------------------------------------------
# 1. NSE Data Harvesting via Angel One SmartAPI
# ---------------------------------------------------------------------------
def _smartapi_client():
    """Authenticate with Angel One SmartConnect. Returns client object."""
    try:
        from SmartApi import SmartConnect
    except ImportError:
        logger.error("smartapi-python not installed. Run: pip install smartapi-python")
        sys.exit(1)

    api_key   = os.getenv("ANGEL_API_KEY")
    client_id = os.getenv("ANGEL_CLIENT_ID")
    pin       = os.getenv("ANGEL_PIN")
    totp_key  = os.getenv("ANGEL_TOTP_KEY", "")

    if not all([api_key, client_id, pin]):
        logger.error(
            "Missing Angel One credentials. "
            "Set ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PIN in .env"
        )
        sys.exit(1)

    smart = SmartConnect(api_key=api_key)

    # TOTP (if configured) — pyotp is optional
    totp_val = ""
    if totp_key:
        try:
            import pyotp
            totp_val = pyotp.TOTP(totp_key).now()
        except ImportError:
            logger.warning("pyotp not installed — skipping TOTP (may fail 2FA).")

    data = smart.generateSession(client_id, pin, totp_val)
    if not data.get("status"):
        logger.error(f"SmartAPI auth failed: {data}")
        sys.exit(1)

    logger.success(f"Angel One SmartAPI authenticated as {client_id}")
    return smart


def fetch_nse_candles(total_days: int = TOTAL_DAYS, chunk_days: int = CHUNK_DAYS) -> pd.DataFrame:
    """
    Fetch 1-min OHLCV for BANKNIFTY by paginating backwards in chunk_days
    windows. Date ranges snap to calendar boundaries (00:00 / 23:59 IST).
    Returns a UTC-indexed DataFrame sorted chronologically.
    """
    smart = _smartapi_client()
    all_rows: list = []

    # Work in whole calendar days (IST date only)
    today     = datetime.now(tz=IST).date()
    end_date  = today
    start_date = today - timedelta(days=total_days)

    cursor_date = end_date
    logger.info(
        f"Fetching {total_days}d of {INTERVAL} data for BANKNIFTY "
        f"[{start_date} -> {end_date}]"
    )

    while cursor_date > start_date:
        chunk_start_date = max(cursor_date - timedelta(days=chunk_days - 1), start_date)

        # Angel One expects "YYYY-MM-DD HH:MM" — use full-day window
        from_str = f"{chunk_start_date} 09:00"
        to_str   = f"{cursor_date} 23:59"

        params = {
            "exchange":    EXCHANGE,
            "symboltoken": SYMBOL_TOKEN,
            "interval":    INTERVAL,
            "fromdate":    from_str,
            "todate":      to_str,
        }

        try:
            resp = smart.getCandleData(params)
            candles = resp.get("data") or []
            if candles:
                all_rows = candles + all_rows
                logger.debug(
                    f"  [{from_str} -> {to_str}]: {len(candles)} bars | total={len(all_rows):,}"
                )
            else:
                logger.warning(
                    f"  No data [{from_str} -> {to_str}] | "
                    f"status={resp.get('status')} msg={resp.get('message','')}"
                )
        except Exception as exc:
            logger.error(f"  API error [{from_str} -> {to_str}]: {exc}")

        cursor_date = chunk_start_date - timedelta(days=1)
        time.sleep(0.4)

    if not all_rows:
        logger.error("No candle data received from Angel One API.")
        sys.exit(1)

    # Angel One returns ISO8601 strings with IST offset: '2026-04-24T09:15:00+05:30'
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)   # converts +05:30 -> UTC
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.success(
        f"Downloaded {len(df):,} bars  "
        f"[{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}]"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Vectorised Feature Engineering (session-aware for NSE)
# ---------------------------------------------------------------------------
def _vec_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = (-delta).where(delta < 0, 0.0)
    avg_g = gain.rolling(length, min_periods=length).mean()
    avg_l = loss.rolling(length, min_periods=length).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _vec_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _vec_macd(close: pd.Series):
    fast   = close.ewm(span=12, adjust=False).mean()
    slow   = close.ewm(span=26, adjust=False).mean()
    line   = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    hist   = line - signal
    return line, signal, hist


def _vec_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()


def _vec_session_vwap_nse(df: pd.DataFrame) -> pd.Series:
    """
    Session-bounded VWAP for NSE.
    - If volume data is available: cumsum(TV/V) grouped by IST session date.
    - If volume is all zero (index data): fall back to cumulative typical price
      mean per session (anchored TWAP — same shape, no NaN).
    """
    df_v = df.copy()
    df_v["ist"]        = df_v["timestamp"].dt.tz_convert(IST)
    df_v["local_date"] = df_v["ist"].dt.date

    tp = (df_v["high"] + df_v["low"] + df_v["close"]) / 3

    if df_v["volume"].sum() > 0:
        df_v["tv"]     = tp * df_v["volume"]
        df_v["cum_tv"] = df_v.groupby("local_date")["tv"].cumsum()
        df_v["cum_v"]  = df_v.groupby("local_date")["volume"].cumsum()
        vwap = df_v["cum_tv"] / df_v["cum_v"].replace(0, np.nan)
    else:
        # Index data: no volume — use cumulative mean of typical price per session
        df_v["tp"]       = tp
        df_v["cum_tp"]   = df_v.groupby("local_date")["tp"].cumsum()
        df_v["bar_num"]  = df_v.groupby("local_date").cumcount() + 1
        vwap = df_v["cum_tp"] / df_v["bar_num"]

    return vwap.fillna(tp)   # last-resort: fill any residual NaN with typical price


def _vec_cvd(df: pd.DataFrame) -> pd.Series:
    direction = np.where(df["close"] >= df["open"], 1, -1)
    return (df["volume"] * direction).cumsum()


def _vec_fvg(df: pd.DataFrame):
    c1_high = df["high"].shift(2)
    c1_low  = df["low"].shift(2)
    c3_low  = df["low"]
    c3_high = df["high"]
    close   = df["close"].replace(0, np.nan)

    bull_gap = c3_low - c1_high
    bear_gap = c1_low - c3_high

    signal = pd.Series(0.0, index=df.index)
    pct    = pd.Series(0.0, index=df.index)

    bull_mask = bull_gap > 0
    bear_mask = (bear_gap > 0) & ~bull_mask

    signal[bull_mask] =  1.0
    signal[bear_mask] = -1.0
    pct[bull_mask]    = (bull_gap[bull_mask] / close[bull_mask]).clip(0)
    pct[bear_mask]    = (bear_gap[bear_mask] / close[bear_mask]).clip(0)

    return signal, pct


def _vec_bos(df: pd.DataFrame, window: int = 20):
    pivot_high = df["high"].shift(1).rolling(window).max()
    pivot_low  = df["low"].shift(1).rolling(window).min()
    close      = df["close"]

    signal   = pd.Series(0.0, index=df.index)
    strength = pd.Series(0.0, index=df.index)

    bull_mask = close > pivot_high
    bear_mask = close < pivot_low

    signal[bull_mask]   =  1.0
    signal[bear_mask]   = -1.0
    strength[bull_mask] = ((close - pivot_high) / pivot_high.replace(0, np.nan)).clip(0)[bull_mask]
    strength[bear_mask] = ((pivot_low  - close) / pivot_low.replace(0,  np.nan)).clip(0)[bear_mask]

    return signal, strength


def _vec_liquidity_sweep(df: pd.DataFrame, window: int = 20):
    pivot_high   = df["high"].shift(1).rolling(window).max()
    pivot_low    = df["low"].shift(1).rolling(window).min()
    high         = df["high"]
    low          = df["low"]
    close        = df["close"]
    candle_range = (high - low).replace(0, np.nan)

    signal  = pd.Series(0.0, index=df.index)
    reclaim = pd.Series(0.0, index=df.index)

    bull_mask = (low < pivot_low) & (close > pivot_low)
    bear_mask = (high > pivot_high) & (close < pivot_high)

    signal[bull_mask]  =  1.0
    signal[bear_mask]  = -1.0
    reclaim[bull_mask] = ((close - low) / candle_range).clip(0, 1)[bull_mask]
    reclaim[bear_mask] = ((high - close) / candle_range).clip(0, 1)[bear_mask]

    return signal, reclaim


def _drop_session_open_bars(df: pd.DataFrame, n: int = DROP_OPEN_BARS) -> pd.DataFrame:
    """Remove the first N bars of each NSE trading session to avoid gap noise."""
    ist_times = df["timestamp"].dt.tz_convert(IST)
    session_start = (
        (ist_times.dt.hour == SESSION_OPEN[0]) &
        (ist_times.dt.minute < SESSION_OPEN[1] + n)
    )
    return df[~session_start].reset_index(drop=True)


def _filter_session_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows within NSE market hours (09:15 – 15:30 IST, Mon–Fri)."""
    ist = df["timestamp"].dt.tz_convert(IST)
    open_min  = SESSION_OPEN[0]  * 60 + SESSION_OPEN[1]
    close_min = SESSION_CLOSE[0] * 60 + SESSION_CLOSE[1]
    minutes   = ist.dt.hour * 60 + ist.dt.minute
    weekday   = ist.dt.dayofweek   # Mon=0, Fri=4

    mask = (minutes >= open_min) & (minutes <= close_min) & (weekday < 5)
    return df[mask].reset_index(drop=True)


def build_nse_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full vectorised feature engineering for NSE data."""
    out = df.copy()

    # ── Core indicators ────────────────────────────────────────────────────
    out["RSI"]         = _vec_rsi(out["close"], RSI_LEN)
    out["EMA_14"]      = _vec_ema(out["close"], 14)
    out["EMA_50"]      = _vec_ema(out["close"], 50)
    macd, macd_sig, macd_hist = _vec_macd(out["close"])
    out["MACD"]        = macd
    out["MACD_signal"] = macd_sig
    out["MACD_hist"]   = macd_hist
    out["VWAP"]        = _vec_session_vwap_nse(out)
    out["ATR"]         = _vec_atr(out, ATR_LEN)
    out["CVD"]         = _vec_cvd(out)

    # ── SMC features ──────────────────────────────────────────────────────
    fvg_sig, fvg_pct = _vec_fvg(out)
    bos_sig, bos_str = _vec_bos(out, BOS_WIN)
    sw_sig, sw_rcl   = _vec_liquidity_sweep(out, SWEEP_WIN)

    out["fvg_signal"]                = fvg_sig
    out["fvg_gap_pct"]               = fvg_pct
    out["structure_break_signal"]    = bos_sig
    out["structure_break_strength"]  = bos_str
    out["liquidity_sweep_signal"]    = sw_sig
    out["liquidity_reclaim_strength"] = sw_rcl

    out["structural_confluence"] = np.where(
        (fvg_sig == 1.0) & (bos_sig == 1.0),   1.0,
        np.where(
        (fvg_sig == -1.0) & (bos_sig == -1.0), -1.0, 0.0
        )
    )

    # ── LOB / broker columns — zero-filled for historical training ─────────
    for col in ("spread", "best_bid", "best_bid_qty",
                "best_ask", "best_ask_qty",
                "recent_long_liq_vol", "recent_short_liq_vol",
                "liq_imbalance", "macro_sentiment_score"):
        out[col] = 0.0

    out["symbol"] = "BANKNIFTY"

    return out


# ---------------------------------------------------------------------------
# 3. Target Definition
# ---------------------------------------------------------------------------
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asymmetric R/R label:
      1 = next 15 bars hit +0.15% gain WITHOUT breaching -0.10% stop
      0 = otherwise
    """
    close = df["close"]
    future_high = df["high"].shift(-1).rolling(LOOKAHEAD).max().shift(-(LOOKAHEAD - 1))
    future_low  = df["low"].shift(-1).rolling(LOOKAHEAD).min().shift(-(LOOKAHEAD - 1))

    df["target"] = np.where(
        (future_high > close * (1 + TARGET_UP_PCT)) &
        (future_low  >= close * (1 - TARGET_DN_PCT)),
        1, 0
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# 4. Training & Export
# ---------------------------------------------------------------------------
def train_and_save(df: pd.DataFrame) -> None:
    logger.info(f"Entering train_and_save with {len(df):,} rows, {len(df.columns)} columns")

    # Filter: no NaN in any of the 30 feature columns, and target must be a valid int
    feature_mask = df[MODEL_FEATURE_COLUMNS].notna().all(axis=1)
    target_mask  = df["target"].notna()
    df = df[feature_mask & target_mask].copy()

    logger.info(f"After NaN filter: {len(df):,} rows")

    pos_rate = df["target"].mean()
    logger.info(
        f"Training set: {len(df):,} rows | "
        f"positive rate: {pos_rate:.2%} | "
        f"features: {len(MODEL_FEATURE_COLUMNS)}"
    )

    model = ApexXGBoostModel(model_name="banknifty_model")
    model.train(df, target_col="target")

    X      = df[MODEL_FEATURE_COLUMNS].fillna(0.0)
    y_true = df["target"].values
    y_prob_arr = model.model.predict_proba(X)
    y_pred     = (y_prob_arr[:, 1] >= 0.5).astype(int)
    y_prob     = y_prob_arr[:, 1]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    ll   = log_loss(y_true, y_prob)

    logger.success("=" * 55)
    logger.success(f"  Accuracy   : {acc:.4f}")
    logger.success(f"  Precision  : {prec:.4f}")
    logger.success(f"  Log-Loss   : {ll:.4f}")
    logger.success("=" * 55)
    logger.success("Model saved -> data/models/banknifty_model.json")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def main():
    # Ensure output directory exists
    (ROOT / "data" / "models").mkdir(parents=True, exist_ok=True)

    raw_df      = fetch_nse_candles()
    raw_df      = _filter_session_hours(raw_df)
    raw_df      = _drop_session_open_bars(raw_df, DROP_OPEN_BARS)

    feature_df  = build_nse_features(raw_df)
    labelled_df = build_target(feature_df)

    # Quick diagnostic before training
    logger.info(f"Rows after feature build: {len(feature_df):,}")
    logger.info(f"Rows after target labelling: {len(labelled_df):,}")
    from src.models.quant_model import MODEL_FEATURE_COLUMNS as MFC
    missing_cols = [c for c in MFC if c not in labelled_df.columns]
    logger.info(f"Missing MODEL_FEATURE_COLUMNS: {missing_cols}")
    logger.info(f"target NaN count: {labelled_df.get('target', pd.Series([])).isna().sum()}")
    logger.info(f"DataFrame columns: {list(labelled_df.columns)}")

    train_and_save(labelled_df)


if __name__ == "__main__":
    main()
