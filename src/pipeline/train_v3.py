"""
Apex Intelligence Engine — V3 Production Training Pipeline
==========================================================
Downloads 20,000 1-minute BTCUSDT Futures candles from Binance,
engineers the full feature set using vectorised Pandas operations
(matching MODEL_FEATURE_COLUMNS exactly), labels targets with an
asymmetric R/R rule, trains ApexXGBoostModel, and saves weights.

Run:
    python -m src.pipeline.train_v3
"""

import asyncio
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import accuracy_score, log_loss, precision_score

# Ensure project root is on the path when run directly
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.quant_model import MODEL_FEATURE_COLUMNS, _BaseXGBoostModel, classify_regime
from sklearn.model_selection import KFold
import xgboost as xgb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL          = "BTCUSDT"
CANDLE_LIMIT    = 20_000          # total 1m candles to fetch
FETCH_BATCH     = 1_000           # Binance max per request (1500 allowed, 1000 is safe)
TIMEFRAME       = "1m"

# Binance Futures klines endpoint — direct REST, no load_markets() call
BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"

TARGET_UP_PCT   = 0.0015          # +0.15% min gain over next 15 bars
TARGET_DN_PCT   = 0.0010          # -0.10% max drawdown over next 15 bars
LOOKAHEAD       = 15              # bars to evaluate the outcome

RSI_LEN         = 14
ATR_LEN         = 14
BOS_WIN         = 20
SWEEP_WIN       = 20


# ---------------------------------------------------------------------------
# 1. Historical Data Harvesting (direct httpx — no ccxt load_markets overhead)
# ---------------------------------------------------------------------------
async def fetch_candles(symbol: str, total: int, batch: int) -> pd.DataFrame:
    """
    Download `total` 1m candles from Binance Futures klines REST endpoint
    using httpx directly. This avoids ccxt's load_markets()/exchangeInfo
    pre-flight which is the call that triggers geo-restrictions.
    """
    all_rows: list = []
    end_ms: int = int(time.time() * 1000)

    logger.info(f"Fetching {total:,} x {TIMEFRAME} candles for {symbol} via REST...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(all_rows) < total:
            params = {
                "symbol": symbol,
                "interval": TIMEFRAME,
                "limit": min(batch, total - len(all_rows)),
                "endTime": end_ms,
            }
            resp = await client.get(BINANCE_KLINES_URL, params=params)
            resp.raise_for_status()
            candles = resp.json()

            if not candles:
                break

            all_rows = candles + all_rows          # prepend older data
            end_ms   = int(candles[0][0]) - 1     # step one ms before earliest

            logger.debug(
                f"  batch={len(candles)} | total={len(all_rows):,} | "
                f"earliest={pd.to_datetime(candles[0][0], unit='ms')}"
            )

            if len(candles) < batch:
                break

    # Binance returns 12 fields; we only need the first 6
    df = pd.DataFrame(all_rows[-total:], columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "_ct", "_cq", "_nt", "_tbv", "_tqv", "_ignore",
    ]).drop(columns=["_ct", "_cq", "_nt", "_tbv", "_tqv", "_ignore"])

    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.success(
        f"Downloaded {len(df):,} candles  "
        f"[{pd.to_datetime(df['timestamp'].iloc[0], unit='ms')} -> "
        f"{pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}]"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Vectorised Feature Engineering
# ---------------------------------------------------------------------------
def _vec_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta   = close.diff()
    gain    = delta.where(delta > 0, 0.0)
    loss    = (-delta).where(delta < 0, 0.0)
    avg_g   = gain.rolling(length, min_periods=length).mean()
    avg_l   = loss.rolling(length, min_periods=length).mean()
    rs      = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _vec_ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _vec_macd(close: pd.Series):
    fast    = close.ewm(span=12, adjust=False).mean()
    slow    = close.ewm(span=26, adjust=False).mean()
    line    = fast - slow
    signal  = line.ewm(span=9, adjust=False).mean()
    hist    = line - signal
    return line, signal, hist


def _vec_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()


def _vec_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    atr = tr.ewm(span=length, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=length, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=length, adjust=False).mean() / atr)

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0.0)
    adx = dx.ewm(span=length, adjust=False).mean()
    
    return adx.fillna(0.0)


def _vec_chop(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(window=length).sum()
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()

    range_hl = highest_high - lowest_low
    range_hl = range_hl.replace(0, np.nan)

    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(length)
    return chop.fillna(0.0)


def _vec_session_vwap(df: pd.DataFrame) -> pd.Series:
    """Crypto: single continuous VWAP over the whole dataset."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def _vec_cvd(df: pd.DataFrame) -> pd.Series:
    direction = np.where(df["close"] >= df["open"], 1, -1)
    return (df["volume"] * direction).cumsum()


def _vec_fvg(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vectorised FVG: compare current low/high against candle-1 high/low (shifted)."""
    c1_high = df["high"].shift(2)       # candle 1 (iloc[-3] in the live fn)
    c1_low  = df["low"].shift(2)
    c3_low  = df["low"]                 # candle 3 (current)
    c3_high = df["high"]
    close   = df["close"].replace(0, np.nan)

    bull_gap = c3_low  - c1_high
    bear_gap = c1_low  - c3_high

    signal = pd.Series(0.0, index=df.index)
    pct    = pd.Series(0.0, index=df.index)

    bull_mask = bull_gap > 0
    bear_mask = (bear_gap > 0) & ~bull_mask

    signal[bull_mask] =  1.0
    signal[bear_mask] = -1.0
    pct[bull_mask]    = (bull_gap[bull_mask] / close[bull_mask]).clip(0)
    pct[bear_mask]    = (bear_gap[bear_mask] / close[bear_mask]).clip(0)

    return signal, pct


def _vec_bos(df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
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


def _vec_liquidity_sweep(df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
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


def build_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorise all indicators over the full 20k-row DataFrame.
    Output columns match MODEL_FEATURE_COLUMNS exactly.
    """
    out = df.copy()

    # ── Core indicators ─────────────────────────────────────────────────────
    out["RSI"]          = _vec_rsi(out["close"], RSI_LEN)
    out["EMA_14"]       = _vec_ema(out["close"], 14)
    out["EMA_50"]       = _vec_ema(out["close"], 50)
    macd, macd_sig, macd_hist = _vec_macd(out["close"])
    out["MACD"]         = macd
    out["MACD_signal"]  = macd_sig
    out["MACD_hist"]    = macd_hist
    out["VWAP"]         = _vec_session_vwap(out)
    out["ATR"]          = _vec_atr(out, ATR_LEN)
    out["ADX"]          = _vec_adx(out, ATR_LEN)
    out["CHOP"]         = _vec_chop(out, ATR_LEN)
    out["CVD"]          = _vec_cvd(out)

    # ── SMC features ────────────────────────────────────────────────────────
    fvg_sig, fvg_pct    = _vec_fvg(out)
    bos_sig, bos_str    = _vec_bos(out, BOS_WIN)
    sw_sig, sw_rcl      = _vec_liquidity_sweep(out, SWEEP_WIN)

    out["fvg_signal"]               = fvg_sig
    out["fvg_gap_pct"]              = fvg_pct
    out["structure_break_signal"]   = bos_sig
    out["structure_break_strength"] = bos_str
    out["liquidity_sweep_signal"]   = sw_sig
    out["liquidity_reclaim_strength"] = sw_rcl

    # Confluence: FVG + BOS in same direction
    out["structural_confluence"] = np.where(
        (fvg_sig == 1.0) & (bos_sig == 1.0),  1.0,
        np.where(
        (fvg_sig == -1.0) & (bos_sig == -1.0), -1.0, 0.0
        )
    )

    # ── LOB / execution columns — not available in history; zero-filled ─────
    # The model was trained with these but the live feed provides real values.
    for col in ("spread", "best_bid", "best_bid_qty",
                "best_ask", "best_ask_qty",
                "recent_long_liq_vol", "recent_short_liq_vol",
                "liq_imbalance", "macro_sentiment_score"):
        out[col] = 0.0

    out["symbol"]    = SYMBOL.replace("/", "")
    out["timestamp"] = pd.to_datetime(out["timestamp"], unit="ms")

    return out


# ---------------------------------------------------------------------------
# 3. Target Definition (Asymmetric R/R)
# ---------------------------------------------------------------------------
def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label = 1 if within the next LOOKAHEAD bars:
      - max high > close * (1 + TARGET_UP_PCT)   — price reaches the target
      - min low  >= close * (1 - TARGET_DN_PCT)  — stop not breached
    """
    close = df["close"]
    future_high = df["high"].shift(-1).rolling(LOOKAHEAD).max().shift(-(LOOKAHEAD - 1))
    future_low  = df["low"].shift(-1).rolling(LOOKAHEAD).min().shift(-(LOOKAHEAD - 1))

    up_target   = close * (1 + TARGET_UP_PCT)
    dn_stop     = close * (1 - TARGET_DN_PCT)

    df["target"] = np.where(
        (future_high > up_target) & (future_low >= dn_stop), 1, 0
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# 4. Training & Export
# ---------------------------------------------------------------------------
def train_with_cv(df: pd.DataFrame, model_name_suffix: str, base_name: str = "crypto_model") -> None:
    df = df.dropna(subset=MODEL_FEATURE_COLUMNS + ["target"])
    df = df[df["target"].notna()].copy()

    if len(df) < 50:
        logger.warning(f"Insufficient data for {model_name_suffix} regime. Skipping training.")
        return

    pos_rate = df["target"].mean()
    logger.info(
        f"Training set [{model_name_suffix}]: {len(df):,} rows | "
        f"positive rate: {pos_rate:.2%} | "
        f"features: {len(MODEL_FEATURE_COLUMNS)}"
    )

    X = df[MODEL_FEATURE_COLUMNS].fillna(0.0)
    y = df["target"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    best_n_estimators = []
    val_accs = []
    val_lls = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="logloss",
            early_stopping_rounds=10,
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        best_n_estimators.append(model.best_iteration)
        
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        val_accs.append(accuracy_score(y_val, y_pred))
        val_lls.append(log_loss(y_val, y_prob))

    avg_acc = np.mean(val_accs)
    avg_ll = np.mean(val_lls)
    avg_trees = int(np.mean(best_n_estimators))

    logger.success("=" * 55)
    logger.success(f"  Regime     : {model_name_suffix.upper()}")
    logger.success(f"  Avg Acc    : {avg_acc:.4f}")
    logger.success(f"  Avg LogLoss: {avg_ll:.4f}")
    logger.success(f"  Avg Trees  : {avg_trees}")
    logger.success("=" * 55)

    final_model = _BaseXGBoostModel(model_name=f"{base_name}_{model_name_suffix}")
    final_model.model.set_params(n_estimators=max(1, avg_trees))
    final_model.train(df, target_col="target")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
async def main():
    raw_df      = await fetch_candles(SYMBOL, CANDLE_LIMIT, FETCH_BATCH)
    feature_df  = build_historical_features(raw_df)
    labelled_df = build_target(feature_df)
    
    regimes = labelled_df.apply(
        lambda row: classify_regime(row.get("ADX", 0.0), row.get("CHOP", 0.0)),
        axis=1
    )
    
    trend_df = labelled_df[regimes == "TRENDING"].copy()
    chop_df = labelled_df[regimes == "CHOPPY"].copy()
    
    logger.info(f"Split into {len(trend_df)} TRENDING samples and {len(chop_df)} CHOPPY samples.")
    
    train_with_cv(trend_df, "trend", base_name="crypto_model")
    train_with_cv(chop_df, "chop", base_name="crypto_model")


if __name__ == "__main__":
    asyncio.run(main())
