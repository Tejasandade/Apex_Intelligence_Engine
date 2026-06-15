"""
Apex Intelligence Engine V5 — Regime Detector
===============================================
Classifies the current market regime using ADX and Choppiness Index.
Different regimes require different models and trading strategies:

- TRENDING: ADX > 25 AND CHOP < 61.8 — momentum strategies work
- RANGING:  ADX < 25 AND CHOP > 61.8 — mean-reversion strategies work
- VOLATILE: High ATR relative to recent history — reduce size, widen stops
- QUIET:    Low ATR — tight stops, quick scalps

The regime detector feeds into:
1. Model routing: select the correct model for current conditions
2. Council advisor: regime advisor can veto counter-regime trades
3. Risk sizing: volatile regimes → smaller positions
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from src.core.logging import get_logger

logger = get_logger("apex.models.regime")


class MarketRegime(str, Enum):
    """Market regime classification."""

    TRENDING_HIGH_VOL = "TRENDING_HIGH_VOL"
    TRENDING_LOW_VOL = "TRENDING_LOW_VOL"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"


def classify_regime(
    adx: float,
    chop: float,
    atr: float = 0.0,
    atr_median: float = 0.0,
    adx_threshold: float = 25.0,
    chop_threshold: float = 61.8,
    atr_volatile_multiplier: float = 1.5,
    atr_quiet_multiplier: float = 0.5,
) -> MarketRegime:
    """
    Classify the current market regime from indicator values.

    Priority logic:
    1. If ATR is extreme (>1.5x median), mark VOLATILE regardless
    2. If ATR is very low (<0.5x median), mark QUIET
    3. Otherwise, use ADX + CHOP to classify TRENDING vs RANGING

    Args:
        adx: Current ADX value (0-100).
        chop: Current Choppiness Index value (0-100).
        atr: Current ATR value.
        atr_median: Median ATR over lookback period.
        adx_threshold: ADX level above which market is trending.
        chop_threshold: CHOP level above which market is choppy.
        atr_volatile_multiplier: ATR ratio for volatile classification.
        atr_quiet_multiplier: ATR ratio for quiet classification.

    Returns:
        MarketRegime enum value.
    """
    # Volatility Quadrant Split
    is_high_vol = False
    if atr_median > 0 and atr > 0:
        atr_ratio = atr / atr_median
        if atr_ratio >= 1.0:
            is_high_vol = True
            
    # Trend Quadrant Split
    is_trending = adx >= adx_threshold and chop < chop_threshold
    
    if is_trending:
        return MarketRegime.TRENDING_HIGH_VOL if is_high_vol else MarketRegime.TRENDING_LOW_VOL
    else:
        return MarketRegime.RANGING_HIGH_VOL if is_high_vol else MarketRegime.RANGING_LOW_VOL


def classify_regime_series(
    df: pd.DataFrame,
    adx_col: str = "ADX",
    chop_col: str = "CHOP",
    atr_col: str = "ATR",
    adx_threshold: float = 25.0,
    chop_threshold: float = 61.8,
    atr_lookback: int = 100,
) -> pd.Series:
    """
    Classify regime for every row in a DataFrame. Used in training
    to split data into regime-specific subsets.

    Args:
        df: DataFrame with ADX, CHOP, and optionally ATR columns.
        adx_threshold: ADX trending threshold.
        chop_threshold: CHOP choppy threshold.
        atr_lookback: Rolling window for ATR median computation.

    Returns:
        Series of MarketRegime string values.
    """
    adx = df[adx_col]
    chop = df[chop_col]

    # Compute rolling ATR median for volatility classification
    if atr_col in df.columns:
        atr = df[atr_col]
        atr_median = atr.rolling(atr_lookback, min_periods=1).median()
    else:
        atr = pd.Series(0.0, index=df.index)
        atr_median = pd.Series(0.0, index=df.index)

    regimes = pd.Series("", index=df.index, dtype=str)

    for i in range(len(df)):
        regime = classify_regime(
            adx=float(adx.iloc[i]),
            chop=float(chop.iloc[i]),
            atr=float(atr.iloc[i]),
            atr_median=float(atr_median.iloc[i]),
            adx_threshold=adx_threshold,
            chop_threshold=chop_threshold,
        )
        regimes.iloc[i] = regime.value

    # Log distribution
    dist = regimes.value_counts().to_dict()
    logger.info("regime_classification_complete", distribution=dist, total=len(df))

    return regimes


class RegimeDetector:
    """
    Stateful regime detector for live inference.
    Maintains a rolling window of indicator values and provides
    the current regime classification.
    """

    def __init__(
        self,
        adx_threshold: float = 25.0,
        chop_threshold: float = 61.8,
        atr_lookback: int = 100,
    ):
        self.adx_threshold = adx_threshold
        self.chop_threshold = chop_threshold
        self.atr_lookback = atr_lookback
        self._atr_history: list[float] = []
        self._current_regime: MarketRegime = MarketRegime.RANGING_LOW_VOL

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    def update(self, adx: float, chop: float, atr: float) -> MarketRegime:
        """
        Update with latest indicator values and return current regime.

        Args:
            adx: Latest ADX value.
            chop: Latest CHOP value.
            atr: Latest ATR value.

        Returns:
            Current market regime.
        """
        self._atr_history.append(atr)
        if len(self._atr_history) > self.atr_lookback:
            self._atr_history = self._atr_history[-self.atr_lookback :]

        atr_median = float(np.median(self._atr_history)) if self._atr_history else 0.0

        self._current_regime = classify_regime(
            adx=adx,
            chop=chop,
            atr=atr,
            atr_median=atr_median,
            adx_threshold=self.adx_threshold,
            chop_threshold=self.chop_threshold,
        )

        return self._current_regime
