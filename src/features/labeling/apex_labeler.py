"""
Apex Intelligence Engine — Triple-Barrier Labeler
==================================================
Implements the Triple-Barrier Method from Marcos Lopez de Prado's
"Advances in Financial Machine Learning" (Chapter 3).

Why this matters:
  Standard labels (e.g. "did price go up in 10 bars?") ignore actual
  trade mechanics. Triple-barrier labels match how the system trades:
  - Upper barrier = TP (tp_multiple × ATR)
  - Lower barrier = SL (sl_multiple × ATR)
  - Time barrier  = max hold (time_stop_bars candles)

This produces cleaner, more predictive labels aligned with real P&L.

Usage:
    labeler = ApexTripleBarrierLabeler(tp_multiple=3.0, sl_multiple=1.0)
    df = labeler.label_series(ohlcv_df_with_atr)
    # df now has: label_long, label_short, exit_bar, exit_reason, return_R
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("apex.features.labeling.triple_barrier")


@dataclass
class LabelResult:
    label: int           # 1 = win, 0 = loss
    exit_bar: int        # bars from entry until exit
    exit_reason: str     # "TP", "SL", "TIME"
    return_R: float      # actual return in R-multiples
    atr_at_entry: float  # ATR used to set barriers


class ApexTripleBarrierLabeler:
    """
    Triple-Barrier Label Generation for 5m OHLCV data.

    Parameters:
        tp_multiple    : Take-profit distance = tp_multiple × ATR (default: 3.0)
        sl_multiple    : Stop-loss distance   = sl_multiple × ATR (default: 1.0)
        time_stop_bars : Time barrier — candles before forced close (default: 48 = 4h on 5m)
        min_atr_filter : Min ATR as pct of price to include bar (filters noise)
        atr_period     : ATR lookback period (default: 14)
        use_high_low   : Use bar high/low for barrier checks (not just close)
    """

    def __init__(
        self,
        tp_multiple: float = 3.0,
        sl_multiple: float = 1.0,
        time_stop_bars: int = 48,
        min_atr_filter: float = 0.0005,   # min 0.05% ATR/price ratio
        atr_period: int = 14,
        use_high_low: bool = True,
    ):
        self.tp_multiple    = tp_multiple
        self.sl_multiple    = sl_multiple
        self.time_stop_bars = time_stop_bars
        self.min_atr_filter = min_atr_filter
        self.atr_period     = atr_period
        self.use_high_low   = use_high_low

    def label_series(
        self,
        df: pd.DataFrame,
        direction: Literal["long", "short", "both"] = "both",
    ) -> pd.DataFrame:
        """
        Apply triple-barrier labeling to the full DataFrame.

        Args:
            df: OHLCV DataFrame. Must have columns:
                open, high, low, close, volume
                (ATR is computed internally if not present)
            direction: Which direction to label.

        Returns:
            df with added columns:
                label_long    : 1=win, 0=loss (NaN if filtered)
                label_short   : 1=win, 0=loss (NaN if filtered)
                exit_bar_long / exit_bar_short
                exit_reason_long / exit_reason_short
                return_R_long / return_R_short
                atr            : ATR at each bar
        """
        df = df.copy()

        # Compute ATR if not present
        if "ATR" not in df.columns:
            df["ATR"] = self._compute_atr(df)

        n = len(df)
        close  = df["close"].values
        high   = df["high"].values if self.use_high_low else close
        low    = df["low"].values  if self.use_high_low else close
        atr    = df["ATR"].values

        # Pre-allocate output arrays
        label_long   = np.full(n, np.nan)
        exit_bar_long = np.full(n, np.nan)
        exit_reason_long = np.full(n, "", dtype=object)
        return_R_long = np.full(n, np.nan)

        label_short   = np.full(n, np.nan)
        exit_bar_short = np.full(n, np.nan)
        exit_reason_short = np.full(n, "", dtype=object)
        return_R_short = np.full(n, np.nan)

        for i in range(n - 1):
            atr_i = atr[i]
            entry = close[i]

            if entry <= 0 or np.isnan(atr_i) or atr_i <= 0:
                continue

            # ATR filter — skip very low-volatility bars (noise)
            if atr_i / entry < self.min_atr_filter:
                continue

            tp_dist = atr_i * self.tp_multiple
            sl_dist = atr_i * self.sl_multiple

            # Look forward up to time_stop_bars
            end_idx = min(i + self.time_stop_bars, n - 1)

            if direction in ("long", "both"):
                res = self._check_barriers(
                    entry=entry,
                    tp_price=entry + tp_dist,
                    sl_price=entry - sl_dist,
                    highs=high[i+1:end_idx+1],
                    lows=low[i+1:end_idx+1],
                    closes=close[i+1:end_idx+1],
                    side="long",
                )
                label_long[i]        = res.label
                exit_bar_long[i]     = res.exit_bar
                exit_reason_long[i]  = res.exit_reason
                return_R_long[i]     = res.return_R

            if direction in ("short", "both"):
                res = self._check_barriers(
                    entry=entry,
                    tp_price=entry - tp_dist,
                    sl_price=entry + sl_dist,
                    highs=high[i+1:end_idx+1],
                    lows=low[i+1:end_idx+1],
                    closes=close[i+1:end_idx+1],
                    side="short",
                )
                label_short[i]        = res.label
                exit_bar_short[i]     = res.exit_bar
                exit_reason_short[i]  = res.exit_reason
                return_R_short[i]     = res.return_R

        df["label_long"]          = label_long
        df["label_short"]         = label_short
        df["exit_bar_long"]       = exit_bar_long
        df["exit_reason_long"]    = exit_reason_long
        df["return_R_long"]       = return_R_long
        df["exit_bar_short"]      = exit_bar_short
        df["exit_reason_short"]   = exit_reason_short
        df["return_R_short"]      = return_R_short

        return df

    def _check_barriers(
        self,
        entry: float,
        tp_price: float,
        sl_price: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        side: Literal["long", "short"],
    ) -> LabelResult:
        """
        Walk forward through bars and check which barrier is hit first.
        Returns a LabelResult with label, exit_bar, exit_reason, return_R.
        """
        for bar_idx, (h, l, c) in enumerate(zip(highs, lows, closes)):
            if side == "long":
                sl_hit = l <= sl_price
                tp_hit = h >= tp_price
            else:
                sl_hit = h >= sl_price
                tp_hit = l <= tp_price

            # Both hit on same bar → check which is closer to open (conservative: SL wins)
            if sl_hit and tp_hit:
                # Ambiguous — take loss (conservative)
                return LabelResult(
                    label=0, exit_bar=bar_idx + 1,
                    exit_reason="SL", return_R=-1.0,
                    atr_at_entry=0.0,
                )

            if tp_hit:
                return LabelResult(
                    label=1, exit_bar=bar_idx + 1,
                    exit_reason="TP",
                    return_R=self.tp_multiple,
                    atr_at_entry=0.0,
                )

            if sl_hit:
                return LabelResult(
                    label=0, exit_bar=bar_idx + 1,
                    exit_reason="SL",
                    return_R=-self.sl_multiple,
                    atr_at_entry=0.0,
                )

        # Time barrier hit — neutral (use close return)
        if len(closes) > 0:
            final_ret = (closes[-1] - entry) / entry if side == "long" else (entry - closes[-1]) / entry
            # Label based on whether we're ahead or behind at time-stop
            ret_R = final_ret * entry / max(1.0, abs(tp_price - entry)) * self.tp_multiple
            label = 1 if final_ret > 0 else 0
        else:
            label, ret_R = 0, 0.0

        return LabelResult(
            label=label, exit_bar=len(closes),
            exit_reason="TIME", return_R=ret_R,
            atr_at_entry=0.0,
        )

    def _compute_atr(self, df: pd.DataFrame, period: int | None = None) -> pd.Series:
        """Compute ATR if not already in DataFrame."""
        p = period or self.atr_period
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window=p, min_periods=p).mean()

    # ── Analysis helpers ───────────────────────────────────────────────────────

    @staticmethod
    def compute_class_balance(labels: pd.Series) -> dict:
        """
        Returns class balance statistics for a label series.

        Returns:
            {
                'total': int, 'class_0': int, 'class_1': int,
                'class_0_pct': float, 'class_1_pct': float,
                'is_balanced': bool,  # True if within 40/60 split
            }
        """
        valid = labels.dropna()
        total = len(valid)
        if total == 0:
            return {"total": 0, "class_0": 0, "class_1": 0,
                    "class_0_pct": 0.0, "class_1_pct": 0.0, "is_balanced": False}

        c0 = int((valid == 0).sum())
        c1 = int((valid == 1).sum())
        p0 = c0 / total
        p1 = c1 / total
        return {
            "total": total,
            "class_0": c0, "class_1": c1,
            "class_0_pct": round(p0 * 100, 1),
            "class_1_pct": round(p1 * 100, 1),
            "is_balanced": 0.35 <= p1 <= 0.65,
        }

    @staticmethod
    def filter_time_stops(df: pd.DataFrame, side: str = "long") -> pd.DataFrame:
        """
        Remove rows where the time barrier was hit (ambiguous labels).
        These are noisiest — removing them improves model quality.
        """
        col = f"exit_reason_{side}"
        if col not in df.columns:
            return df
        return df[df[col] != "TIME"].copy()

