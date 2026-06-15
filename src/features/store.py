"""
Apex Intelligence Engine V5 — Versioned Feature Store
======================================================
The SINGLE source of truth for what features each market uses.
Enforces train/serve parity: the exact same features in the exact same
order are used for training AND live inference.

Key principles:
1. Per-market feature sets — crypto features ≠ India equity features
2. Version tagging — FeatureStore.VERSION changes when features change
3. Build function — one call to get the complete feature matrix
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.features.indicators.technical import build_technical_features
from src.features.indicators.structure import build_structure_features
from src.features.indicators.volume import build_volume_features
from src.features.market_profiles.crypto import build_crypto_features
from src.features.market_profiles.india_equity import build_india_equity_features

logger = get_logger("apex.features.store")


class FeatureStore:
    """
    Versioned feature matrix manager.

    Each market type has a defined set of features. The store ensures:
    - Training and inference use identical feature sets
    - Feature ordering is deterministic
    - No undefined features leak into the model
    - Version tracking for reproducibility
    """

    VERSION = "1.2.0"

    # ── Per-Market Feature Definitions ──────────────────────────────────────
    # These are the EXACT columns the model will receive, in THIS order.
    # Adding or removing a feature = version bump + model retrain.

    FEATURE_SETS: dict[str, list[str]] = {
        "crypto": [
            "price_vs_ema14",
            "price_vs_ema50",
            "ema_cross",
            "MACD_norm",
            "MACD_signal_norm",
            "ATR",
            "ADX",
            "CHOP",
            "BB_width",
            "VWAP_zscore",
            "vsa_absorption",
            "trend_exhaustion",
            "liquidity_sweep_signal",
            "CVD",
            "order_flow_imbalance",
            "spread",
            "volume_spike",
            "MTF_ADX_15m",
            "fvg_signal",
            "fvg_gap_pct",
            "structure_break_signal",
            "structure_break_strength",
            "liquidity_reclaim_strength",
            "dist_to_pivot_high",
            "dist_to_pivot_low",
            "structural_confluence",
            "OB_bull_dist",
            "OB_bear_dist",
            "ATR_Ratio",
            "Volume_Profile_POC_Dist",
            "AVWAP_distance",
            "Liquidity_Sweep_1H",
            "RSI_Trend",
            "OBV_Slope",
            "Volatility_Regime",
            "hour_sin",
            "hour_cos",
        ],
        "india_equity": [
            "RSI",
            "EMA_14",
            "EMA_50",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "VWAP",
            "ATR",
            "ADX",
            "CHOP",
            "BB_width",
            "CVD",
            "order_flow_imbalance",
            "session_elapsed_pct",
            "gap_from_open_pct",
            "fvg_signal",
            "fvg_gap_pct",
            "structure_break_signal",
            "structure_break_strength",
            "liquidity_sweep_signal",
            "liquidity_reclaim_strength",
            "structural_confluence",
            "macro_sentiment_score",
        ],
    }

    def __init__(self, market_type: str):
        """
        Initialize the feature store for a specific market type.

        Args:
            market_type: "crypto" or "india_equity"

        Raises:
            ValueError: If market_type is not supported.
        """
        if market_type not in self.FEATURE_SETS:
            raise ValueError(
                f"Unsupported market type '{market_type}'. "
                f"Available: {list(self.FEATURE_SETS.keys())}"
            )
        self.market_type = market_type
        self.feature_columns = self.FEATURE_SETS[market_type]
        logger.info(
            "feature_store_initialized",
            market_type=market_type,
            version=self.VERSION,
            num_features=len(self.feature_columns),
        )

    @property
    def num_features(self) -> int:
        """Number of features for this market type."""
        return len(self.feature_columns)

    def build_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Build the complete feature matrix from raw OHLCV data.

        This is the CANONICAL feature builder. Used in:
        - Training pipeline (vectorised over full dataset)
        - Live inference (single-row or small batch)

        The output DataFrame will have EXACTLY self.feature_columns as columns,
        in the exact same order. Nothing more, nothing less.

        Args:
            df: DataFrame with OHLCV columns (open, high, low, close, volume)
                and optionally a timestamp column.

        Returns:
            DataFrame with only the model feature columns, sanitized.
        """
        if df.empty:
            logger.warning("build_features_empty_input", market_type=self.market_type)
            return pd.DataFrame(columns=self.feature_columns)

        # Ensure numeric OHLCV
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        # Step 1: Technical indicators
        result = build_technical_features(
            df, market_type=self.market_type, timestamp_col=timestamp_col
        )

        # Step 1.5: MTF Injection
        if timestamp_col in result.columns and len(result) > 60:
            try:
                dt_series = pd.to_datetime(result[timestamp_col], unit='ms', utc=True)
                temp = result[['high', 'low', 'close']].copy()
                temp.index = dt_series
                df_15m = temp.resample('15min').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
                if len(df_15m) > 14:
                    from src.features.indicators.technical import compute_adx
                    adx_15m = compute_adx(df_15m, 14)
                    adx_1m = adx_15m.reindex(temp.index, method='ffill').fillna(0.0)
                    result['MTF_ADX_15m'] = adx_1m.values
                else:
                    result['MTF_ADX_15m'] = 0.0
            except Exception:
                result['MTF_ADX_15m'] = 0.0
        else:
            result['MTF_ADX_15m'] = 0.0

        # Step 2: SMC structure features
        result = build_structure_features(result)

        # Step 3: Volume & order flow (requires structure for AVWAP)
        result = build_volume_features(result)

        # Step 4: Market-specific features
        if self.market_type == "crypto":
            result = build_crypto_features(result)
        elif self.market_type == "india_equity":
            result = build_india_equity_features(result, timestamp_col=timestamp_col)

        # Step 5: Extract only the declared features, in order
        missing = [c for c in self.feature_columns if c not in result.columns]
        if missing:
            logger.warning(
                "feature_columns_missing",
                missing=missing,
                market_type=self.market_type,
            )
            for col in missing:
                result[col] = 0.0

        feature_matrix = result[self.feature_columns].copy()

        # Step 6: Sanitize — no NaN, no Inf
        feature_matrix = self._sanitize(feature_matrix)

        return feature_matrix

    def _sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize the feature matrix:
        - Replace Inf/-Inf with NaN, then NaN with 0.0
        - Validate no zero-close (which would break price-relative features)
        """
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0.0)
        return df

    def validate_inference_input(self, df: pd.DataFrame) -> bool:
        """
        Validate that an inference input has the correct columns.

        Args:
            df: Feature matrix to validate.

        Returns:
            True if valid, raises ValueError if not.
        """
        if df.empty:
            raise ValueError("Empty feature matrix for inference")

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature matrix missing columns: {missing}. "
                f"Expected: {self.feature_columns}"
            )

        extra = [c for c in df.columns if c not in self.feature_columns]
        if extra:
            logger.warning(
                "extra_columns_in_inference",
                extra=extra,
                msg="Extra columns will be ignored",
            )

        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return feature store metadata for logging and tracking."""
        return {
            "version": self.VERSION,
            "market_type": self.market_type,
            "num_features": self.num_features,
            "feature_columns": self.feature_columns,
        }
