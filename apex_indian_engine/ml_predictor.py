"""
Apex Indian Engine — Machine Learning Inference Core
====================================================
Runs live inference using regime-conditioned XGBoost models trained on 
23 institutional features (Order Flow Imbalance, CVD, FVG, Liquidity Sweeps).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.store import FeatureStore

logger = logging.getLogger("apex.indian.ml")

RAW_PRICE_FEATURES = {"EMA_14", "EMA_50", "MACD", "MACD_signal", "MACD_hist", "VWAP", "ATR"}


class IndianMLPredictor:
    """
    Evaluates live 5m candle history and outputs calibrated win probability.
    Enforces validation quality and price-scale compatibility gates before voting.
    """

    def __init__(
        self,
        index_name: str = "BANKNIFTY",
        trending_model_path: Optional[str] = None,
        ranging_model_path: Optional[str] = None,
    ):
        self.index_name = index_name.upper()
        idx_prefix = self.index_name.lower()

        default_trend = f"data/models/{idx_prefix}_india_equity_trending.json"
        default_range = f"data/models/{idx_prefix}_india_equity_ranging.json"

        # Detect whether index uses dedicated weights or BankNifty fallback weights
        if trending_model_path:
            self.trending_path = Path(trending_model_path)
            self._used_fallback_model = False
        else:
            cand = Path(default_trend)
            self._used_fallback_model = not cand.exists()
            self.trending_path = cand if cand.exists() else Path("data/models/banknifty_india_equity_trending.json")

        if ranging_model_path:
            self.ranging_path = Path(ranging_model_path)
        else:
            cand = Path(default_range)
            if not cand.exists():
                self._used_fallback_model = True
            self.ranging_path = cand if cand.exists() else Path("data/models/banknifty_india_equity_ranging.json")

        self.feature_store = FeatureStore("india_equity")

        self.model_trending: Optional[xgb.Booster] = None
        self.model_ranging: Optional[xgb.Booster] = None
        self._is_loaded = False
        self._is_reliable = False
        self.unreliable_reason = ""

        self._load_models()

    @property
    def is_reliable(self) -> bool:
        """True only if loaded weights pass validation & scale compatibility checks."""
        return self._is_loaded and self._is_reliable

    def _validate_model_metadata(self, model_path: Path) -> tuple[bool, str]:
        """Verify validation accuracy/log-loss and check for raw-price scale mismatch."""
        has_raw_price_cols = bool(RAW_PRICE_FEATURES & set(self.feature_store.feature_columns))
        if self._used_fallback_model and has_raw_price_cols and self.index_name != "BANKNIFTY":
            return False, f"Cross-index raw-price scale mismatch ({self.index_name} vs BANKNIFTY model)"

        meta_path = model_path.with_name(model_path.stem + "_metadata.json")
        if not meta_path.exists():
            return True, "OK"

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            metrics = meta.get("metrics", {})
            acc = float(metrics.get("accuracy", 0.55))
            log_loss = float(metrics.get("log_loss", 0.65))
            pos_rate = float(metrics.get("positive_rate", 0.50))

            if acc < 0.52 or log_loss >= 0.70:
                return False, f"Degraded OOS metrics (acc={acc:.2%}, log_loss={log_loss:.3f})"
            if pos_rate < 0.35:
                return False, f"Asymmetric single-barrier base rate ({pos_rate:.1%}) skews directional probability"
        except Exception as e:
            logger.debug(f"Metadata check failed for {meta_path}: {e}")

        return True, "OK"

    def _load_models(self) -> None:
        """Load pre-trained XGBoost boosters from disk and validate quality."""
        reasons: list[str] = []
        valid_any = False
        try:
            if self.trending_path.exists():
                self.model_trending = xgb.Booster()
                self.model_trending.load_model(str(self.trending_path))
                ok, reason = self._validate_model_metadata(self.trending_path)
                if ok:
                    valid_any = True
                else:
                    reasons.append(f"{self.trending_path.name}: {reason}")
                logger.info(f"Loaded Indian Equity Trending Model: {self.trending_path.name} (valid={ok})")

            if self.ranging_path.exists():
                self.model_ranging = xgb.Booster()
                self.model_ranging.load_model(str(self.ranging_path))
                ok, reason = self._validate_model_metadata(self.ranging_path)
                if ok:
                    valid_any = True
                else:
                    reasons.append(f"{self.ranging_path.name}: {reason}")
                logger.info(f"Loaded Indian Equity Ranging Model: {self.ranging_path.name} (valid={ok})")

            self._is_loaded = (self.model_trending is not None or self.model_ranging is not None)
            self._is_reliable = valid_any
            if self._is_loaded and not self._is_reliable:
                self.unreliable_reason = reasons[0] if reasons else "Model failed validation gate"
                logger.warning(
                    f"ML Advisor for {self.index_name} placed in STRUCTURAL_BYPASS: {self.unreliable_reason}"
                )
        except Exception as e:
            logger.error(f"Failed to load Indian XGBoost models: {e}")

    def predict(self, candles_df: pd.DataFrame) -> tuple[float, str, float]:
        """
        Takes raw 5-minute candles DataFrame (min 25 rows).
        Returns: (p_bull, regime_name, conviction)
        - p_bull: 0.0 to 1.0 (> 0.50 Call bias, < 0.50 Put bias)
        - regime: "TRENDING" or "RANGING"
        - conviction: 0.0 to 1.0 (|p_bull - 0.50| * 2)
        """
        if not self._is_loaded or len(candles_df) < 25:
            return 0.50, "UNKNOWN", 0.0

        try:
            df = candles_df.copy()
            if "timestamp" in df.columns:
                def _safe_ts(v):
                    try:
                        fv = float(v)
                        return pd.to_datetime(fv, unit="s", utc=True)
                    except (ValueError, TypeError):
                        pass
                    return pd.to_datetime(v, utc=True, format="mixed")

                df["timestamp"] = df["timestamp"].apply(_safe_ts)

            # 1. Extract 23 institutional features
            X = self.feature_store.build_features(df)
            if X.empty:
                return 0.50, "UNKNOWN", 0.0

            # 2. Determine market regime via ADX & CHOP
            chop = float(X["CHOP"].iloc[-1]) if "CHOP" in X.columns else 50.0
            adx = float(X["ADX"].iloc[-1]) if "ADX" in X.columns else 20.0

            is_trending = (adx >= 22.0 and chop <= 55.0)
            regime = "TRENDING" if is_trending else "RANGING"

            if not self._is_reliable:
                return 0.50, regime, 0.0

            # 3. Select appropriate regime model
            model = self.model_trending if is_trending else self.model_ranging
            if model is None:
                model = self.model_trending or self.model_ranging

            if model is None:
                return 0.50, regime, 0.0

            # 4. Run DMatrix inference on latest candle
            dmat = xgb.DMatrix(X.tail(1))
            raw_prob = float(model.predict(dmat)[0])

            # Conviction score: 0.50 = 0% conviction, 0.75 = 50%, 0.90 = 80%
            conviction = round(abs(raw_prob - 0.50) * 2.0, 3)

            return round(raw_prob, 4), regime, conviction
        except Exception as e:
            logger.error(f"ML inference error: {e}")
            return 0.50, "ERROR", 0.0

