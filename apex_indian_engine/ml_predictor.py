"""
Apex Indian Engine — Machine Learning Inference Core
====================================================
Runs live inference using regime-conditioned XGBoost models trained on 
23 institutional features (Order Flow Imbalance, CVD, FVG, Liquidity Sweeps).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.store import FeatureStore

logger = logging.getLogger("apex.indian.ml")


class IndianMLPredictor:
    """
    Evaluates live 5m candle history and outputs calibrated win probability.
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

        # Fallback to banknifty model if dedicated index model is not present
        if trending_model_path:
            self.trending_path = Path(trending_model_path)
        else:
            cand = Path(default_trend)
            self.trending_path = cand if cand.exists() else Path("data/models/banknifty_india_equity_trending.json")

        if ranging_model_path:
            self.ranging_path = Path(ranging_model_path)
        else:
            cand = Path(default_range)
            self.ranging_path = cand if cand.exists() else Path("data/models/banknifty_india_equity_ranging.json")

        self.feature_store = FeatureStore("india_equity")
        
        self.model_trending: Optional[xgb.Booster] = None
        self.model_ranging: Optional[xgb.Booster] = None
        self._is_loaded = False
        
        self._load_models()

    def _load_models(self) -> None:
        """Load pre-trained XGBoost boosters from disk."""
        try:
            if self.trending_path.exists():
                self.model_trending = xgb.Booster()
                self.model_trending.load_model(str(self.trending_path))
                logger.info(f"Loaded Indian Equity Trending Model: {self.trending_path.name}")

            if self.ranging_path.exists():
                self.model_ranging = xgb.Booster()
                self.model_ranging.load_model(str(self.ranging_path))
                logger.info(f"Loaded Indian Equity Ranging Model: {self.ranging_path.name}")

            self._is_loaded = (self.model_trending is not None or self.model_ranging is not None)
        except Exception as e:
            logger.error(f"Failed to load Indian XGBoost models: {e}")

    def predict(self, candles_df: pd.DataFrame) -> tuple[float, str, float]:
        """
        Takes raw 5-minute candles DataFrame (min 30 rows).
        Returns: (p_bull, regime_name, conviction)
        - p_bull: 0.0 to 1.0 (> 0.50 Call bias, < 0.50 Put bias)
        - regime: "TRENDING" or "RANGING"
        - conviction: 0.0 to 1.0 (|p_bull - 0.50| * 2)
        """
        if not self._is_loaded or len(candles_df) < 25:
            return 0.50, "UNKNOWN", 0.0

        try:
            # 1. Extract 23 institutional features
            X = self.feature_store.build_features(candles_df)
            if X.empty:
                return 0.50, "UNKNOWN", 0.0

            # 2. Determine market regime via ADX & CHOP
            chop = float(X["CHOP"].iloc[-1]) if "CHOP" in X.columns else 50.0
            adx = float(X["ADX"].iloc[-1]) if "ADX" in X.columns else 20.0
            
            is_trending = (adx >= 22.0 and chop <= 55.0)
            regime = "TRENDING" if is_trending else "RANGING"

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

