import os

import pandas as pd
import xgboost as xgb
from loguru import logger


MODEL_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "RSI",
    "EMA_14",
    "EMA_50",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "VWAP",
    "ATR",
    "spread",
    "CVD",
    "best_bid",
    "best_bid_qty",
    "best_ask",
    "best_ask_qty",
    "recent_long_liq_vol",
    "recent_short_liq_vol",
    "liq_imbalance",
    "fvg_signal",
    "fvg_gap_pct",
    "liquidity_sweep_signal",
    "liquidity_reclaim_strength",
    "structure_break_signal",
    "structure_break_strength",
    "structural_confluence",
    "macro_sentiment_score",
]


class ApexXGBoostModel:
    """
    XGBoost Classifier for predicting directional probability (bullish movement).
    Excels at tabular limit order book and quantitative feature data.
    """

    def __init__(self, weights_dir: str = None, model_name: str = "crypto_model"):
        if weights_dir:
            self.model_path = os.path.join(weights_dir, f"{model_name}.json")
        else:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.model_path = os.path.join(
                root_dir, "data", "models", f"{model_name}.json"
            )

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="logloss",
        )
        self.is_trained = False
        self._last_valid_features = None
        self._load_if_exists()

    def _load_if_exists(self):
        logger.info(f"Attempting to load model from: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                self.is_trained = True
                logger.success(f"Model loaded successfully from {self.model_path}")
            except Exception as exc:
                logger.error(f"Failed to load model weights: {exc}")
        else:
            logger.warning(f"No model file found at {self.model_path}. Model is untrained.")

    def train(self, df: pd.DataFrame, target_col: str = "target"):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Always train in canonical MODEL_FEATURE_COLUMNS order so the booster's
        # internal feature names exactly match what predict() sends at inference.
        available = [c for c in MODEL_FEATURE_COLUMNS if c in df.columns]
        X = df[available].fillna(0.0)
        y = df[target_col]

        logger.info(f"Training XGBoost Model on {len(df)} samples with {len(available)} features...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Training complete.")

        self.model.save_model(self.model_path)
        logger.info(f"Model weights saved to {self.model_path}")

    def predict(self, df: pd.DataFrame) -> float:
        """
        Accepts a single-row DataFrame from DataFetcher.build_dataset(),
        aligns columns to the exact training order, and returns bullish probability.
        """
        if not self.is_trained:
            logger.warning("Model is not trained. Returning default probability of 0.5")
            return 0.5

        try:
            X_infer = df.reindex(columns=["symbol", "timestamp", *MODEL_FEATURE_COLUMNS])
            X_infer = X_infer[MODEL_FEATURE_COLUMNS].fillna(0.0)

            if X_infer["best_bid"].iloc[0] == 0 or X_infer["close"].iloc[0] == 0:
                if self._last_valid_features is not None:
                    logger.warning("Zeroed/corrupted features detected. Falling back to last valid features.")
                    X_infer = self._last_valid_features
                else:
                    return 0.5
            else:
                self._last_valid_features = X_infer.copy()

            probabilities = self.model.predict_proba(X_infer)
            bullish_prob = float(probabilities[0][1])
            return bullish_prob
        except Exception as exc:
            logger.error(f"Prediction error: {exc}")
            return 0.5
