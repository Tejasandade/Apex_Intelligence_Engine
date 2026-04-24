import os
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

class ApexXGBoostModel:
    """
    XGBoost Classifier for predicting directional probability (bullish movement).
    Excels at tabular limit order book and quantitative feature data.
    """
    def __init__(self, weights_dir: str = None):
        # Resolve model path relative to this file's location
        if weights_dir:
            self.model_path = os.path.join(weights_dir, "final_model.json")
        else:
            self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_model.json")
        
        # Initialize a basic XGBoost Classifier
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            eval_metric="logloss",
        )
        self.is_trained = False
        
        # Load weights if they already exist
        self._load_if_exists()

    def _load_if_exists(self):
        logger.info(f"Attempting to load model from: {self.model_path}")
        if os.path.exists(self.model_path):
            try:
                self.model.load_model(self.model_path)
                self.is_trained = True
                logger.success(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model weights: {e}")
        else:
            logger.warning(f"No model file found at {self.model_path}. Model is untrained.")

    def train(self, df: pd.DataFrame, target_col: str = 'target'):
        """
        Trains the XGBoost model on historical features and saves the weights.
        Assumes df contains feature columns and the target label.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Drop non-feature columns (like timestamp and symbol) for training
        drop_cols = [target_col, 'timestamp', 'symbol']
        features = [c for c in df.columns if c not in drop_cols]
        
        X = df[features]
        y = df[target_col]

        logger.info(f"Training XGBoost Model on {len(df)} samples...")
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("Training complete.")

        # Save weights locally
        os.makedirs(self.weights_dir, exist_ok=True)
        self.model.save_model(self.model_path)
        logger.info(f"Model weights saved to {self.model_path}")

    def predict(self, df: pd.DataFrame) -> float:
        """
        Accepts a single-row DataFrame from DataFetcher.build_dataset(),
        extracts the 21-feature vector, and returns a bullish probability.
        """
        if not self.is_trained:
            logger.warning("Model is not trained. Returning default probability of 0.5")
            return 0.5

        try:
            # The dataset_builder now provides all 21 features directly
            # Just drop non-feature columns and pass to the model
            drop_cols = ['timestamp', 'symbol']
            features = [c for c in df.columns if c not in drop_cols]
            X_infer = df[features]

            probabilities = self.model.predict_proba(X_infer)
            bullish_prob = float(probabilities[0][1])

            return bullish_prob

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.5

