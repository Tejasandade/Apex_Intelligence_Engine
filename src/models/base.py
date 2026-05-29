"""
Apex Intelligence Engine V5 — Abstract Base Model
===================================================
All prediction models must implement this interface.
Ensures consistent API for training, inference, calibration, and persistence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all Apex prediction models.

    Every model — XGBoost, neural net, ensemble — must implement this
    interface to be compatible with the backtester, council, and registry.
    """

    @abstractmethod
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Train the model on feature matrix X and target y.

        Args:
            X: Feature matrix (rows=samples, columns=features).
            y: Target labels (0/1 for classification).
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Dict of training metrics (accuracy, log_loss, etc.).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> float:
        """
        Predict the probability of a positive outcome for the given features.

        Args:
            X: Feature matrix (single row or small batch).

        Returns:
            Calibrated probability (0.0 to 1.0).
            For single row input, returns a scalar float.
        """
        ...

    @abstractmethod
    def predict_batch(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict probabilities for a batch of samples.

        Args:
            X: Feature matrix (multiple rows).

        Returns:
            Series of calibrated probabilities.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model weights and metadata to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model weights and metadata from disk."""
        ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores."""
        ...

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Whether the model has been trained."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Unique model identifier."""
        ...

    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Model metadata: version, training date, feature count, etc."""
        ...
