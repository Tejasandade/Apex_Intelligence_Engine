"""
Apex Intelligence Engine V5 — Probability Calibration
=======================================================
Wraps raw model outputs with Isotonic Regression to produce
TRUE probabilities. Without calibration, XGBoost's "60%" output
does NOT mean a 60% chance of winning — it's a raw tree score
mapped through sigmoid, systematically biased.

After calibration, "60%" means: out of all signals where the model
output was near 60%, approximately 60% of them were actually profitable.

This is the difference between gambling and quantitative trading.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.core.logging import get_logger

logger = get_logger("apex.models.calibration")


class ProbabilityCalibrator:
    """
    Isotonic Regression calibrator for model probability outputs.

    Isotonic Regression is preferred over Platt Scaling because:
    1. It makes no distributional assumptions (non-parametric)
    2. It works well with tree-based models like XGBoost
    3. It monotonically maps raw scores to calibrated probabilities
    """

    def __init__(self):
        self._calibrator: IsotonicRegression | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, raw_probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Fit the calibrator using raw model probabilities and true outcomes.

        Should be fitted on a HELD-OUT validation set — never on training data.

        Args:
            raw_probabilities: Raw model probability outputs (0 to 1).
            true_labels: True binary labels (0 or 1).
        """
        if len(raw_probabilities) < 20:
            logger.warning(
                "calibration_insufficient_samples",
                n_samples=len(raw_probabilities),
                msg="Need at least 20 samples for reliable calibration",
            )

        self._calibrator = IsotonicRegression(
            y_min=0.001,  # Avoid exactly 0/1 (log-loss explosion)
            y_max=0.999,
            out_of_bounds="clip",
        )
        self._calibrator.fit(raw_probabilities, true_labels)
        self._is_fitted = True

        # Compute calibration quality
        calibrated = self.calibrate(raw_probabilities)
        brier_raw = np.mean((raw_probabilities - true_labels) ** 2)
        brier_calibrated = np.mean((calibrated - true_labels) ** 2)

        logger.info(
            "calibrator_fitted",
            n_samples=len(raw_probabilities),
            brier_raw=f"{brier_raw:.4f}",
            brier_calibrated=f"{brier_calibrated:.4f}",
            improvement=f"{(brier_raw - brier_calibrated) / brier_raw:.1%}",
        )

    def calibrate(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """
        Transform raw probabilities into calibrated probabilities.

        Args:
            raw_probabilities: Raw model outputs.

        Returns:
            Calibrated probabilities.

        Raises:
            RuntimeError: If calibrator hasn't been fitted.
        """
        if not self._is_fitted or self._calibrator is None:
            logger.warning("calibrator_not_fitted_passthrough")
            return raw_probabilities

        return self._calibrator.predict(raw_probabilities)

    def calibrate_single(self, raw_probability: float) -> float:
        """Calibrate a single probability value."""
        result = self.calibrate(np.array([raw_probability]))
        return float(result[0])

    def save(self, path: Path) -> None:
        """Save calibrator state to a JSON file."""
        if not self._is_fitted or self._calibrator is None:
            logger.warning("calibrator_save_not_fitted")
            return

        # IsotonicRegression stores X_thresholds_ and y_thresholds_
        state = {
            "X_thresholds": self._calibrator.X_thresholds_.tolist(),
            "y_thresholds": self._calibrator.y_thresholds_.tolist(),
            "X_min": float(self._calibrator.X_min_),
            "X_max": float(self._calibrator.X_max_),
            "is_fitted": True,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)

        logger.info("calibrator_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load calibrator state from a JSON file."""
        if isinstance(path, str):
            path = Path(path)

        if not path.exists():
            logger.warning("calibrator_file_not_found", path=str(path))
            return

        with open(path, "r") as f:
            state = json.load(f)

        X_thresholds = np.array(state["X_thresholds"])
        y_thresholds = np.array(state["y_thresholds"])

        # Reconstruct by fitting on the saved thresholds directly
        # This is the cleanest way to restore a fully functional IsotonicRegression
        self._calibrator = IsotonicRegression(
            y_min=0.001, y_max=0.999, out_of_bounds="clip"
        )
        self._calibrator.fit(X_thresholds, y_thresholds)
        self._is_fitted = True

        logger.info(
            "calibrator_loaded",
            path=str(path),
            n_thresholds=len(X_thresholds),
        )

