"""
Apex Intelligence Engine V5 — Model Ensemble
===============================================
Combines multiple models into a single prediction via weighted voting.

Why ensemble?
1. Individual models overfit to specific patterns
2. Ensemble smooths out individual model errors
3. Disagreement between models = low conviction = smaller position or no trade
4. Agreement = high conviction = bigger position

Weighting methods:
- EQUAL: All models vote equally
- PERFORMANCE: Weight by recent accuracy (models that are right more get more vote)
- REGIME: Weight by performance in the CURRENT regime specifically

Architecture:
    trending_model  ─┐
    ranging_model   ─┤─→ WeightedVote → Final Probability
    volatile_model  ─┘
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.models.base import BaseModel

logger = get_logger("apex.models.ensemble")


class WeightingMethod(str, Enum):
    EQUAL = "equal"
    PERFORMANCE = "performance"
    REGIME = "regime"
    RL = "rl"


@dataclass
class EnsembleMember:
    """A single model in the ensemble with tracking metadata."""

    model: BaseModel
    regime: str  # Which regime this model was trained for
    weight: float = 1.0

    # Performance tracking (updated live)
    predictions: int = 0
    correct_predictions: int = 0
    total_brier: float = 0.0  # Sum of (pred - actual)^2
    recent_accuracy: float = 0.5  # Rolling accuracy (last 100 predictions)

    # Rolling window for recent performance
    _recent_results: list[bool] = field(default_factory=list)
    _recent_window: int = 100

    def update_performance(self, prediction: float, actual: int) -> None:
        """Update performance tracking after a prediction resolves."""
        self.predictions += 1
        correct = (prediction >= 0.5 and actual == 1) or (prediction < 0.5 and actual == 0)
        if correct:
            self.correct_predictions += 1

        self.total_brier += (prediction - actual) ** 2

        # Rolling window
        self._recent_results.append(correct)
        if len(self._recent_results) > self._recent_window:
            self._recent_results = self._recent_results[-self._recent_window:]

        self.recent_accuracy = sum(self._recent_results) / len(self._recent_results)

    @property
    def lifetime_accuracy(self) -> float:
        if self.predictions == 0:
            return 0.5
        return self.correct_predictions / self.predictions

    @property
    def brier_score(self) -> float:
        if self.predictions == 0:
            return 0.25  # Random baseline
        return self.total_brier / self.predictions


class ModelEnsemble:
    """
    Ensemble of models with weighted voting.

    Combines predictions from multiple models (potentially trained on
    different regimes or time windows) into a single calibrated probability.
    """

    def __init__(
        self,
        weighting: WeightingMethod = WeightingMethod.PERFORMANCE,
        min_weight: float = 0.1,
        disagreement_penalty: float = 0.3,
    ):
        """
        Args:
            weighting: How to weight member votes.
            min_weight: Minimum weight for any member (prevents zero-weight).
            disagreement_penalty: Conviction reduction when models disagree.
        """
        self.weighting = weighting
        self.min_weight = min_weight
        self.disagreement_penalty = disagreement_penalty
        self._members: list[EnsembleMember] = []
        self._rl_weights: dict[str, float] = {}

    def set_rl_weights(self, weights: dict[str, float]) -> None:
        """Inject explicit weights dynamically from the RL Meta-Controller."""
        self._rl_weights = weights

    @property
    def num_members(self) -> int:
        return len(self._members)

    def add_member(
        self,
        model: BaseModel,
        regime: str = "all",
        initial_weight: float = 1.0,
    ) -> None:
        """Add a model to the ensemble."""
        member = EnsembleMember(
            model=model,
            regime=regime,
            weight=initial_weight,
        )
        self._members.append(member)
        logger.info(
            "ensemble_member_added",
            model=model.model_name,
            regime=regime,
            total_members=len(self._members),
        )

    def predict(
        self,
        X: pd.DataFrame,
        current_regime: str = "all",
    ) -> tuple[float, float, dict[str, float]]:
        """
        Generate an ensemble prediction.

        Args:
            X: Feature matrix (1 row).
            current_regime: Current market regime for regime-weighted voting.

        Returns:
            Tuple of:
            - final_probability: Weighted average probability [0, 1]
            - conviction_modifier: Multiplier based on agreement [0, 1]
            - member_predictions: Dict of model_name → individual prediction
        """
        if not self._members:
            return 0.5, 0.0, {}

        predictions: dict[str, float] = {}
        weights: dict[str, float] = {}

        for member in self._members:
            try:
                prob = member.model.predict(X)
                predictions[member.model.model_name] = prob
                member._last_prediction = prob

                # Compute weight based on method
                w = self._compute_weight(member, current_regime)
                weights[member.model.model_name] = w

            except Exception as e:
                logger.warning(
                    "ensemble_member_error",
                    model=member.model.model_name,
                    error=str(e),
                )

        if not predictions:
            return 0.5, 0.0, {}

        # Weighted average
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1.0

        final_prob = sum(
            predictions[name] * weights[name] for name in predictions
        ) / total_weight

        # Conviction modifier based on agreement
        conviction = self._compute_conviction(predictions, final_prob)

        logger.debug(
            "ensemble_prediction",
            final_prob=f"{final_prob:.4f}",
            conviction=f"{conviction:.4f}",
            members=len(predictions),
            regime=current_regime,
            individual={k: f"{v:.4f}" for k, v in predictions.items()},
        )

        return final_prob, conviction, predictions

    def _compute_weight(
        self, member: EnsembleMember, current_regime: str
    ) -> float:
        """Compute the voting weight for a member."""
        if self.weighting == WeightingMethod.EQUAL:
            return 1.0

        elif self.weighting == WeightingMethod.PERFORMANCE:
            # Weight by recent accuracy, with min floor
            w = max(member.recent_accuracy, self.min_weight)
            return w

        elif self.weighting == WeightingMethod.REGIME:
            # Boost models trained for the current regime
            base_weight = max(member.recent_accuracy, self.min_weight)
            if member.regime == current_regime.lower():
                return base_weight * 2.0  # 2x boost for matching regime
            elif member.regime == "all":
                return base_weight * 1.0  # Neutral for general models
            else:
                return base_weight * 0.5  # Penalty for wrong-regime model
                
        elif self.weighting == WeightingMethod.RL:
            # RL agent explicitly sets the weight dynamically
            return self._rl_weights.get(member.model.model_name, self.min_weight)

        return 1.0

    def _compute_conviction(
        self, predictions: dict[str, float], final_prob: float
    ) -> float:
        """
        Compute conviction based on model agreement.

        High agreement (all models agree) → conviction near 1.0
        High disagreement (models contradict) → conviction reduced

        Uses coefficient of variation of predictions.
        """
        if len(predictions) <= 1:
            return 1.0

        probs = list(predictions.values())
        std = np.std(probs)
        mean = np.mean(probs)

        # Coefficient of variation (normalized spread)
        if mean > 0:
            cv = std / mean
        else:
            cv = 1.0

        # Map CV to conviction: CV=0 → conviction=1.0, CV≥1 → conviction≈0
        conviction = max(0.0, 1.0 - cv * self.disagreement_penalty)

        # Also check: are models on the same SIDE?
        bullish = sum(1 for p in probs if p >= 0.5)
        bearish = len(probs) - bullish

        # If models disagree on direction, reduce conviction significantly
        if bullish > 0 and bearish > 0:
            agreement_ratio = max(bullish, bearish) / len(probs)
            conviction *= agreement_ratio

        return round(conviction, 4)

    def update_member_performance(
        self, model_name: str, prediction: float, actual: int
    ) -> None:
        """Update performance tracking for a specific member after trade resolves."""
        for member in self._members:
            if member.model.model_name == model_name:
                member.update_performance(prediction, actual)
                return

    def update_all_performance(self, actual: int) -> None:
        """Update all members with the same actual outcome (they all predicted on the same candle)."""
        for member in self._members:
            if hasattr(member, "_last_prediction") and member._last_prediction is not None:
                member.update_performance(member._last_prediction, actual)
                member._last_prediction = None

    def get_member_stats(self) -> list[dict[str, Any]]:
        """Return performance stats for all ensemble members."""
        stats = []
        for member in self._members:
            stats.append({
                "model": member.model.model_name,
                "regime": member.regime,
                "weight": member.weight,
                "predictions": member.predictions,
                "lifetime_accuracy": member.lifetime_accuracy,
                "recent_accuracy": member.recent_accuracy,
                "brier_score": member.brier_score,
            })
        return stats

    def get_feature_importance(self) -> dict[str, float]:
        """Aggregate feature importance across all members."""
        combined: dict[str, float] = {}
        total_weight = 0.0

        for member in self._members:
            importance = member.model.get_feature_importance()
            w = member.weight
            total_weight += w

            for feat, imp in importance.items():
                combined[feat] = combined.get(feat, 0.0) + imp * w

        # Normalize
        if total_weight > 0:
            combined = {k: v / total_weight for k, v in combined.items()}

        return combined
