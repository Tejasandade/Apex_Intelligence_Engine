"""
Apex Intelligence Engine V5 — Purged Walk-Forward Cross-Validation
===================================================================
Standard K-Fold CV leaks future information into the training set
because financial data is time-ordered. Purged Walk-Forward prevents this:

1. Train on bars [0, split_point)
2. PURGE: remove bars within a gap window around the split (no leakage)
3. Validate on bars [split_point + purge_gap, split_point + val_size)
4. Walk the split point forward and repeat

This produces realistic out-of-sample performance estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.logging import get_logger

logger = get_logger("apex.models.training.cv")


@dataclass
class CVFold:
    """Indices for a single cross-validation fold."""

    fold_num: int
    train_indices: np.ndarray
    val_indices: np.ndarray


def purged_walk_forward_splits(
    n_samples: int,
    n_folds: int = 5,
    purge_gap: int = 15,
    min_train_size: int = 1000,
) -> list[CVFold]:
    """
    Generate purged walk-forward cross-validation fold indices.

    Unlike standard K-Fold:
    - Training data always comes BEFORE validation data (temporal order)
    - A purge gap separates train/val to prevent label leakage
    - Each fold uses more training data than the previous (expanding window)

    Args:
        n_samples: Total number of samples.
        n_folds: Number of validation folds.
        purge_gap: Number of samples to purge between train and validation.
        min_train_size: Minimum training set size.

    Returns:
        List of CVFold objects with train/val index arrays.
    """
    if n_samples < min_train_size + purge_gap + 50:
        logger.warning(
            "cv_insufficient_data",
            n_samples=n_samples,
            min_required=min_train_size + purge_gap + 50,
        )
        # Fall back to a single split
        split = int(n_samples * 0.8)
        return [
            CVFold(
                fold_num=0,
                train_indices=np.arange(0, split - purge_gap),
                val_indices=np.arange(split, n_samples),
            )
        ]

    # Calculate fold boundaries
    val_size = (n_samples - min_train_size - purge_gap) // n_folds
    if val_size < 50:
        val_size = 50
        n_folds = max(1, (n_samples - min_train_size - purge_gap) // val_size)

    folds: list[CVFold] = []

    for i in range(n_folds):
        val_end = n_samples - (n_folds - i - 1) * val_size
        val_start = val_end - val_size
        train_end = val_start - purge_gap

        if train_end < min_train_size // 2:
            continue

        train_indices = np.arange(0, train_end)
        val_indices = np.arange(val_start, val_end)

        folds.append(
            CVFold(
                fold_num=i,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

        logger.debug(
            "cv_fold_created",
            fold=i,
            train_size=len(train_indices),
            val_size=len(val_indices),
            purge_gap=purge_gap,
            train_range=f"[0, {train_end})",
            val_range=f"[{val_start}, {val_end})",
        )

    logger.info(
        "cv_splits_generated",
        n_folds=len(folds),
        n_samples=n_samples,
        purge_gap=purge_gap,
    )

    return folds
