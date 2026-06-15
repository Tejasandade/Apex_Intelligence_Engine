"""
Apex Intelligence Engine V5 — Purged Time-Series Split
======================================================
Implements Combinatorial Purged Cross-Validation (CPCV) logic
to prevent data leakage caused by rolling window overlaps in finance.
"""

from __future__ import annotations

import numpy as np


class PurgedTimeSeriesSplit:
    """
    K-Fold Cross Validation for financial time series.
    Enforces a blackout 'embargo' period between training and validation sets
    to ensure rolling-window features (like EMAs) don't leak future data.
    """
    
    def __init__(self, n_splits: int = 5, embargo_size: int = 100):
        """
        Args:
            n_splits: Number of CV folds.
            embargo_size: Number of observations to drop between train and test sets.
                          (Should be >= the maximum lookback window of your features).
        """
        self.n_splits = n_splits
        self.embargo_size = embargo_size

    def split(self, X, y=None, groups=None):
        """
        Generates indices to split data into training and test set.
        Yields:
            train_indices: The training set indices for that split.
            test_indices: The testing set indices for that split.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Split into approximately equal chunks
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            # For the last fold, take all remaining data
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            test_indices = indices[test_start:test_end]
            
            # Train indices are everything outside the test window + embargo
            train_left_end = max(0, test_start - self.embargo_size)
            train_right_start = min(n_samples, test_end + self.embargo_size)
            
            train_indices_left = indices[0:train_left_end]
            train_indices_right = indices[train_right_start:n_samples]
            
            train_indices = np.concatenate([train_indices_left, train_indices_right])
            
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
