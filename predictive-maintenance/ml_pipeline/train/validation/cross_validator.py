"""
Cross-Validation for Time-Series Models
Implements time-series aware and equipment-based CV to prevent data leakage
"""

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from typing import Dict, List, Tuple, Generator, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesCrossValidator:
    """
    Time-series aware cross-validation
    Prevents data leakage by respecting temporal ordering
    """

    def __init__(self, config: Dict):
        """
        Initialize CV

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.cv_config = config.get("cross_validation", {})
        logger.info("Time Series Cross Validator initialized")

    def time_series_split(
        self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, gap: int = 0
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Time-series split with gap between train and test

        Args:
            X: Features
            y: Targets
            n_splits: Number of splits
            gap: Gap between train and test sets (in samples)

        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        test_size = n_samples // (n_splits + 1)

        logger.info(f"Time-series CV: {n_splits} splits, gap={gap}")

        for i in range(n_splits):
            # Train on all data up to test start
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            # Apply gap
            train_end = test_start - gap

            if train_end <= 0 or test_end > n_samples:
                continue

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)

            logger.debug(
                f"Split {i + 1}: Train size={len(train_indices)}, Test size={len(test_indices)}"
            )

            yield train_indices, test_indices

    def equipment_based_split(
        self, equipment_ids: np.ndarray, n_splits: int = 5
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Equipment-based split - each fold has different equipment
        Prevents data leakage across equipment

        Args:
            equipment_ids: Array of equipment identifiers
            n_splits: Number of splits

        Yields:
            (train_indices, test_indices) tuples
        """
        unique_equipment = np.unique(equipment_ids)
        n_equipment = len(unique_equipment)

        logger.info(f"Equipment-based CV: {n_splits} splits, {n_equipment} equipment")

        # Shuffle equipment
        np.random.seed(42)
        shuffled_equipment = np.random.permutation(unique_equipment)

        fold_size = n_equipment // n_splits

        for i in range(n_splits):
            # Test equipment for this fold
            test_eq_start = i * fold_size
            test_eq_end = test_eq_start + fold_size if i < n_splits - 1 else n_equipment
            test_equipment = shuffled_equipment[test_eq_start:test_eq_end]

            # Train on all other equipment
            train_equipment = np.setdiff1d(shuffled_equipment, test_equipment)

            # Get indices
            train_indices = np.where(np.isin(equipment_ids, train_equipment))[0]
            test_indices = np.where(np.isin(equipment_ids, test_equipment))[0]

            logger.debug(
                f"Split {i + 1}: Train equipment={len(train_equipment)}, Test equipment={len(test_equipment)}"
            )
            logger.debug(
                f"Split {i + 1}: Train size={len(train_indices)}, Test size={len(test_indices)}"
            )

            yield train_indices, test_indices

    def combined_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        equipment_ids: np.ndarray,
        cycle_numbers: np.ndarray,
        n_splits: int = 5,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Combined time-series and equipment-based split
        First splits by equipment, then by time within each equipment

        Args:
            X: Features
            y: Targets
            equipment_ids: Equipment identifiers
            cycle_numbers: Cycle numbers for temporal ordering
            n_splits: Number of splits

        Yields:
            (train_indices, test_indices) tuples
        """
        unique_equipment = np.unique(equipment_ids)
        n_equipment = len(unique_equipment)

        logger.info(f"Combined CV: {n_splits} splits")

        # Split equipment into folds
        np.random.seed(42)
        shuffled_equipment = np.random.permutation(unique_equipment)
        fold_size = n_equipment // n_splits

        for i in range(n_splits):
            test_eq_start = i * fold_size
            test_eq_end = test_eq_start + fold_size if i < n_splits - 1 else n_equipment
            test_equipment = shuffled_equipment[test_eq_start:test_eq_end]
            train_equipment = np.setdiff1d(shuffled_equipment, test_equipment)

            train_indices = []
            test_indices = []

            # For train equipment: use earlier cycles
            for eq in train_equipment:
                eq_mask = equipment_ids == eq
                eq_indices = np.where(eq_mask)[0]
                eq_cycles = cycle_numbers[eq_mask]

                # Sort by cycle
                sorted_idx = eq_indices[np.argsort(eq_cycles)]

                # Use 80% for training
                split_point = int(len(sorted_idx) * 0.8)
                train_indices.extend(sorted_idx[:split_point])

            # For test equipment: use all cycles
            for eq in test_equipment:
                eq_mask = equipment_ids == eq
                eq_indices = np.where(eq_mask)[0]
                test_indices.extend(eq_indices)

            train_indices = np.array(train_indices)
            test_indices = np.array(test_indices)

            logger.debug(
                f"Split {i + 1}: Train size={len(train_indices)}, Test size={len(test_indices)}"
            )

            yield train_indices, test_indices

    def cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "time_series",
        equipment_ids: Optional[np.ndarray] = None,
        cycle_numbers: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            model: Model to evaluate (must have fit and predict methods)
            X: Features
            y: Targets
            method: 'time_series', 'equipment_based', or 'combined'
            equipment_ids: Equipment identifiers (for equipment-based or combined)
            cycle_numbers: Cycle numbers (for combined)

        Returns:
            Dictionary with CV results
        """
        n_splits = self.cv_config.get("n_splits", 5)

        # Get splits
        if method == "time_series":
            gap = self.cv_config.get("time_series", {}).get("gap", 10)
            splits = self.time_series_split(X, y, n_splits, gap)
        elif method == "equipment_based":
            if equipment_ids is None:
                raise ValueError("equipment_ids required for equipment-based CV")
            splits = self.equipment_based_split(equipment_ids, n_splits)
        elif method == "combined":
            if equipment_ids is None or cycle_numbers is None:
                raise ValueError(
                    "equipment_ids and cycle_numbers required for combined CV"
                )
            splits = self.combined_split(X, y, equipment_ids, cycle_numbers, n_splits)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Evaluate each fold
        fold_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Evaluating fold {fold_idx + 1}/{n_splits}")

            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]

            # Train model
            model.fit(X_train_fold, y_train_fold)

            # Predict
            y_pred_fold = model.predict(X_test_fold)

            # Calculate metrics (MSE for regression)
            mse = np.mean((y_test_fold - y_pred_fold) ** 2)
            mae = np.mean(np.abs(y_test_fold - y_pred_fold))

            fold_scores.append(
                {
                    "fold": fold_idx + 1,
                    "mse": float(mse),
                    "mae": float(mae),
                    "train_size": len(train_idx),
                    "test_size": len(test_idx),
                }
            )

            logger.info(f"Fold {fold_idx + 1}: MSE={mse:.4f}, MAE={mae:.4f}")

        # Summary
        mse_scores = [s["mse"] for s in fold_scores]
        mae_scores = [s["mae"] for s in fold_scores]

        results = {
            "method": method,
            "n_splits": n_splits,
            "fold_scores": fold_scores,
            "mean_mse": float(np.mean(mse_scores)),
            "std_mse": float(np.std(mse_scores)),
            "mean_mae": float(np.mean(mae_scores)),
            "std_mae": float(np.std(mae_scores)),
        }

        logger.info(
            f"CV Results: MSE={results['mean_mse']:.4f} (+/- {results['std_mse']:.4f})"
        )

        return results
