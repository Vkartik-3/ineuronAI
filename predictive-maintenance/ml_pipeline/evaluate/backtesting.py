"""
Backtesting Framework for Time-Series Models
Walk-forward validation and performance tracking over time
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BacktestFramework:
    """
    Backtesting framework for predictive maintenance models
    """

    def __init__(self):
        logger.info("Backtest Framework initialized")

    def walk_forward_validation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        equipment_ids: np.ndarray,
        cycle_numbers: np.ndarray,
        train_size: int = 200,
        test_size: int = 50,
        step: int = 25,
    ) -> Dict:
        """
        Perform walk-forward validation

        Args:
            model: Model object with fit() and predict() methods
            X: Features
            y: Targets
            equipment_ids: Equipment identifiers
            cycle_numbers: Cycle numbers
            train_size: Training window size
            test_size: Test window size
            step: Step size for moving window

        Returns:
            Dictionary with backtest results
        """
        logger.info(
            f"Starting walk-forward validation: train={train_size}, test={test_size}, step={step}"
        )

        n_samples = len(X)
        results = []

        current_pos = 0
        fold = 0

        while current_pos + train_size + test_size <= n_samples:
            fold += 1

            # Define windows
            train_start = current_pos
            train_end = current_pos + train_size
            test_start = train_end
            test_end = test_start + test_size

            # Get data
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            # Train model
            logger.info(f"Fold {fold}: Training on cycles {train_start}-{train_end}")
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Calculate metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            results.append(
                {
                    "fold": fold,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                }
            )

            logger.info(f"Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}")

            # Move window
            current_pos += step

        # Summary statistics
        summary = {
            "n_folds": len(results),
            "mean_mae": float(np.mean([r["mae"] for r in results])),
            "std_mae": float(np.std([r["mae"] for r in results])),
            "mean_rmse": float(np.mean([r["rmse"] for r in results])),
            "std_rmse": float(np.std([r["rmse"] for r in results])),
            "fold_results": results,
        }

        logger.info(f"Walk-forward validation complete: {len(results)} folds")
        logger.info(
            f"Mean MAE: {summary['mean_mae']:.2f} (+/- {summary['std_mae']:.2f})"
        )

        return summary

    def expanding_window_validation(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        initial_train_size: int = 200,
        test_size: int = 50,
        step: int = 25,
    ) -> Dict:
        """
        Perform expanding window validation
        Training set grows while test set slides

        Args:
            model: Model object
            X: Features
            y: Targets
            initial_train_size: Initial training size
            test_size: Test window size
            step: Step size

        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting expanding window validation")

        n_samples = len(X)
        results = []
        fold = 0

        current_pos = initial_train_size

        while current_pos + test_size <= n_samples:
            fold += 1

            # Training on all data up to current position
            X_train = X[:current_pos]
            y_train = y[:current_pos]

            # Test on next window
            test_start = current_pos
            test_end = current_pos + test_size
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            # Train
            logger.info(f"Fold {fold}: Training on {len(X_train)} samples")
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            results.append(
                {
                    "fold": fold,
                    "train_size": len(X_train),
                    "test_start": test_start,
                    "test_end": test_end,
                    "mae": float(mae),
                    "rmse": float(rmse),
                }
            )

            logger.info(f"Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}")

            current_pos += step

        summary = {
            "n_folds": len(results),
            "mean_mae": float(np.mean([r["mae"] for r in results])),
            "std_mae": float(np.std([r["mae"] for r in results])),
            "mean_rmse": float(np.mean([r["rmse"] for r in results])),
            "std_rmse": float(np.std([r["rmse"] for r in results])),
            "fold_results": results,
        }

        logger.info(f"Expanding window validation complete: {len(results)} folds")

        return summary

    def equipment_hold_out_validation(
        self, model, X: np.ndarray, y: np.ndarray, equipment_ids: np.ndarray
    ) -> Dict:
        """
        Leave-one-equipment-out validation

        Args:
            model: Model object
            X: Features
            y: Targets
            equipment_ids: Equipment identifiers

        Returns:
            Dictionary with results per equipment
        """
        logger.info("Starting leave-one-equipment-out validation")

        unique_equipment = np.unique(equipment_ids)
        results = []

        for eq_id in unique_equipment:
            logger.info(f"Testing on equipment: {eq_id}")

            # Train on all except this equipment
            train_mask = equipment_ids != eq_id
            test_mask = equipment_ids == eq_id

            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

            results.append(
                {
                    "equipment_id": str(eq_id),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "mae": float(mae),
                    "rmse": float(rmse),
                }
            )

            logger.info(f"Equipment {eq_id}: MAE={mae:.2f}, RMSE={rmse:.2f}")

        summary = {
            "n_equipment": len(unique_equipment),
            "mean_mae": float(np.mean([r["mae"] for r in results])),
            "std_mae": float(np.std([r["mae"] for r in results])),
            "mean_rmse": float(np.mean([r["rmse"] for r in results])),
            "std_rmse": float(np.std([r["rmse"] for r in results])),
            "equipment_results": results,
        }

        logger.info(
            f"Leave-one-out validation complete for {len(unique_equipment)} equipment"
        )

        return summary

    def temporal_degradation_analysis(
        self, y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray
    ) -> Dict:
        """
        Analyze model performance over time

        Args:
            y_true: True values
            y_pred: Predicted values
            timestamps: Timestamps or cycle numbers

        Returns:
            Performance metrics over time
        """
        logger.info("Analyzing temporal model performance")

        # Sort by time
        sorted_idx = np.argsort(timestamps)
        y_true = y_true[sorted_idx]
        y_pred = y_pred[sorted_idx]
        timestamps = timestamps[sorted_idx]

        # Split into time periods
        n_periods = 10
        period_size = len(timestamps) // n_periods

        period_metrics = []

        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = start_idx + period_size if i < n_periods - 1 else len(timestamps)

            period_y_true = y_true[start_idx:end_idx]
            period_y_pred = y_pred[start_idx:end_idx]

            mae = np.mean(np.abs(period_y_true - period_y_pred))
            rmse = np.sqrt(np.mean((period_y_true - period_y_pred) ** 2))

            period_metrics.append(
                {
                    "period": i + 1,
                    "start_time": float(timestamps[start_idx]),
                    "end_time": float(timestamps[end_idx - 1]),
                    "n_samples": end_idx - start_idx,
                    "mae": float(mae),
                    "rmse": float(rmse),
                }
            )

        # Check for performance degradation
        early_mae = np.mean([p["mae"] for p in period_metrics[:3]])
        late_mae = np.mean([p["mae"] for p in period_metrics[-3:]])
        degradation_pct = (
            ((late_mae - early_mae) / early_mae * 100) if early_mae > 0 else 0
        )

        results = {
            "period_metrics": period_metrics,
            "early_period_mae": float(early_mae),
            "late_period_mae": float(late_mae),
            "degradation_pct": float(degradation_pct),
            "requires_retraining": degradation_pct > 20,
        }

        logger.info(
            f"Temporal analysis: Early MAE={early_mae:.2f}, Late MAE={late_mae:.2f}, Degradation={degradation_pct:.1f}%"
        )

        return results

    def prediction_stability_analysis(
        self, predictions_over_time: List[np.ndarray], true_values: np.ndarray
    ) -> Dict:
        """
        Analyze stability of predictions over multiple runs

        Args:
            predictions_over_time: List of prediction arrays
            true_values: True values

        Returns:
            Stability metrics
        """
        logger.info("Analyzing prediction stability")

        # Stack predictions
        all_predictions = np.stack(predictions_over_time, axis=0)

        # Mean and std of predictions
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        # Coefficient of variation
        cv = std_predictions / (mean_predictions + 1e-8)

        # Metrics using mean predictions
        mae = np.mean(np.abs(true_values - mean_predictions))
        rmse = np.sqrt(np.mean((true_values - mean_predictions) ** 2))

        results = {
            "n_runs": len(predictions_over_time),
            "mean_mae": float(mae),
            "mean_rmse": float(rmse),
            "mean_prediction_std": float(np.mean(std_predictions)),
            "mean_cv": float(np.mean(cv)),
            "max_cv": float(np.max(cv)),
            "stable_predictions_pct": float(np.mean(cv < 0.1) * 100),
        }

        logger.info(
            f"Stability analysis: Mean CV={results['mean_cv']:.4f}, Stable={results['stable_predictions_pct']:.1f}%"
        )

        return results

    def generate_backtest_report(
        self,
        walk_forward_results: Dict,
        expanding_window_results: Dict,
        equipment_results: Dict,
    ) -> str:
        """
        Generate comprehensive backtest report

        Args:
            walk_forward_results: Walk-forward validation results
            expanding_window_results: Expanding window results
            equipment_results: Equipment hold-out results

        Returns:
            Formatted report string
        """
        report = "=" * 80 + "\n"
        report += "BACKTESTING REPORT\n"
        report += "=" * 80 + "\n\n"

        report += "1. Walk-Forward Validation\n"
        report += "-" * 80 + "\n"
        report += f"Number of folds: {walk_forward_results['n_folds']}\n"
        report += f"Mean MAE: {walk_forward_results['mean_mae']:.2f} (+/- {walk_forward_results['std_mae']:.2f})\n"
        report += f"Mean RMSE: {walk_forward_results['mean_rmse']:.2f} (+/- {walk_forward_results['std_rmse']:.2f})\n\n"

        report += "2. Expanding Window Validation\n"
        report += "-" * 80 + "\n"
        report += f"Number of folds: {expanding_window_results['n_folds']}\n"
        report += f"Mean MAE: {expanding_window_results['mean_mae']:.2f} (+/- {expanding_window_results['std_mae']:.2f})\n"
        report += f"Mean RMSE: {expanding_window_results['mean_rmse']:.2f} (+/- {expanding_window_results['std_rmse']:.2f})\n\n"

        report += "3. Leave-One-Equipment-Out Validation\n"
        report += "-" * 80 + "\n"
        report += f"Number of equipment: {equipment_results['n_equipment']}\n"
        report += f"Mean MAE: {equipment_results['mean_mae']:.2f} (+/- {equipment_results['std_mae']:.2f})\n"
        report += f"Mean RMSE: {equipment_results['mean_rmse']:.2f} (+/- {equipment_results['std_rmse']:.2f})\n\n"

        report += "=" * 80 + "\n"
        report += f"Report generated: {datetime.now().isoformat()}\n"
        report += "=" * 80 + "\n"

        return report
