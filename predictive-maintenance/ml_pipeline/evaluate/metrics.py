"""
Evaluation Metrics for Predictive Maintenance Models
Comprehensive metrics for RUL regression and health classification
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """
    Metrics for RUL regression evaluation
    """

    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate all regression metrics

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values

        Returns:
            Dictionary with all metrics
        """
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE (Mean Absolute Percentage Error)
        mask = y_true != 0
        mape = (
            np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            if np.any(mask)
            else 0
        )

        # Max error
        max_error = np.max(np.abs(y_true - y_pred))

        # Median absolute error
        median_ae = np.median(np.abs(y_true - y_pred))

        # Explained variance score
        numerator = np.var(y_true - y_pred)
        denominator = np.var(y_true)
        evs = 1 - (numerator / denominator) if denominator > 0 else 0

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
            "max_error": float(max_error),
            "median_absolute_error": float(median_ae),
            "explained_variance_score": float(evs),
        }

        logger.info(f"Regression metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

        return metrics

    @staticmethod
    def calculate_rul_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray, tolerance: int = 10
    ) -> float:
        """
        Calculate RUL prediction accuracy within tolerance

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            tolerance: Acceptable error in cycles

        Returns:
            Accuracy (percentage within tolerance)
        """
        within_tolerance = np.abs(y_true - y_pred) <= tolerance
        accuracy = np.mean(within_tolerance) * 100

        logger.info(f"RUL accuracy within {tolerance} cycles: {accuracy:.2f}%")

        return float(accuracy)

    @staticmethod
    def calculate_early_late_predictions(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict:
        """
        Analyze early vs late predictions

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values

        Returns:
            Dictionary with early/late prediction stats
        """
        errors = y_pred - y_true

        early_predictions = errors > 0  # Overestimated RUL
        late_predictions = errors < 0  # Underestimated RUL

        stats = {
            "early_predictions_pct": float(np.mean(early_predictions) * 100),
            "late_predictions_pct": float(np.mean(late_predictions) * 100),
            "mean_early_error": float(np.mean(errors[early_predictions]))
            if np.any(early_predictions)
            else 0,
            "mean_late_error": float(np.mean(errors[late_predictions]))
            if np.any(late_predictions)
            else 0,
        }

        logger.info(
            f"Early predictions: {stats['early_predictions_pct']:.1f}%, Late: {stats['late_predictions_pct']:.1f}%"
        )

        return stats

    @staticmethod
    def calculate_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate NASA PHM08 Challenge scoring function.

        This is the official scoring metric from the PHM08 Prognostics Challenge.
        It asymmetrically penalizes late predictions more than early predictions:
        - Early predictions (y_pred > y_true): Less penalty
        - Late predictions (y_pred < y_true): More penalty

        Reference: A. Saxena, K. Goebel, "Damage Propagation Modeling for
                   Aircraft Engine Run-to-Failure Simulation", PHM08, 2008.

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values

        Returns:
            NASA score (lower is better, 0 is perfect)
        """
        diff = y_pred - y_true

        # Asymmetric exponential penalty
        score = np.where(
            diff < 0,
            np.exp(-diff / 13.0) - 1,  # Late prediction (underestimated RUL)
            np.exp(diff / 10.0) - 1,  # Early prediction (overestimated RUL)
        )

        total_score = np.sum(score)

        logger.info(f"NASA PHM08 Score: {total_score:.2f} (lower is better)")

        return float(total_score)

    @staticmethod
    def calculate_nasa_score_per_engine(
        y_true: np.ndarray, y_pred: np.ndarray, unit_ids: np.ndarray
    ) -> Dict:
        """
        Calculate NASA score per engine unit.

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            unit_ids: Engine unit identifiers

        Returns:
            Dictionary with per-engine scores and statistics
        """
        scores_per_engine = {}

        for unit_id in np.unique(unit_ids):
            mask = unit_ids == unit_id
            unit_true = y_true[mask]
            unit_pred = y_pred[mask]

            # Take only the last prediction for each engine (most critical)
            if len(unit_true) > 0:
                last_idx = np.argmax(
                    unit_true.index
                    if hasattr(unit_true, "index")
                    else np.arange(len(unit_true))
                )
                diff = unit_pred[last_idx] - unit_true[last_idx]

                score = (
                    np.exp(-diff / 13.0) - 1 if diff < 0 else np.exp(diff / 10.0) - 1
                )
                scores_per_engine[int(unit_id)] = float(score)

        return {
            "per_engine_scores": scores_per_engine,
            "mean_score": float(np.mean(list(scores_per_engine.values()))),
            "median_score": float(np.median(list(scores_per_engine.values()))),
            "worst_score": float(np.max(list(scores_per_engine.values()))),
            "best_score": float(np.min(list(scores_per_engine.values()))),
            "total_engines": len(scores_per_engine),
        }


class ClassificationMetrics:
    """
    Metrics for health status classification
    """

    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
    ) -> Dict:
        """
        Calculate all classification metrics

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_proba: Prediction probabilities (for ROC-AUC)
            class_names: Class names

        Returns:
            Dictionary with all metrics
        """
        if class_names is None:
            class_names = ["healthy", "warning", "critical", "imminent_failure"]

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # ROC-AUC
        roc_auc = None
        if y_proba is not None:
            try:
                roc_auc = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="weighted"
                )
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                roc_auc = 0.0

        # Classification report
        report = classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "precision_per_class": precision_per_class.tolist(),
            "recall_per_class": recall_per_class.tolist(),
            "f1_per_class": f1_per_class.tolist(),
            "confusion_matrix": cm.tolist(),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "classification_report": report,
        }

        logger.info(f"Classification metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}")

        return metrics

    @staticmethod
    def calculate_critical_class_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        critical_classes: list = [2, 3],  # critical, imminent_failure
    ) -> Dict:
        """
        Calculate metrics specifically for critical classes

        Args:
            y_true: True labels
            y_pred: Predicted labels
            critical_classes: List of critical class indices

        Returns:
            Dictionary with critical class metrics
        """
        # Binary problem: critical vs non-critical
        y_true_binary = np.isin(y_true, critical_classes).astype(int)
        y_pred_binary = np.isin(y_pred, critical_classes).astype(int)

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # False negatives (missed critical failures)
        false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

        metrics = {
            "critical_precision": float(precision),
            "critical_recall": float(recall),
            "critical_f1": float(f1),
            "missed_critical_failures": int(false_negatives),
        }

        logger.info(
            f"Critical class metrics: Precision={precision:.4f}, Recall={recall:.4f}"
        )

        return metrics


class CustomPredictiveMaintenanceMetrics:
    """
    Custom metrics specific to predictive maintenance
    """

    @staticmethod
    def calculate_early_warning_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        warning_threshold: int = 50,
        critical_threshold: int = 30,
    ) -> Dict:
        """
        Calculate early warning detection performance

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            warning_threshold: RUL threshold for warning
            critical_threshold: RUL threshold for critical

        Returns:
            Dictionary with early warning metrics
        """
        # True states
        true_warning = y_true <= warning_threshold
        true_critical = y_true <= critical_threshold

        # Predicted states
        pred_warning = y_pred <= warning_threshold
        pred_critical = y_pred <= critical_threshold

        # Warning detection
        warning_tp = np.sum(true_warning & pred_warning)
        warning_fp = np.sum(~true_warning & pred_warning)
        warning_fn = np.sum(true_warning & ~pred_warning)
        warning_precision = (
            warning_tp / (warning_tp + warning_fp)
            if (warning_tp + warning_fp) > 0
            else 0
        )
        warning_recall = (
            warning_tp / (warning_tp + warning_fn)
            if (warning_tp + warning_fn) > 0
            else 0
        )

        # Critical detection
        critical_tp = np.sum(true_critical & pred_critical)
        critical_fp = np.sum(~true_critical & pred_critical)
        critical_fn = np.sum(true_critical & ~pred_critical)
        critical_precision = (
            critical_tp / (critical_tp + critical_fp)
            if (critical_tp + critical_fp) > 0
            else 0
        )
        critical_recall = (
            critical_tp / (critical_tp + critical_fn)
            if (critical_tp + critical_fn) > 0
            else 0
        )

        metrics = {
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
            "warning_precision": float(warning_precision),
            "warning_recall": float(warning_recall),
            "warning_f1": float(
                2
                * warning_precision
                * warning_recall
                / (warning_precision + warning_recall)
            )
            if (warning_precision + warning_recall) > 0
            else 0,
            "critical_precision": float(critical_precision),
            "critical_recall": float(critical_recall),
            "critical_f1": float(
                2
                * critical_precision
                * critical_recall
                / (critical_precision + critical_recall)
            )
            if (critical_precision + critical_recall) > 0
            else 0,
            "missed_critical_warnings": int(critical_fn),
        }

        logger.info(
            f"Early warning: Critical F1={metrics['critical_f1']:.4f}, Missed={metrics['missed_critical_warnings']}"
        )

        return metrics

    @staticmethod
    def calculate_maintenance_cost_savings(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_per_failure: float = 10000,
        cost_per_maintenance: float = 1000,
        rul_threshold: int = 30,
    ) -> Dict:
        """
        Estimate cost savings from predictive maintenance

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            cost_per_failure: Cost of unplanned failure
            cost_per_maintenance: Cost of planned maintenance
            rul_threshold: RUL threshold for maintenance

        Returns:
            Dictionary with cost analysis
        """
        # Failures that would have occurred (true RUL <= threshold)
        true_failures = y_true <= rul_threshold

        # Correctly predicted failures (both true and pred <= threshold)
        pred_maintenance = y_pred <= rul_threshold
        correctly_predicted = true_failures & pred_maintenance

        # Costs
        prevented_failures = np.sum(correctly_predicted)
        false_alarms = np.sum(~true_failures & pred_maintenance)
        missed_failures = np.sum(true_failures & ~pred_maintenance)

        savings = prevented_failures * (cost_per_failure - cost_per_maintenance)
        false_alarm_cost = false_alarms * cost_per_maintenance
        missed_failure_cost = missed_failures * cost_per_failure

        net_savings = savings - false_alarm_cost - missed_failure_cost

        metrics = {
            "prevented_failures": int(prevented_failures),
            "false_alarms": int(false_alarms),
            "missed_failures": int(missed_failures),
            "gross_savings": float(savings),
            "false_alarm_cost": float(false_alarm_cost),
            "missed_failure_cost": float(missed_failure_cost),
            "net_savings": float(net_savings),
            "cost_per_failure": float(cost_per_failure),
            "cost_per_maintenance": float(cost_per_maintenance),
        }

        logger.info(
            f"Cost analysis: Net savings=${net_savings:,.2f}, Prevented={prevented_failures}, Missed={missed_failures}"
        )

        return metrics

    @staticmethod
    def calculate_lead_time_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, min_lead_time: int = 10
    ) -> Dict:
        """
        Calculate metrics related to prediction lead time

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            min_lead_time: Minimum required lead time for maintenance

        Returns:
            Dictionary with lead time metrics
        """
        # Predictions with sufficient lead time
        sufficient_lead_time = y_pred >= min_lead_time

        # Among true failures (RUL near 0)
        true_near_failure = y_true <= min_lead_time

        # Sufficient lead time for near failures
        timely_predictions = np.sum(true_near_failure & sufficient_lead_time)
        late_predictions = np.sum(true_near_failure & ~sufficient_lead_time)

        metrics = {
            "min_lead_time": min_lead_time,
            "timely_predictions": int(timely_predictions),
            "late_predictions": int(late_predictions),
            "lead_time_sufficiency_rate": float(timely_predictions / len(y_true))
            if len(y_true) > 0
            else 0,
            "mean_predicted_lead_time": float(np.mean(y_pred[true_near_failure]))
            if np.any(true_near_failure)
            else 0,
        }

        logger.info(f"Lead time: Timely={timely_predictions}, Late={late_predictions}")

        return metrics


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """

    def __init__(self):
        self.regression_metrics = RegressionMetrics()
        self.classification_metrics = ClassificationMetrics()
        self.custom_metrics = CustomPredictiveMaintenanceMetrics()

        logger.info("Model Evaluator initialized")

    def evaluate_rul_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tolerance: int = 10,
        warning_threshold: int = 50,
        critical_threshold: int = 30,
    ) -> Dict:
        """
        Complete evaluation for RUL prediction model

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            tolerance: Accuracy tolerance
            warning_threshold: Warning RUL threshold
            critical_threshold: Critical RUL threshold

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Evaluating RUL prediction model")

        results = {
            "regression_metrics": self.regression_metrics.calculate_all_metrics(
                y_true, y_pred
            ),
            "rul_accuracy": self.regression_metrics.calculate_rul_accuracy(
                y_true, y_pred, tolerance
            ),
            "early_late_analysis": self.regression_metrics.calculate_early_late_predictions(
                y_true, y_pred
            ),
            "early_warning_score": self.custom_metrics.calculate_early_warning_score(
                y_true, y_pred, warning_threshold, critical_threshold
            ),
            "cost_savings": self.custom_metrics.calculate_maintenance_cost_savings(
                y_true, y_pred
            ),
            "lead_time_metrics": self.custom_metrics.calculate_lead_time_metrics(
                y_true, y_pred
            ),
        }

        return results

    def evaluate_health_classifier(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
    ) -> Dict:
        """
        Complete evaluation for health status classifier

        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_proba: Prediction probabilities
            class_names: Class names

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Evaluating health classification model")

        results = {
            "classification_metrics": self.classification_metrics.calculate_all_metrics(
                y_true, y_pred, y_proba, class_names
            ),
            "critical_class_metrics": self.classification_metrics.calculate_critical_class_metrics(
                y_true, y_pred
            ),
        }

        return results
