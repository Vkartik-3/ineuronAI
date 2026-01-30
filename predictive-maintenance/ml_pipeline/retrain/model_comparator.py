"""
Model Comparator

Compares candidate model performance against production model
to determine if retraining improved results.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import mlflow
from dataclasses import dataclass
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonReport:
    """Container for model comparison results"""

    timestamp: datetime
    champion_model: str
    challenger_model: str
    winner: str
    improvement_pct: float
    metrics_comparison: Dict[str, Dict[str, float]]
    recommendation: str
    should_promote: bool
    details: Dict[str, Any]


class ModelComparator:
    """
    Compares candidate models against production model to determine
    if retraining improved performance.
    """

    def __init__(self, config_path: str = "config/retrain_config.yaml"):
        """
        Initialize model comparator.

        Args:
            config_path: Path to retraining configuration
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.comparison_config = self.config.get("model_comparison", {})
        self.min_improvement = self.comparison_config.get("min_improvement_pct", 5.0)
        self.primary_metric = self.comparison_config.get("primary_metric", "mae")
        self.test_size = self.comparison_config.get("test_size", 0.2)

        logger.info(
            f"ModelComparator initialized. "
            f"Primary metric: {self.primary_metric}, "
            f"Min improvement: {self.min_improvement}%"
        )

    def compare_models(
        self,
        champion_model_uri: str,
        challenger_model_uri: str,
        test_data: pd.DataFrame,
        target_col: str = "rul",
    ) -> ComparisonReport:
        """
        Compare two models on test data.

        Args:
            champion_model_uri: URI of current production model
            challenger_model_uri: URI of newly trained model
            test_data: Test dataset for evaluation
            target_col: Target column name

        Returns:
            ComparisonReport with comparison results
        """
        # Load models
        try:
            champion_model = mlflow.sklearn.load_model(champion_model_uri)
            challenger_model = mlflow.sklearn.load_model(challenger_model_uri)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return self._create_error_report(str(e))

        # Prepare data
        feature_cols = [
            c
            for c in test_data.columns
            if c not in [target_col, "failure", "equipment_id", "timestamp"]
        ]
        X_test = test_data[feature_cols]
        y_test = test_data[target_col]

        # Get predictions
        champion_preds = champion_model.predict(X_test)
        challenger_preds = challenger_model.predict(X_test)

        # Calculate metrics
        champion_metrics = self._calculate_metrics(y_test, champion_preds)
        challenger_metrics = self._calculate_metrics(y_test, challenger_preds)

        # Determine winner
        primary_metric_value = self.primary_metric.lower()
        champion_score = champion_metrics[primary_metric_value]
        challenger_score = challenger_metrics[primary_metric_value]

        # Lower is better for error metrics
        if primary_metric_value in ["mae", "rmse", "mape"]:
            improvement_pct = (
                (champion_score - challenger_score) / champion_score
            ) * 100
            winner = "challenger" if challenger_score < champion_score else "champion"
        else:  # Higher is better (r2)
            improvement_pct = (
                (challenger_score - champion_score) / abs(champion_score)
            ) * 100
            winner = "challenger" if challenger_score > champion_score else "champion"

        # Recommendation
        should_promote = (
            winner == "challenger" and improvement_pct >= self.min_improvement
        )

        if should_promote:
            recommendation = (
                f"PROMOTE: Challenger model shows {improvement_pct:.1f}% improvement"
            )
        elif winner == "challenger":
            recommendation = f"HOLD: Improvement ({improvement_pct:.1f}%) below threshold ({self.min_improvement}%)"
        else:
            recommendation = "REJECT: Challenger model performs worse than champion"

        # Statistical significance test
        stat_test_results = self._statistical_significance_test(
            y_test, champion_preds, challenger_preds
        )

        return ComparisonReport(
            timestamp=datetime.now(),
            champion_model=champion_model_uri,
            challenger_model=challenger_model_uri,
            winner=winner,
            improvement_pct=improvement_pct,
            metrics_comparison={
                "champion": champion_metrics,
                "challenger": challenger_metrics,
            },
            recommendation=recommendation,
            should_promote=should_promote,
            details={
                "test_samples": len(y_test),
                "feature_count": len(feature_cols),
                "statistical_test": stat_test_results,
                "min_improvement_threshold": self.min_improvement,
            },
        )

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        }

    def _statistical_significance_test(
        self,
        y_true: np.ndarray,
        champion_preds: np.ndarray,
        challenger_preds: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Perform paired t-test on residuals to check statistical significance.
        """
        from scipy import stats

        champion_errors = np.abs(y_true - champion_preds)
        challenger_errors = np.abs(y_true - challenger_preds)

        # Paired t-test
        statistic, p_value = stats.ttest_rel(champion_errors, challenger_errors)

        return {
            "test": "paired_t_test",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "interpretation": (
                "Significant difference"
                if p_value < 0.05
                else "No significant difference"
            ),
        }

    def _create_error_report(self, error_msg: str) -> ComparisonReport:
        """Create error report when comparison fails"""
        return ComparisonReport(
            timestamp=datetime.now(),
            champion_model="unknown",
            challenger_model="unknown",
            winner="unknown",
            improvement_pct=0.0,
            metrics_comparison={},
            recommendation=f"ERROR: {error_msg}",
            should_promote=False,
            details={"error": error_msg},
        )

    def log_comparison_report(self, report: ComparisonReport) -> None:
        """
        Log comparison report to MLflow.

        Args:
            report: ComparisonReport to log
        """
        try:
            mlflow.set_experiment("model_comparison")

            with mlflow.start_run(
                run_name=f"comparison_{report.timestamp.strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_param("champion_model", report.champion_model)
                mlflow.log_param("challenger_model", report.challenger_model)
                mlflow.log_param("winner", report.winner)
                mlflow.log_metric("improvement_pct", report.improvement_pct)
                mlflow.log_metric("should_promote", int(report.should_promote))

                # Log metrics
                if "champion" in report.metrics_comparison:
                    for metric, value in report.metrics_comparison["champion"].items():
                        mlflow.log_metric(f"champion_{metric}", value)

                if "challenger" in report.metrics_comparison:
                    for metric, value in report.metrics_comparison[
                        "challenger"
                    ].items():
                        mlflow.log_metric(f"challenger_{metric}", value)

                mlflow.log_dict(report.details, "comparison_details.json")
                mlflow.set_tag("recommendation", report.recommendation)

            logger.info(
                f"Comparison report logged: winner={report.winner}, improvement={report.improvement_pct:.1f}%"
            )

        except Exception as e:
            logger.error(f"Failed to log comparison report: {e}")

    def get_comparison_history(self, days: int = 90) -> pd.DataFrame:
        """
        Retrieve model comparison history from MLflow.

        Args:
            days: Number of days of history to retrieve

        Returns:
            DataFrame with comparison results over time
        """
        try:
            from datetime import timedelta

            experiment = mlflow.get_experiment_by_name("model_comparison")
            if experiment is None:
                return pd.DataFrame()

            cutoff_time = int(
                (datetime.now() - timedelta(days=days)).timestamp() * 1000
            )

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"attributes.start_time > {cutoff_time}",
                order_by=["start_time DESC"],
            )

            return runs

        except Exception as e:
            logger.error(f"Failed to retrieve comparison history: {e}")
            return pd.DataFrame()


def main():
    """Test model comparison"""
    # Create synthetic test data
    np.random.seed(42)

    test_data = pd.DataFrame(
        {
            "vibration_rms": np.random.normal(0.5, 0.1, 500),
            "temperature": np.random.normal(75, 5, 500),
            "pressure": np.random.normal(100, 10, 500),
            "power": np.random.normal(500, 50, 500),
            "rul": np.random.normal(100, 30, 500),
        }
    )

    print("\n=== Model Comparison Test ===")
    print("Note: This is a demo. Real usage requires trained models in MLflow.")
    print("\nTypical comparison flow:")
    print("1. Load champion model from production")
    print("2. Load challenger model from latest training run")
    print("3. Evaluate both on holdout test set")
    print("4. Compare metrics and statistical significance")
    print("5. Promote if improvement exceeds threshold")


if __name__ == "__main__":
    main()
