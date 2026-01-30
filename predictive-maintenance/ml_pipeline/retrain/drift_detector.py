"""
Drift Detection Module

Monitors for data drift and concept drift to trigger model retraining.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow
from dataclasses import dataclass
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Container for drift detection results"""

    timestamp: datetime
    drift_detected: bool
    drift_type: str  # 'data', 'concept', 'both', 'none'
    drift_score: float
    affected_features: List[str]
    details: Dict[str, Any]
    recommendation: str


class DriftDetector:
    """
    Detects data drift and concept drift to trigger model retraining.

    Data Drift: Changes in input feature distributions
    Concept Drift: Changes in relationship between features and target
    """

    def __init__(self, config_path: str = "config/retrain_config.yaml"):
        """
        Initialize drift detector.

        Args:
            config_path: Path to retraining configuration
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.drift_config = self.config.get("drift_detection", {})
        self.data_drift_threshold = self.drift_config.get("data_drift_threshold", 0.05)
        self.concept_drift_threshold = self.drift_config.get(
            "concept_drift_threshold", 0.15
        )
        self.window_size = self.drift_config.get("window_size_days", 7)
        self.min_samples = self.drift_config.get("min_samples", 1000)

        logger.info(
            f"DriftDetector initialized. "
            f"Data threshold: {self.data_drift_threshold}, "
            f"Concept threshold: {self.concept_drift_threshold}"
        )

    def detect_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        target_col: str = "rul",
    ) -> DriftReport:
        """
        Detect both data and concept drift.

        Args:
            reference_data: Historical baseline data
            current_data: Recent data to compare
            target_col: Target column name

        Returns:
            DriftReport with detection results
        """
        if len(current_data) < self.min_samples:
            return DriftReport(
                timestamp=datetime.now(),
                drift_detected=False,
                drift_type="none",
                drift_score=0.0,
                affected_features=[],
                details={"reason": "insufficient_samples"},
                recommendation="Collect more data before drift detection",
            )

        # Detect data drift
        data_drift_results = self._detect_data_drift(reference_data, current_data)

        # Detect concept drift
        concept_drift_results = self._detect_concept_drift(
            reference_data, current_data, target_col
        )

        # Combine results
        data_drift_detected = data_drift_results["drift_detected"]
        concept_drift_detected = concept_drift_results["drift_detected"]

        if data_drift_detected and concept_drift_detected:
            drift_type = "both"
            drift_score = max(
                data_drift_results["drift_score"], concept_drift_results["drift_score"]
            )
            recommendation = "CRITICAL: Retrain immediately with expanded feature set"
        elif concept_drift_detected:
            drift_type = "concept"
            drift_score = concept_drift_results["drift_score"]
            recommendation = "HIGH: Retrain model with recent data"
        elif data_drift_detected:
            drift_type = "data"
            drift_score = data_drift_results["drift_score"]
            recommendation = "MEDIUM: Monitor closely, consider retraining"
        else:
            drift_type = "none"
            drift_score = 0.0
            recommendation = "Model is stable, no action needed"

        return DriftReport(
            timestamp=datetime.now(),
            drift_detected=(data_drift_detected or concept_drift_detected),
            drift_type=drift_type,
            drift_score=drift_score,
            affected_features=data_drift_results.get("affected_features", []),
            details={
                "data_drift": data_drift_results,
                "concept_drift": concept_drift_results,
            },
            recommendation=recommendation,
        )

    def _detect_data_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.

        Uses Kolmogorov-Smirnov test for continuous features.
        """
        numeric_cols = reference.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ["rul", "failure"]]

        drift_scores = {}
        affected_features = []

        for col in numeric_cols:
            if col not in current.columns:
                continue

            ref_values = reference[col].dropna()
            cur_values = current[col].dropna()

            if len(ref_values) < 30 or len(cur_values) < 30:
                continue

            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_values, cur_values)

            drift_scores[col] = {
                "ks_statistic": statistic,
                "p_value": p_value,
                "drifted": p_value < self.data_drift_threshold,
            }

            if p_value < self.data_drift_threshold:
                affected_features.append(col)

        # Calculate overall drift score
        if drift_scores:
            max_drift = max(1 - score["p_value"] for score in drift_scores.values())
        else:
            max_drift = 0.0

        drift_detected = len(affected_features) > 0

        return {
            "drift_detected": drift_detected,
            "drift_score": max_drift,
            "affected_features": affected_features,
            "feature_scores": drift_scores,
            "num_affected": len(affected_features),
            "total_features": len(drift_scores),
        }

    def _detect_concept_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame, target_col: str
    ) -> Dict[str, Any]:
        """
        Detect concept drift by comparing prediction errors.

        Loads production model and compares its performance on
        reference vs current data.
        """
        try:
            # Load production model from MLflow
            model_uri = "models:/predictive_maintenance_model/Production"
            model = mlflow.sklearn.load_model(model_uri)

            # Prepare features
            feature_cols = [
                c
                for c in reference.columns
                if c not in [target_col, "failure", "equipment_id", "timestamp"]
            ]

            # Predictions on reference data
            X_ref = reference[feature_cols]
            y_ref = reference[target_col]
            ref_preds = model.predict(X_ref)
            ref_mae = mean_absolute_error(y_ref, ref_preds)

            # Predictions on current data
            X_cur = current[feature_cols]
            y_cur = current[target_col]
            cur_preds = model.predict(X_cur)
            cur_mae = mean_absolute_error(y_cur, cur_preds)

            # Calculate drift score (relative error increase)
            if ref_mae > 0:
                error_increase = (cur_mae - ref_mae) / ref_mae
            else:
                error_increase = 0.0

            drift_detected = error_increase > self.concept_drift_threshold

            return {
                "drift_detected": drift_detected,
                "drift_score": error_increase,
                "reference_mae": float(ref_mae),
                "current_mae": float(cur_mae),
                "error_increase_pct": float(error_increase * 100),
                "threshold": self.concept_drift_threshold,
            }

        except Exception as e:
            logger.warning(f"Concept drift detection failed: {e}")
            return {"drift_detected": False, "drift_score": 0.0, "error": str(e)}

    def get_drift_history(self, days: int = 30) -> pd.DataFrame:
        """
        Retrieve drift detection history from MLflow.

        Args:
            days: Number of days of history to retrieve

        Returns:
            DataFrame with drift metrics over time
        """
        try:
            experiment = mlflow.get_experiment_by_name("drift_monitoring")
            if experiment is None:
                return pd.DataFrame()

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"attributes.start_time > {int((datetime.now() - timedelta(days=days)).timestamp() * 1000)}",
                order_by=["start_time DESC"],
            )

            return runs

        except Exception as e:
            logger.error(f"Failed to retrieve drift history: {e}")
            return pd.DataFrame()

    def log_drift_report(self, report: DriftReport) -> None:
        """
        Log drift report to MLflow for tracking.

        Args:
            report: DriftReport to log
        """
        try:
            mlflow.set_experiment("drift_monitoring")

            with mlflow.start_run(
                run_name=f"drift_check_{report.timestamp.strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_param("drift_type", report.drift_type)
                mlflow.log_metric("drift_score", report.drift_score)
                mlflow.log_metric("drift_detected", int(report.drift_detected))
                mlflow.log_metric(
                    "num_affected_features", len(report.affected_features)
                )

                if report.affected_features:
                    mlflow.log_param(
                        "affected_features", ", ".join(report.affected_features)
                    )

                mlflow.log_dict(report.details, "drift_details.json")
                mlflow.set_tag("recommendation", report.recommendation)

            logger.info(
                f"Drift report logged: {report.drift_type}, score={report.drift_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to log drift report: {e}")


def main():
    """Test drift detection"""
    # Create synthetic data
    np.random.seed(42)

    # Reference data (normal distribution)
    ref_data = pd.DataFrame(
        {
            "vibration_rms": np.random.normal(0.5, 0.1, 1000),
            "temperature": np.random.normal(75, 5, 1000),
            "pressure": np.random.normal(100, 10, 1000),
            "power": np.random.normal(500, 50, 1000),
            "rul": np.random.normal(100, 30, 1000),
        }
    )

    # Current data with drift (shifted distribution)
    cur_data = pd.DataFrame(
        {
            "vibration_rms": np.random.normal(0.7, 0.12, 1000),  # Shifted
            "temperature": np.random.normal(80, 5, 1000),  # Shifted
            "pressure": np.random.normal(100, 10, 1000),  # No drift
            "power": np.random.normal(500, 50, 1000),  # No drift
            "rul": np.random.normal(90, 30, 1000),
        }
    )

    detector = DriftDetector()
    report = detector.detect_drift(ref_data, cur_data)

    print("\n=== Drift Detection Report ===")
    print(f"Timestamp: {report.timestamp}")
    print(f"Drift Detected: {report.drift_detected}")
    print(f"Drift Type: {report.drift_type}")
    print(f"Drift Score: {report.drift_score:.3f}")
    print(f"Affected Features: {report.affected_features}")
    print(f"Recommendation: {report.recommendation}")


if __name__ == "__main__":
    main()
