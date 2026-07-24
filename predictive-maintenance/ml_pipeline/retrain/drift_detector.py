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

    Data Drift:    Statistical distribution shift in input features,
                   detected with KS tests + Benjamini-Hochberg FDR correction.
    Concept Drift: Increase in model prediction error on current vs reference data.

    **Important scope limitation — offline validation, not live monitoring.**

    ``_load_reference_data()`` and ``_load_current_data()`` in RetrainingPipeline
    both read from the NASA C-MAPSS dataset (training split vs test split).
    This is an *offline* sanity check that the model generalises from train to
    test, NOT a live production drift signal.

    To detect true live drift the reference data should be the historical
    production feature distribution and current data should be recent
    TimescaleDB readings.  Wiring that live source is an architecture change
    not yet implemented.

    TODO: Read reference/current distributions from TimescaleDB so drift
    detection operates on actual production sensor streams.
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

    # Minimum fraction of features that must show drift (after BH correction)
    # before the overall drift signal is raised.  Prevents a single spurious
    # KS p-value from firing retraining when 200 features are tested.
    _MIN_DRIFT_FRACTION = 0.05  # at least 5% of tested features must drift

    def _detect_data_drift(
        self, reference: pd.DataFrame, current: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect data drift using Kolmogorov-Smirnov tests with Benjamini-Hochberg
        false-discovery-rate correction.

        Without multiple-comparison correction, testing N features at α=0.05
        produces ~0.05·N false positives by chance.  For 200 features that is
        ~10 spurious drift signals per run.  BH correction controls the
        expected proportion of false discoveries at the configured threshold
        level, making the drift signal meaningful.

        Drift is raised only when **at least min_drift_fraction** of all
        tested features exceed the BH-corrected threshold, preventing a single
        noisy feature from triggering unnecessary retraining.
        """
        numeric_cols = list(reference.select_dtypes(include=[np.number]).columns)
        numeric_cols = [c for c in numeric_cols if c not in ["rul", "failure"]]

        # Collect raw (uncorrected) KS test results
        raw_results: Dict[str, Dict] = {}
        for col in numeric_cols:
            if col not in current.columns:
                continue
            ref_values = reference[col].dropna()
            cur_values = current[col].dropna()
            if len(ref_values) < 30 or len(cur_values) < 30:
                continue
            statistic, p_value = stats.ks_2samp(ref_values, cur_values)
            raw_results[col] = {"ks_statistic": float(statistic), "p_value": float(p_value)}

        if not raw_results:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "affected_features": [],
                "feature_scores": {},
                "num_affected": 0,
                "total_features": 0,
                "correction": "benjamini_hochberg",
            }

        # Benjamini-Hochberg FDR correction
        cols_tested = list(raw_results.keys())
        p_values = np.array([raw_results[c]["p_value"] for c in cols_tested])
        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        bh_threshold = (np.arange(1, n + 1) / n) * self.data_drift_threshold
        # Find the largest k such that p_(k) ≤ (k/n)·α; reject H1..Hk
        below = p_values[sorted_idx] <= bh_threshold
        if below.any():
            cutoff_rank = int(np.where(below)[0].max())
            bh_cutoff = bh_threshold[cutoff_rank]
        else:
            bh_cutoff = 0.0

        drift_scores: Dict[str, Dict] = {}
        affected_features: List[str] = []
        for col in cols_tested:
            p = raw_results[col]["p_value"]
            is_drifted = p <= bh_cutoff if bh_cutoff > 0 else False
            drift_scores[col] = {
                **raw_results[col],
                "bh_corrected": is_drifted,
                "drifted": is_drifted,
            }
            if is_drifted:
                affected_features.append(col)

        # Overall drift score: fraction of features that drifted (BH-corrected)
        drift_fraction = len(affected_features) / n if n > 0 else 0.0
        max_drift = float(1 - min(p_values)) if len(p_values) > 0 else 0.0

        # Require at least min_drift_fraction of features to drift before raising signal
        min_required = max(1, int(np.ceil(self._MIN_DRIFT_FRACTION * n)))
        drift_detected = len(affected_features) >= min_required

        logger.info(
            "Data drift check — %d/%d features drifted (BH-corrected, min_required=%d): "
            "drift_detected=%s | affected=%s",
            len(affected_features), n, min_required, drift_detected,
            affected_features[:5],
        )

        return {
            "drift_detected": drift_detected,
            "drift_score": max_drift,
            "affected_features": affected_features,
            "feature_scores": drift_scores,
            "num_affected": len(affected_features),
            "total_features": n,
            "correction": "benjamini_hochberg",
            "min_required_features": min_required,
            "drift_fraction": round(drift_fraction, 4),
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
            # Load production model from MLflow — pyfunc works for both PyTorch and sklearn
            model_uri = "models:/predictive_maintenance_model/Production"
            model = mlflow.pyfunc.load_model(model_uri)

            # Prepare features
            feature_cols = [
                c
                for c in reference.columns
                if c not in [target_col, "failure", "equipment_id", "timestamp"]
            ]

            # Predictions on reference data
            X_ref = reference[feature_cols]
            y_ref = reference[target_col].values
            ref_raw = model.predict(X_ref)
            ref_preds = ref_raw.values.flatten() if hasattr(ref_raw, "values") else np.array(ref_raw).flatten()
            ref_mae = mean_absolute_error(y_ref, ref_preds)

            # Predictions on current data
            X_cur = current[feature_cols]
            y_cur = current[target_col].values
            cur_raw = model.predict(X_cur)
            cur_preds = cur_raw.values.flatten() if hasattr(cur_raw, "values") else np.array(cur_raw).flatten()
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
