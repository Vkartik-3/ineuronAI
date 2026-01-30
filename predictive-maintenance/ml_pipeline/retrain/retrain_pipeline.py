"""
Automated Retraining Pipeline

Orchestrates the complete retraining workflow:
1. Drift detection
2. Data validation and preparation
3. Model training
4. Model comparison
5. Deployment (if approved)
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import mlflow

from drift_detector import DriftDetector
from model_comparator import ModelComparator
from deployment_manager import DeploymentManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Orchestrates automated model retraining with drift detection
    and intelligent deployment.
    """

    def __init__(self, config_path: str = "config/retrain_config.yaml"):
        """
        Initialize retraining pipeline.

        Args:
            config_path: Path to retraining configuration
        """
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.pipeline_config = self.config.get("pipeline", {})
        self.auto_deploy = self.pipeline_config.get("auto_deploy", False)
        self.require_approval = self.pipeline_config.get("require_approval", True)

        # Initialize components
        self.drift_detector = DriftDetector(config_path)
        self.model_comparator = ModelComparator(config_path)
        self.deployment_manager = DeploymentManager(config_path)

        logger.info(
            f"RetrainingPipeline initialized. "
            f"Auto-deploy: {self.auto_deploy}, Require approval: {self.require_approval}"
        )

    def run_scheduled_check(self) -> Dict[str, Any]:
        """
        Run scheduled drift check and trigger retraining if needed.

        Returns:
            Dictionary with check results and actions taken
        """
        logger.info("Starting scheduled retraining check...")

        try:
            # Load reference and current data
            reference_data = self._load_reference_data()
            current_data = self._load_current_data()

            # Detect drift
            drift_report = self.drift_detector.detect_drift(
                reference_data, current_data
            )
            self.drift_detector.log_drift_report(drift_report)

            # Log results
            logger.info(
                f"Drift check complete. Drift detected: {drift_report.drift_detected}, "
                f"Type: {drift_report.drift_type}, Score: {drift_report.drift_score:.3f}"
            )

            # Trigger retraining if drift detected
            if drift_report.drift_detected:
                logger.info("Drift detected. Triggering retraining...")
                retrain_result = self.trigger_retraining(
                    reason="drift_detected", drift_report=drift_report
                )
                return {
                    "check_timestamp": datetime.now().isoformat(),
                    "drift_detected": True,
                    "drift_report": drift_report,
                    "retraining_triggered": True,
                    "retrain_result": retrain_result,
                }
            else:
                logger.info("No drift detected. Retraining not needed.")
                return {
                    "check_timestamp": datetime.now().isoformat(),
                    "drift_detected": False,
                    "drift_report": drift_report,
                    "retraining_triggered": False,
                }

        except Exception as e:
            logger.error(f"Scheduled check failed: {e}")
            return {
                "check_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed",
            }

    def trigger_retraining(
        self, reason: str = "manual", drift_report: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Trigger model retraining workflow.

        Args:
            reason: Reason for retraining ('drift_detected', 'scheduled', 'manual')
            drift_report: Optional drift report if triggered by drift

        Returns:
            Dictionary with retraining results
        """
        logger.info(f"Starting retraining pipeline. Reason: {reason}")

        try:
            # Step 1: Prepare training data
            logger.info("Step 1/5: Preparing training data...")
            train_data, val_data, test_data = self._prepare_training_data()

            # Step 2: Train new model
            logger.info("Step 2/5: Training new model...")
            new_model_uri = self._train_model(train_data, val_data)

            # Step 3: Compare with production model
            logger.info("Step 3/5: Comparing with production model...")
            comparison_report = self.model_comparator.compare_models(
                champion_model_uri="models:/predictive_maintenance_model/Production",
                challenger_model_uri=new_model_uri,
                test_data=test_data,
            )
            self.model_comparator.log_comparison_report(comparison_report)

            # Step 4: Decide on deployment
            logger.info("Step 4/5: Evaluating deployment decision...")
            should_deploy = self._should_deploy(comparison_report)

            # Step 5: Deploy if approved
            deployment_report = None
            if should_deploy and (self.auto_deploy or not self.require_approval):
                logger.info("Step 5/5: Deploying new model to production...")
                deployment_report = self.deployment_manager.promote_to_production(
                    model_uri=new_model_uri
                )
            else:
                logger.info(
                    "Step 5/5: Deployment skipped (approval required or model not better)"
                )

            # Log complete workflow
            self._log_retraining_workflow(
                reason=reason,
                drift_report=drift_report,
                comparison_report=comparison_report,
                deployment_report=deployment_report,
            )

            return {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "new_model_uri": new_model_uri,
                "comparison_report": comparison_report,
                "deployed": deployment_report is not None,
                "deployment_report": deployment_report,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "error": str(e),
                "status": "failed",
            }

    def _load_reference_data(self) -> pd.DataFrame:
        """
        Load reference data for drift detection.

        Uses data from model training time as baseline.
        """
        # This would load from TimescaleDB or feature store
        # For demo, return synthetic data
        logger.info("Loading reference data from feature store...")

        # In production, query from database:
        # SELECT * FROM sensor_features
        # WHERE timestamp BETWEEN training_start AND training_end

        # Placeholder - replace with actual database query
        return self._generate_synthetic_data(samples=5000, seed=42)

    def _load_current_data(self) -> pd.DataFrame:
        """
        Load current data for drift detection.

        Uses recent data from past N days.
        """
        logger.info("Loading current data from feature store...")

        # In production, query recent data:
        # SELECT * FROM sensor_features
        # WHERE timestamp >= NOW() - INTERVAL '7 days'

        # Placeholder - replace with actual database query
        return self._generate_synthetic_data(samples=2000, seed=123)

    def _prepare_training_data(self):
        """
        Prepare training, validation, and test datasets.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info("Preparing training datasets...")

        # Load data from feature store
        # In production, this queries TimescaleDB
        data = self._generate_synthetic_data(samples=10000, seed=456)

        # Split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(data))
        val_size = int(0.15 * len(data))

        train_data = data[:train_size]
        val_data = data[train_size : train_size + val_size]
        test_data = data[train_size + val_size :]

        logger.info(
            f"Data prepared. Train: {len(train_data)}, "
            f"Val: {len(val_data)}, Test: {len(test_data)}"
        )

        return train_data, val_data, test_data

    def _train_model(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> str:
        """
        Train new model using training pipeline.

        Args:
            train_data: Training dataset
            val_data: Validation dataset

        Returns:
            Model URI in MLflow
        """
        logger.info("Training new model...")

        # This would call the actual training pipeline from ml_pipeline/train/
        # For demo, we'll create a placeholder

        with mlflow.start_run(
            run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ):
            # Log parameters
            mlflow.log_param("retrain_trigger", "automated")
            mlflow.log_param("train_samples", len(train_data))
            mlflow.log_param("val_samples", len(val_data))

            # In production, train actual model here using:
            # from ml_pipeline.train.train_pipeline import train_model
            # model = train_model(train_data, val_data)

            # For demo, log metrics
            mlflow.log_metric("train_mae", 8.5)
            mlflow.log_metric("val_mae", 9.2)

            # Get run ID for model URI
            run_id = mlflow.active_run().info.run_id
            model_uri = f"runs:/{run_id}/model"

        logger.info(f"Model trained successfully. URI: {model_uri}")
        return model_uri

    def _should_deploy(self, comparison_report: Any) -> bool:
        """
        Determine if new model should be deployed.

        Args:
            comparison_report: ComparisonReport from model comparison

        Returns:
            True if model should be deployed
        """
        return comparison_report.should_promote

    def _generate_synthetic_data(self, samples: int, seed: int) -> pd.DataFrame:
        """Generate synthetic sensor data for demo"""
        np.random.seed(seed)

        return pd.DataFrame(
            {
                "equipment_id": np.random.choice(["EQ001", "EQ002", "EQ003"], samples),
                "vibration_rms": np.random.normal(0.5, 0.1, samples),
                "temperature": np.random.normal(75, 5, samples),
                "pressure": np.random.normal(100, 10, samples),
                "power": np.random.normal(500, 50, samples),
                "vibration_peak": np.random.normal(1.2, 0.3, samples),
                "rpm": np.random.normal(1800, 100, samples),
                "rul": np.random.exponential(100, samples),
            }
        )

    def _log_retraining_workflow(
        self,
        reason: str,
        drift_report: Optional[Any],
        comparison_report: Any,
        deployment_report: Optional[Any],
    ) -> None:
        """
        Log complete retraining workflow to MLflow.

        Args:
            reason: Retraining trigger reason
            drift_report: Drift detection report
            comparison_report: Model comparison report
            deployment_report: Deployment report
        """
        try:
            mlflow.set_experiment("retraining_pipeline")

            with mlflow.start_run(
                run_name=f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_param("trigger_reason", reason)
                mlflow.log_param("auto_deploy", self.auto_deploy)

                if drift_report:
                    mlflow.log_metric(
                        "drift_detected", int(drift_report.drift_detected)
                    )
                    mlflow.log_metric("drift_score", drift_report.drift_score)
                    mlflow.log_param("drift_type", drift_report.drift_type)

                if comparison_report:
                    mlflow.log_param("winner", comparison_report.winner)
                    mlflow.log_metric(
                        "improvement_pct", comparison_report.improvement_pct
                    )
                    mlflow.log_metric(
                        "should_promote", int(comparison_report.should_promote)
                    )

                if deployment_report:
                    mlflow.log_param(
                        "deployment_status", deployment_report.deployment_status
                    )
                    mlflow.log_param(
                        "deployed_version", deployment_report.model_version
                    )

                mlflow.set_tag("workflow_status", "completed")

        except Exception as e:
            logger.warning(f"Failed to log workflow: {e}")


def main():
    """Test retraining pipeline"""
    print("\n=== Retraining Pipeline Test ===")
    print("\nThis pipeline automates:")
    print("1. Drift Detection - Monitors for data/concept drift")
    print("2. Data Preparation - Loads and splits training data")
    print("3. Model Training - Trains new model with current data")
    print("4. Model Comparison - Evaluates against production model")
    print("5. Deployment - Promotes to production if improved")
    print("\nScheduled Check Flow:")
    print("  → Run drift detection on recent data")
    print("  → If drift detected, trigger retraining")
    print("  → Compare new model with production")
    print("  → Deploy if improvement exceeds threshold")
    print("\nUsage:")
    print("  pipeline = RetrainingPipeline()")
    print("  result = pipeline.run_scheduled_check()  # Run as cron job")
    print("  # Or manual trigger:")
    print("  result = pipeline.trigger_retraining(reason='manual')")


if __name__ == "__main__":
    main()
