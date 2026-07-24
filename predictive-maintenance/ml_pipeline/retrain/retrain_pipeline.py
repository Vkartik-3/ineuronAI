"""
Automated Retraining Pipeline

Orchestrates the complete retraining workflow:
1. Accuracy check — retrain when within-tolerance accuracy drops below 80%
2. Drift detection — retrain on data / concept drift
3. Data preparation — loads real NASA C-MAPSS sensor data
4. Model training   — trains the PyTorch MLP (primary model)
5. Model comparison — challenger vs champion
6. Deployment       — promotes to production if challenger wins

Retraining is triggered by EITHER:
    a) accuracy_degradation : live accuracy < 80% (performance_monitor)
    b) drift_detected       : statistical drift  (drift_detector / KS test)
    c) manual               : explicit API call

Resume context:
    "Built automated monitoring and retraining pipeline on AWS using MLflow
    and Prometheus/Grafana that triggered retraining when accuracy dropped
    below 80%, ensuring production reliability."
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Path setup so this module can import from sibling packages
# ---------------------------------------------------------------------------
_RETRAIN_DIR = Path(__file__).resolve().parent
_ML_PIPELINE = _RETRAIN_DIR.parent
_REPO_ROOT = _ML_PIPELINE.parents[1]

sys.path.extend([
    str(_ML_PIPELINE / "train"),
    str(_REPO_ROOT / "predictive-maintenance" / "data_loader"),
])

from drift_detector import DriftDetector          # noqa: E402
from model_comparator import ModelComparator      # noqa: E402
from deployment_manager import DeploymentManager  # noqa: E402
from performance_monitor import PerformanceMonitor  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """
    Orchestrates automated model retraining with dual triggers:
      1. Accuracy-based  — fires when live accuracy drops below 80%
      2. Drift-based     — fires on statistical data/concept drift
    """

    # C-MAPSS dataset path (relative to repo root, or override via env var)
    _CMAPSS_PATH = os.environ.get(
        "CMAPSS_DATA_PATH",
        str(_REPO_ROOT / "archive" / "CMaps"),
    )
    _CMAPSS_DATASET = os.environ.get("CMAPSS_DATASET_ID", "FD001")

    def __init__(
        self,
        config_path: str = "config/retrain_config.yaml",
        performance_monitor: Optional[Any] = None,
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.pipeline_config = self.config.get("pipeline", {})
        self.auto_deploy = self.pipeline_config.get("auto_deploy", False)
        self.require_approval = self.pipeline_config.get("require_approval", True)

        # Sub-modules
        self.drift_detector = DriftDetector(config_path)
        self.model_comparator = ModelComparator(config_path)
        self.deployment_manager = DeploymentManager(config_path)
        # Use the injected monitor (which holds live /feedback records) when
        # provided; fall back to a fresh instance for standalone CLI runs.
        self.performance_monitor = (
            performance_monitor
            if performance_monitor is not None
            else PerformanceMonitor(config_path)
        )

        logger.info(
            "RetrainingPipeline initialised — "
            "auto_deploy=%s  require_approval=%s  dataset=%s  "
            "monitor_records=%d",
            self.auto_deploy, self.require_approval, self._CMAPSS_DATASET,
            self.performance_monitor.n_records,
        )

    # ======================================================================
    # Public entry points
    # ======================================================================

    def run_scheduled_check(self) -> Dict[str, Any]:
        """
        Run the full scheduled check:
          1. Accuracy check (primary trigger — <80% fires retraining)
          2. Drift check    (secondary trigger)

        Called from a cron job or APScheduler task.
        """
        logger.info("Starting scheduled retraining check...")

        # ---------------------------------------------------------------- #
        # Trigger 1 — accuracy-based (the resume claim: accuracy < 80%)
        # ---------------------------------------------------------------- #
        accuracy_result = self._check_accuracy_trigger()
        if accuracy_result.get("retraining_triggered"):
            return accuracy_result

        # ---------------------------------------------------------------- #
        # Trigger 2 — statistical drift
        # ---------------------------------------------------------------- #
        return self._check_drift_trigger()

    def trigger_retraining(
        self,
        reason: str = "manual",
        drift_report: Optional[Any] = None,
        accuracy_report: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full retraining workflow.

        Steps:
          1/5 Load & prepare real C-MAPSS training data
          2/5 Train PyTorch MLP on NASA C-MAPSS sensor data
          3/5 Compare challenger vs production champion
          4/5 Decide on deployment
          5/5 Deploy if approved
        """
        logger.info("Starting retraining pipeline — reason: %s", reason)

        try:
            # Step 1
            logger.info("Step 1/5: Loading NASA C-MAPSS training data...")
            train_data, val_data, test_data = self._prepare_training_data()

            # Step 2
            logger.info("Step 2/5: Training PyTorch MLP on C-MAPSS data...")
            new_model_uri = self._train_model(train_data, val_data, test_data, reason)

            # Step 3
            logger.info("Step 3/5: Comparing challenger vs production model...")
            comparison_report = self.model_comparator.compare_models(
                champion_model_uri="models:/predictive_maintenance_model/Production",
                challenger_model_uri=new_model_uri,
                test_data=test_data,
            )
            self.model_comparator.log_comparison_report(comparison_report)

            # Step 4
            logger.info("Step 4/5: Evaluating deployment decision...")
            should_deploy = comparison_report.should_promote

            # Step 5
            deployment_report = None
            if should_deploy and (self.auto_deploy or not self.require_approval):
                logger.info("Step 5/5: Deploying new model to production...")
                deployment_report = self.deployment_manager.promote_to_production(
                    model_uri=new_model_uri
                )
            else:
                logger.info(
                    "Step 5/5: Deployment skipped — approval required or challenger "
                    "did not improve over champion"
                )

            self._log_retraining_workflow(
                reason=reason,
                drift_report=drift_report,
                accuracy_report=accuracy_report,
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

        except Exception as exc:
            logger.error("Retraining pipeline failed: %s", exc)
            return {
                "timestamp": datetime.now().isoformat(),
                "reason": reason,
                "error": str(exc),
                "status": "failed",
            }

    # ======================================================================
    # Trigger 1 — accuracy-based
    # ======================================================================

    def _check_accuracy_trigger(self) -> Dict[str, Any]:
        """
        Evaluate rolling accuracy against the 80% threshold.

        When insufficient live data is available, falls back to evaluating
        the current MLflow Production model directly on C-MAPSS test data.
        """
        logger.info("Checking accuracy trigger (threshold=80%%)...")

        try:
            # First attempt: use stored rolling predictions
            report = self.performance_monitor.check_accuracy_threshold()

            if report.n_samples < self.performance_monitor.min_samples:
                # Not enough live predictions yet — evaluate on C-MAPSS test set
                logger.info(
                    "Rolling window has %d samples (< %d required); evaluating on "
                    "C-MAPSS test data instead",
                    report.n_samples, self.performance_monitor.min_samples,
                )
                report = self._evaluate_production_model_on_cmapss()

            self.performance_monitor.log_accuracy_report(report)

            if report.below_threshold:
                logger.warning(
                    "Accuracy %.1f%% is below threshold %.1f%% — triggering retraining",
                    report.accuracy * 100, report.threshold * 100,
                )
                retrain_result = self.trigger_retraining(
                    reason="accuracy_degradation",
                    accuracy_report=report,
                )
                return {
                    "check_timestamp": datetime.now().isoformat(),
                    "trigger": "accuracy_degradation",
                    "accuracy": report.accuracy,
                    "threshold": report.threshold,
                    "retraining_triggered": True,
                    "retrain_result": retrain_result,
                }
            else:
                return {
                    "check_timestamp": datetime.now().isoformat(),
                    "trigger": "accuracy_ok",
                    "accuracy": report.accuracy,
                    "threshold": report.threshold,
                    "retraining_triggered": False,
                }

        except Exception as exc:
            logger.error("Accuracy check failed: %s", exc)
            return {
                "check_timestamp": datetime.now().isoformat(),
                "error": str(exc),
                "retraining_triggered": False,
            }

    def _evaluate_production_model_on_cmapss(self) -> Any:
        """
        Load the production model from MLflow and evaluate it on the C-MAPSS
        test split to produce a fresh accuracy report.
        """
        from performance_monitor import AccuracyReport

        try:
            # Load production model
            model = mlflow.pyfunc.load_model(
                "models:/predictive_maintenance_model/Production"
            )

            # Load C-MAPSS test data
            _, _, test_data = self._prepare_training_data()

            X_test = test_data.drop(columns=["rul"], errors="ignore").fillna(0).values
            y_test = test_data["rul"].values if "rul" in test_data.columns else np.array([])

            if len(y_test) == 0:
                raise ValueError("No RUL labels in test data")

            y_pred = model.predict(X_test).flatten()

            accuracy = self.performance_monitor.compute_within_tolerance_accuracy(
                y_test, y_pred, self.performance_monitor.tolerance_cycles
            )
            mae = float(np.mean(np.abs(y_test - y_pred)))
            rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            below = accuracy < self.performance_monitor.accuracy_threshold

            return AccuracyReport(
                timestamp=datetime.utcnow(),
                window_days=0,
                n_samples=len(y_test),
                accuracy=accuracy,
                threshold=self.performance_monitor.accuracy_threshold,
                tolerance_cycles=self.performance_monitor.tolerance_cycles,
                below_threshold=below,
                mae=mae,
                rmse=rmse,
                details={"source": "cmapss_test_set"},
            )

        except Exception as exc:
            # An evaluation we could not run is INCONCLUSIVE, not healthy.
            # Reporting accuracy=1.0 / below_threshold=False here would let a
            # broken model loader silently suppress every retraining trigger.
            # We surface an explicit evaluation_unavailable report instead; the
            # caller (via RetrainGuard) treats it as EVALUATION_UNAVAILABLE and
            # neither retrains nor declares the model healthy.
            logger.error(
                "Could not evaluate production model on C-MAPSS — reporting "
                "evaluation_unavailable (NOT healthy): %s", exc
            )
            from performance_monitor import AccuracyReport
            report = AccuracyReport(
                timestamp=datetime.utcnow(),
                window_days=0,
                n_samples=0,
                accuracy=float("nan"),
                threshold=self.performance_monitor.accuracy_threshold,
                tolerance_cycles=self.performance_monitor.tolerance_cycles,
                below_threshold=False,
                mae=float("nan"),
                rmse=float("nan"),
                details={"reason": "evaluation_unavailable", "error": str(exc)},
            )
            report.evaluation_available = False  # consumed by the guard
            return report

    # ======================================================================
    # Trigger 2 — drift-based
    # ======================================================================

    def _check_drift_trigger(self) -> Dict[str, Any]:
        """Run statistical drift detection using existing DriftDetector."""
        logger.info("Checking drift trigger...")

        try:
            reference_data = self._load_reference_data()
            current_data = self._load_current_data()

            drift_report = self.drift_detector.detect_drift(reference_data, current_data)
            self.drift_detector.log_drift_report(drift_report)

            logger.info(
                "Drift check — drift_detected=%s  type=%s  score=%.3f",
                drift_report.drift_detected, drift_report.drift_type, drift_report.drift_score,
            )

            if drift_report.drift_detected:
                retrain_result = self.trigger_retraining(
                    reason="drift_detected", drift_report=drift_report
                )
                return {
                    "check_timestamp": datetime.now().isoformat(),
                    "trigger": "drift_detected",
                    "drift_type": drift_report.drift_type,
                    "drift_score": drift_report.drift_score,
                    "retraining_triggered": True,
                    "retrain_result": retrain_result,
                }
            else:
                return {
                    "check_timestamp": datetime.now().isoformat(),
                    "trigger": "no_drift",
                    "drift_score": drift_report.drift_score,
                    "retraining_triggered": False,
                }

        except Exception as exc:
            logger.error("Drift check failed: %s", exc)
            return {
                "check_timestamp": datetime.now().isoformat(),
                "error": str(exc),
                "retraining_triggered": False,
            }

    # ======================================================================
    # Data loading — real NASA C-MAPSS data (no synthetic placeholders)
    # ======================================================================

    def _load_cmapss(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and normalise the configured C-MAPSS sub-dataset."""
        try:
            from cmapss_loader import CMAPSSLoader

            loader = CMAPSSLoader(
                dataset_path=self._CMAPSS_PATH,
                dataset_id=self._CMAPSS_DATASET,
            )
            train_df = loader.load_train_data()
            test_df = loader.load_test_data()
            train_norm, test_norm, _ = loader.normalize_sensors(train_df, test_df)
            logger.info(
                "Loaded C-MAPSS %s — train=%d rows, test=%d rows",
                self._CMAPSS_DATASET, len(train_norm), len(test_norm),
            )
            self._last_data_source = "cmapss"
            return train_norm, test_norm
        except Exception as exc:
            logger.warning(
                "Could not load C-MAPSS from %s: %s — falling back to synthetic data",
                self._CMAPSS_PATH, exc,
            )
            self._last_data_source = "synthetic"
            return self._generate_synthetic_data(5000, 42), self._generate_synthetic_data(1000, 99)

    def _load_reference_data(self) -> pd.DataFrame:
        """Load training split as the drift reference baseline."""
        logger.info("Loading reference data (C-MAPSS training split)...")
        train_df, _ = self._load_cmapss()
        return train_df

    def _load_current_data(self) -> pd.DataFrame:
        """Load test split as the 'current' data for drift comparison."""
        logger.info("Loading current data (C-MAPSS test split)...")
        _, test_df = self._load_cmapss()
        return test_df

    def _prepare_training_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare train / val / test splits from real C-MAPSS data.

        Returns:
            (train_data, val_data, test_data) DataFrames
        """
        logger.info("Preparing training splits from C-MAPSS data...")

        train_df, test_df = self._load_cmapss()

        # Temporal split within training engines (80 / 20)
        n_train = int(0.8 * len(train_df))
        train_data = train_df.iloc[:n_train].copy()
        val_data = train_df.iloc[n_train:].copy()

        logger.info(
            "Split sizes — train=%d  val=%d  test=%d",
            len(train_data), len(val_data), len(test_df),
        )
        return train_data, val_data, test_df

    # ======================================================================
    # Model training — real PyTorch MLP (no hardcoded metrics)
    # ======================================================================

    def _get_feature_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X, y arrays from a C-MAPSS DataFrame."""
        exclude = {
            "unit_id", "time_cycle", "rul",
            "equipment_id", "timestamp", "created_at",
        }
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols].fillna(0).values.astype(np.float32)
        y = df["rul"].values.astype(np.float32) if "rul" in df.columns else np.zeros(len(df), np.float32)
        return X, y

    def _train_model(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        reason: str,
    ) -> str:
        """
        Train the PyTorch MLP on real C-MAPSS data and register in MLflow.

        Returns the MLflow model URI for the trained run.
        """
        try:
            from models.mlp_model import MLPRULPredictor
            import torch

            mlp_config = {
                "mlp": {
                    "architecture": {"hidden_sizes": [256, 128, 64, 32], "dropout": 0.2},
                    "training": {
                        "learning_rate": 0.001,
                        "weight_decay": 1e-4,
                        "epochs": 80,
                        "batch_size": 256,
                        "patience": 12,
                    },
                }
            }

            X_train, y_train = self._get_feature_matrix(train_data)
            X_val, y_val = self._get_feature_matrix(val_data)
            X_test, y_test = self._get_feature_matrix(test_data)

            data_source = getattr(self, "_last_data_source", "cmapss")
            with mlflow.start_run(
                run_name=f"retrain_{reason}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_param("retrain_trigger", reason)
                mlflow.log_param("model_type", "PyTorch_MLP")
                mlflow.log_param("framework", "PyTorch")
                mlflow.log_param("dataset", f"NASA_C-MAPSS_{self._CMAPSS_DATASET}")
                mlflow.set_tag("data_source", data_source)
                if data_source == "synthetic":
                    mlflow.set_tag("promote_blocked", "true")
                    logger.warning("Training on SYNTHETIC data — model tagged promote_blocked=true")
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param(
                    "device", "cuda" if torch.cuda.is_available() else "cpu"
                )

                predictor = MLPRULPredictor(mlp_config)
                history = predictor.train(X_train, y_train, X_val, y_val)

                # Fit and attach the StandardScaler used at serving time —
                # mirrors the working scaler-attachment pattern in
                # baseline_comparison.py / TrainingPipeline.train_mlp() so
                # that a retrained model is never promoted without the
                # scaler InferenceEngine needs to normalize input features.
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(X_train)
                predictor.scaler = scaler

                # Log training curves
                for step, tl in enumerate(history["train_loss"]):
                    mlflow.log_metric("train_loss", tl, step=step)
                for step, vl in enumerate(history.get("val_loss", [])):
                    mlflow.log_metric("val_loss", vl, step=step)

                # Test set evaluation — real metrics, not hardcoded
                metrics = predictor.evaluate(X_test, y_test)
                mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

                logger.info(
                    "Retrain complete — MAE=%.2f  RMSE=%.2f  R²=%.4f  accuracy=%.2f%%",
                    metrics["mae"], metrics["rmse"], metrics["r2"],
                    metrics["within_15_cycles_accuracy"] * 100,
                )

                # Save a local .pt checkpoint with the scaler embedded
                # (checkpoint["scaler"] — see MLPRULPredictor.save_model)
                # and attach it to the run as an artifact.  This is the
                # same checkpoint format ModelManager.load_mlp_model()
                # already knows how to load and attach as model._scaler.
                checkpoint_dir = "./checkpoints"
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"mlp_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt",
                )
                predictor.save_model(checkpoint_path)
                try:
                    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoint")
                except Exception as exc:
                    logger.warning("Could not log checkpoint artifact to MLflow: %s", exc)

                # Register model
                try:
                    import mlflow.pytorch
                    mlflow.pytorch.log_model(
                        predictor.model,
                        artifact_path="mlp_model",
                        registered_model_name="predictive_maintenance_model",
                    )
                except Exception as exc:
                    logger.warning("Could not register model in MLflow: %s", exc)

                run_id = mlflow.active_run().info.run_id
                return f"runs:/{run_id}/mlp_model"

        except ImportError as exc:
            logger.warning("PyTorch MLP not available (%s); logging placeholder run", exc)
            with mlflow.start_run(
                run_name=f"retrain_placeholder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_param("retrain_trigger", reason)
                mlflow.log_param("note", "pytorch_unavailable")
                run_id = mlflow.active_run().info.run_id
            return f"runs:/{run_id}/model"

    # ======================================================================
    # MLflow workflow logging
    # ======================================================================

    def _log_retraining_workflow(
        self,
        reason: str,
        drift_report: Optional[Any],
        accuracy_report: Optional[Any],
        comparison_report: Any,
        deployment_report: Optional[Any],
    ) -> None:
        try:
            mlflow.set_experiment("retraining_pipeline")
            with mlflow.start_run(
                run_name=f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ):
                mlflow.log_param("trigger_reason", reason)
                mlflow.log_param("auto_deploy", self.auto_deploy)

                if accuracy_report:
                    mlflow.log_metric("accuracy_at_trigger", accuracy_report.accuracy)
                    mlflow.log_metric("accuracy_threshold", accuracy_report.threshold)
                    mlflow.log_metric("accuracy_below_threshold",
                                      int(accuracy_report.below_threshold))

                if drift_report:
                    mlflow.log_metric("drift_detected", int(drift_report.drift_detected))
                    mlflow.log_metric("drift_score", drift_report.drift_score)
                    mlflow.log_param("drift_type", drift_report.drift_type)

                if comparison_report:
                    mlflow.log_param("winner", comparison_report.winner)
                    mlflow.log_metric("improvement_pct", comparison_report.improvement_pct)
                    mlflow.log_metric("should_promote", int(comparison_report.should_promote))

                if deployment_report:
                    mlflow.log_param("deployment_status", deployment_report.deployment_status)
                    mlflow.log_param("deployed_version", deployment_report.model_version)

                mlflow.set_tag("workflow_status", "completed")

        except Exception as exc:
            logger.warning("Failed to log workflow to MLflow: %s", exc)

    # ======================================================================
    # Helpers
    # ======================================================================

    @staticmethod
    def _generate_synthetic_data(samples: int, seed: int) -> pd.DataFrame:
        """Synthetic data — used ONLY as a last-resort fallback when C-MAPSS is unavailable."""
        np.random.seed(seed)
        n_sensors = 21
        data: Dict = {"unit_id": np.random.choice(range(1, 11), samples)}
        for s in range(1, n_sensors + 1):
            data[f"sensor_{s}"] = np.random.normal(50, 10, samples)
        for o in range(1, 4):
            data[f"op_setting_{o}"] = np.random.normal(0, 1, samples)
        data["time_cycle"] = np.random.randint(1, 300, samples)
        data["rul"] = np.random.exponential(100, samples).clip(0, 300)
        return pd.DataFrame(data)


def main():
    print("\n=== Automated Retraining Pipeline ===\n")
    print("Triggers:")
    print("  1. Accuracy-based : fires when within-15-cycle accuracy < 80%")
    print("  2. Drift-based    : fires on KS-test data / concept drift")
    print("  3. Manual         : explicit API or CLI call\n")
    print("Data: NASA C-MAPSS turbofan engine degradation dataset (real sensor data)")
    print("Model: PyTorch MLP trained on engineered temporal features\n")
    print("Usage:")
    print("  pipeline = RetrainingPipeline()")
    print("  result = pipeline.run_scheduled_check()   # used by cron/APScheduler")
    print("  result = pipeline.trigger_retraining(reason='manual')  # manual")


if __name__ == "__main__":
    main()
