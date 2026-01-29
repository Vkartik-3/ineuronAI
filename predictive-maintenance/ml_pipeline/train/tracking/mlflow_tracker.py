"""
MLflow Integration for Experiment Tracking
Logs parameters, metrics, models, and artifacts
"""

import mlflow
import mlflow.tensorflow
import mlflow.sklearn
from typing import Dict, Any, Optional
import logging
import os
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracking wrapper
    """

    def __init__(self, config: Dict):
        """
        Initialize MLflow tracker

        Args:
            config: Configuration dictionary
        """
        self.config = config
        mlflow_config = config.get("mlflow", {})

        # MLflow settings
        self.tracking_uri = mlflow_config.get("tracking_uri", "http://localhost:5000")
        self.experiment_name = mlflow_config.get(
            "experiment_name", "predictive_maintenance"
        )
        self.artifact_location = mlflow_config.get("artifact_location", "./mlruns")
        self.auto_log = mlflow_config.get("auto_log", True)

        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)

        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifact_location
            )
        except Exception:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            self.experiment_id = experiment.experiment_id if experiment else None

        logger.info(
            f"MLflow initialized. Experiment: {self.experiment_name}, ID: {self.experiment_id}"
        )

    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> str:
        """
        Start MLflow run

        Args:
            run_name: Optional run name
            nested: Whether this is a nested run

        Returns:
            Run ID
        """
        run = mlflow.start_run(
            experiment_id=self.experiment_id, run_name=run_name, nested=nested
        )

        logger.info(f"MLflow run started: {run.info.run_id}")

        return run.info.run_id

    def end_run(self):
        """End current MLflow run"""
        mlflow.end_run()
        logger.info("MLflow run ended")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters

        Args:
            params: Parameter dictionary
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_param(self, key: str, value: Any):
        """Log single parameter"""
        mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics

        Args:
            metrics: Metrics dictionary
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log single metric"""
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log artifact file

        Args:
            local_path: Path to local file
            artifact_path: Optional path within artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_dict(self, dictionary: Dict, filename: str):
        """
        Log dictionary as JSON artifact

        Args:
            dictionary: Dictionary to log
            filename: Artifact filename
        """
        mlflow.log_dict(dictionary, filename)

    def log_model_tensorflow(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log TensorFlow/Keras model

        Args:
            model: Keras model
            artifact_path: Path within artifact store
            registered_model_name: Name for model registry
        """
        mlflow.tensorflow.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Logged TensorFlow model: {artifact_path}")

    def log_model_sklearn(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ):
        """
        Log scikit-learn model

        Args:
            model: sklearn model
            artifact_path: Path within artifact store
            registered_model_name: Name for model registry
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Logged sklearn model: {artifact_path}")

    def log_training_history(self, history, prefix: str = ""):
        """
        Log Keras training history

        Args:
            history: Keras History object
            prefix: Metric name prefix
        """
        for epoch, values in enumerate(
            zip(*[history.history[k] for k in history.history.keys()])
        ):
            metrics = {}
            for key, value in zip(history.history.keys(), values):
                metric_name = f"{prefix}{key}" if prefix else key
                metrics[metric_name] = float(value)

            self.log_metrics(metrics, step=epoch)

        logger.info(f"Logged training history for {len(history.epoch)} epochs")

    def log_confusion_matrix(self, cm: list, labels: list):
        """
        Log confusion matrix

        Args:
            cm: Confusion matrix
            labels: Class labels
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")

        # Save and log
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        self.log_artifact(cm_path)
        os.remove(cm_path)

        logger.info("Logged confusion matrix")

    def log_feature_importance(self, importance_df):
        """
        Log feature importance

        Args:
            importance_df: DataFrame with feature importances
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        importance_df.head(20).plot(x="feature", y="importance", kind="barh")
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()

        # Save and log
        fi_path = "feature_importance.png"
        plt.savefig(fi_path)
        plt.close()

        self.log_artifact(fi_path)
        os.remove(fi_path)

        # Also log as CSV
        csv_path = "feature_importance.csv"
        importance_df.to_csv(csv_path, index=False)
        self.log_artifact(csv_path)
        os.remove(csv_path)

        logger.info("Logged feature importance")

    def log_rul_predictions(self, y_true, y_pred, equipment_ids=None):
        """
        Log RUL prediction plot

        Args:
            y_true: True RUL values
            y_pred: Predicted RUL values
            equipment_ids: Optional equipment identifiers
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        axes[0].set_xlabel("True RUL")
        axes[0].set_ylabel("Predicted RUL")
        axes[0].set_title("RUL Predictions vs Actual")
        axes[0].grid(True, alpha=0.3)

        # Residuals
        residuals = y_pred - y_true
        axes[1].scatter(y_true, residuals, alpha=0.5)
        axes[1].axhline(y=0, color="r", linestyle="--", lw=2)
        axes[1].set_xlabel("True RUL")
        axes[1].set_ylabel("Residuals")
        axes[1].set_title("Prediction Residuals")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save and log
        plot_path = "rul_predictions.png"
        plt.savefig(plot_path)
        plt.close()

        self.log_artifact(plot_path)
        os.remove(plot_path)

        logger.info("Logged RUL predictions plot")

    def log_config(self, config: Dict):
        """
        Log configuration as artifact

        Args:
            config: Configuration dictionary
        """
        config_path = "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        self.log_artifact(config_path)
        os.remove(config_path)

        logger.info("Logged configuration")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set run tags

        Args:
            tags: Tag dictionary
        """
        mlflow.set_tags(tags)
        logger.debug(f"Set {len(tags)} tags")

    def set_tag(self, key: str, value: str):
        """Set single tag"""
        mlflow.set_tag(key, value)
