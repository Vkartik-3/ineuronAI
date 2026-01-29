"""
Training Pipeline Orchestrator
Main pipeline for training LSTM and Random Forest models
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

from models.lstm_model import LSTMRULPredictor
from models.random_forest_model import RandomForestHealthClassifier
from tuning.hyperparameter_tuner import HyperparameterTuner
from validation.cross_validator import TimeSeriesCrossValidator
from tracking.mlflow_tracker import MLflowTracker

# Import feature store
from feature_store.pipeline import FeatureStorePipeline

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Main training pipeline orchestrator
    """

    def __init__(self, config_path: str):
        """
        Initialize training pipeline

        Args:
            config_path: Path to training configuration
        """
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.lstm_model = None
        self.rf_model = None
        self.tuner = HyperparameterTuner(self.config)
        self.cv = TimeSeriesCrossValidator(self.config)
        self.mlflow = MLflowTracker(self.config)

        # Feature store
        feature_config_path = self.config.get("data", {}).get("feature_store_config")
        if feature_config_path:
            self.feature_store = FeatureStorePipeline(feature_config_path)
        else:
            self.feature_store = None

        logger.info("Training Pipeline initialized")

    def load_data_from_feature_store(
        self,
        equipment_ids: list,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load data from feature store

        Args:
            equipment_ids: List of equipment IDs
            start_cycle: Start cycle
            end_cycle: End cycle

        Returns:
            DataFrame with features and labels
        """
        if self.feature_store is None:
            raise ValueError("Feature store not configured")

        logger.info(
            f"Loading data for {len(equipment_ids)} equipment from feature store"
        )

        df = self.feature_store.load_from_timescaledb(
            start_cycle=start_cycle, end_cycle=end_cycle, equipment_ids=equipment_ids
        )

        # Process if needed
        if df.empty or "rul" not in df.columns:
            df = self.feature_store.process_equipment_data(df)

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        return df

    def prepare_lstm_data(
        self,
        df: pd.DataFrame,
        sequence_length: Optional[int] = None,
        target_col: str = "rul",
    ) -> Dict:
        """
        Prepare data for LSTM training

        Args:
            df: DataFrame with features
            sequence_length: Sequence length
            target_col: Target column

        Returns:
            Dictionary with X, y, feature_names, equipment_ids
        """
        if sequence_length is None:
            sequence_length = self.config.get("data", {}).get("sequence_length", 50)

        logger.info(f"Preparing LSTM sequences (length={sequence_length})")

        # Get feature columns
        exclude_cols = self.config.get("data", {}).get("exclude_features", [])
        exclude_cols.extend(
            [
                "rul",
                "rul_normalized",
                "failure_imminent",
                "health_status",
                "health_status_code",
                "degradation_rate",
            ]
        )

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Generate sequences
        from feature_store.features.time_series_features import SequenceGenerator

        seq_gen = SequenceGenerator(sequence_length=sequence_length, stride=1)
        X, y, eq_ids = seq_gen.generate_sequences_per_equipment(
            df, feature_cols, target_col
        )

        logger.info(f"Generated sequences: X shape {X.shape}, y shape {y.shape}")

        return {"X": X, "y": y, "feature_names": feature_cols, "equipment_ids": eq_ids}

    def prepare_rf_data(
        self, df: pd.DataFrame, target_col: str = "health_status_code"
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        Prepare data for Random Forest training

        Args:
            df: DataFrame with features
            target_col: Target column

        Returns:
            X, y, feature_names
        """
        logger.info("Preparing Random Forest data")

        # Get feature columns
        exclude_cols = self.config.get("data", {}).get("exclude_features", [])
        exclude_cols.extend(
            [
                "rul",
                "rul_normalized",
                "failure_imminent",
                "health_status",
                "health_status_code",
                "degradation_rate",
            ]
        )

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]

        X = df[feature_cols].values
        y = df[target_col].values

        logger.info(f"RF data: X shape {X.shape}, y shape {y.shape}")

        return X, y, feature_cols

    def train_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        run_name: str = "lstm_rul_training",
    ) -> Dict:
        """
        Train LSTM model

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets
            X_test: Test sequences
            y_test: Test targets
            run_name: MLflow run name

        Returns:
            Training results dictionary
        """
        logger.info("Starting LSTM training")

        # Start MLflow run
        self.mlflow.start_run(run_name=run_name)

        try:
            # Log config
            self.mlflow.log_config(self.config)
            self.mlflow.set_tags(
                {
                    "model_type": "LSTM",
                    "task": "RUL_prediction",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Build and train model
            self.lstm_model = LSTMRULPredictor(self.config)
            input_shape = (X_train.shape[1], X_train.shape[2])
            self.lstm_model.build_model(input_shape)

            # Log model architecture
            self.mlflow.log_param("model_architecture", "LSTM")
            self.mlflow.log_param("sequence_length", X_train.shape[1])
            self.mlflow.log_param("n_features", X_train.shape[2])
            self.mlflow.log_param("n_train_samples", len(X_train))

            # Train
            checkpoint_path = os.path.join(
                "./checkpoints",
                f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras",
            )
            os.makedirs("./checkpoints", exist_ok=True)

            history = self.lstm_model.train(
                X_train, y_train, X_val, y_val, checkpoint_path
            )

            # Log training history
            self.mlflow.log_training_history(history)

            # Evaluate on test set
            test_metrics = {}
            if X_test is not None and y_test is not None:
                test_metrics = self.lstm_model.evaluate(X_test, y_test)
                self.mlflow.log_metrics(
                    {f"test_{k}": v for k, v in test_metrics.items()}
                )

                # Log predictions plot
                y_pred = self.lstm_model.predict(X_test)
                self.mlflow.log_rul_predictions(y_test, y_pred)

            # Save model
            model_path = os.path.join(
                "./models", f"lstm_rul_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
            )
            os.makedirs("./models", exist_ok=True)
            self.lstm_model.save_model(model_path)

            # Log model to MLflow
            self.mlflow.log_model_tensorflow(
                self.lstm_model.model,
                artifact_path="model",
                registered_model_name="lstm_rul_predictor",
            )

            logger.info("LSTM training completed")

            results = {
                "model": self.lstm_model,
                "history": history,
                "test_metrics": test_metrics,
                "model_path": model_path,
            }

            return results

        finally:
            self.mlflow.end_run()

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        run_name: str = "rf_health_classification",
    ) -> Dict:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names
            run_name: MLflow run name

        Returns:
            Training results dictionary
        """
        logger.info("Starting Random Forest training")

        # Start MLflow run
        self.mlflow.start_run(run_name=run_name)

        try:
            # Log config
            self.mlflow.set_tags(
                {
                    "model_type": "RandomForest",
                    "task": "health_classification",
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Build and train model
            self.rf_model = RandomForestHealthClassifier(self.config)
            self.rf_model.build_model()

            # Log model params
            model_info = self.rf_model.get_model_info()
            self.mlflow.log_params(
                {f"rf_{k}": v for k, v in model_info.items() if v is not None}
            )
            self.mlflow.log_param("n_train_samples", len(X_train))

            # Train
            self.rf_model.train(X_train, y_train, X_val, y_val, feature_names)

            # Evaluate on test set
            test_metrics = {}
            if X_test is not None and y_test is not None:
                test_metrics = self.rf_model.evaluate(X_test, y_test)
                self.mlflow.log_metrics(
                    {
                        f"test_{k}": v
                        for k, v in test_metrics.items()
                        if not isinstance(v, (list, str))
                    }
                )

                # Log confusion matrix
                if "confusion_matrix" in test_metrics:
                    self.mlflow.log_confusion_matrix(
                        test_metrics["confusion_matrix"],
                        ["healthy", "warning", "critical", "imminent_failure"],
                    )

            # Log feature importance
            if feature_names:
                importance_df = self.rf_model.get_feature_importance()
                self.mlflow.log_feature_importance(importance_df)

            # Save model
            model_path = os.path.join(
                "./models",
                f"rf_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
            )
            os.makedirs("./models", exist_ok=True)
            self.rf_model.save_model(model_path)

            # Log model to MLflow
            self.mlflow.log_model_sklearn(
                self.rf_model.model,
                artifact_path="model",
                registered_model_name="rf_health_classifier",
            )

            logger.info("Random Forest training completed")

            results = {
                "model": self.rf_model,
                "test_metrics": test_metrics,
                "model_path": model_path,
            }

            return results

        finally:
            self.mlflow.end_run()

    def run_full_pipeline(
        self,
        equipment_ids: list,
        train_lstm: bool = True,
        train_rf: bool = True,
        tune_hyperparams: bool = False,
    ) -> Dict:
        """
        Run complete training pipeline

        Args:
            equipment_ids: List of equipment IDs
            train_lstm: Whether to train LSTM
            train_rf: Whether to train Random Forest
            tune_hyperparams: Whether to tune hyperparameters

        Returns:
            Results dictionary
        """
        logger.info("Running full training pipeline")

        results = {}

        # Load data
        df = self.load_data_from_feature_store(equipment_ids)

        # Split data
        splits = self.feature_store.split_train_test(df)
        train_df = splits["train"]
        val_df = splits["val"]
        test_df = splits["test"]

        logger.info(
            f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

        # Train LSTM
        if train_lstm:
            lstm_data_train = self.prepare_lstm_data(train_df)
            lstm_data_val = self.prepare_lstm_data(val_df)
            lstm_data_test = self.prepare_lstm_data(test_df)

            lstm_results = self.train_lstm(
                lstm_data_train["X"],
                lstm_data_train["y"],
                lstm_data_val["X"],
                lstm_data_val["y"],
                lstm_data_test["X"],
                lstm_data_test["y"],
            )

            results["lstm"] = lstm_results

        # Train Random Forest
        if train_rf:
            X_train, y_train, feature_names = self.prepare_rf_data(train_df)
            X_val, y_val, _ = self.prepare_rf_data(val_df)
            X_test, y_test, _ = self.prepare_rf_data(test_df)

            rf_results = self.train_random_forest(
                X_train, y_train, X_val, y_val, X_test, y_test, feature_names
            )

            results["random_forest"] = rf_results

        logger.info("Full pipeline completed")

        return results
