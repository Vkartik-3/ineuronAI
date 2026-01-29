"""
Random Forest Model for Health Status Classification
Classifies equipment health: healthy, warning, critical, imminent_failure
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import joblib
import logging

logger = logging.getLogger(__name__)


class RandomForestHealthClassifier:
    """
    Random Forest classifier for equipment health status
    Multi-class: 0=healthy, 1=warning, 2=critical, 3=imminent_failure
    """

    def __init__(self, config: Dict):
        """
        Initialize Random Forest classifier

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.rf_config = config.get("random_forest", {})
        self.model = None
        self.feature_importances_ = None
        self.feature_names_ = None

        logger.info("Random Forest Health Classifier initialized")

    def build_model(
        self, custom_params: Optional[Dict] = None
    ) -> RandomForestClassifier:
        """
        Build Random Forest model

        Args:
            custom_params: Optional custom parameters

        Returns:
            RandomForestClassifier instance
        """
        params = custom_params or self.rf_config.get("model_params", {})

        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 20),
            min_samples_split=params.get("min_samples_split", 5),
            min_samples_leaf=params.get("min_samples_leaf", 2),
            max_features=params.get("max_features", "sqrt"),
            bootstrap=params.get("bootstrap", True),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
            class_weight=params.get("class_weight", "balanced"),
            verbose=0,
        )

        logger.info(
            f"Random Forest model built with {params.get('n_estimators', 200)} trees"
        )

        return self.model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
    ):
        """
        Train Random Forest model

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
        """
        if self.model is None:
            self.build_model()

        logger.info(
            f"Training Random Forest. Train samples: {len(X_train)}, Classes: {np.unique(y_train)}"
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Store feature names and importances
        self.feature_names_ = feature_names
        self.feature_importances_ = self.model.feature_importances_

        # Training metrics
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        logger.info(f"Training accuracy: {train_acc:.4f}")

        # Validation metrics
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            logger.info(f"Validation accuracy: {val_acc:.4f}")

        logger.info("Training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict health status

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test data

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # ROC AUC (one-vs-rest for multi-class)
        try:
            roc_auc = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )
        except Exception:
            roc_auc = 0.0

        # Classification report
        class_report = classification_report(
            y_test,
            y_pred,
            target_names=["healthy", "warning", "critical", "imminent_failure"],
            zero_division=0,
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
        }

        logger.info(
            f"Evaluation metrics: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}"
        )

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if self.model is None or self.feature_importances_ is None:
            raise ValueError("Model not trained")

        # Create DataFrame
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names_
                or [f"feature_{i}" for i in range(len(self.feature_importances_))],
                "importance": self.feature_importances_,
            }
        )

        # Sort by importance
        importance_df = importance_df.sort_values("importance", ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "accuracy"
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Features
            y: Labels
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Dictionary with CV results
        """
        if self.model is None:
            self.build_model()

        logger.info(f"Performing {cv}-fold cross-validation")

        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        cv_results = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
            "scoring": scoring,
        }

        logger.info(
            f"CV {scoring}: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})"
        )

        return cv_results

    def save_model(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")

        # Save model and metadata
        model_data = {
            "model": self.model,
            "feature_names": self.feature_names_,
            "feature_importances": self.feature_importances_,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model from file"""
        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.feature_names_ = model_data.get("feature_names")
        self.feature_importances_ = model_data.get("feature_importances")

        logger.info(f"Model loaded from {filepath}")

    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {"status": "Model not trained"}

        info = {
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "min_samples_split": self.model.min_samples_split,
            "min_samples_leaf": self.model.min_samples_leaf,
            "max_features": self.model.max_features,
            "n_features": self.model.n_features_in_
            if hasattr(self.model, "n_features_in_")
            else None,
            "n_classes": self.model.n_classes_
            if hasattr(self.model, "n_classes_")
            else None,
        }

        return info
