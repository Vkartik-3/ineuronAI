"""
Inference Engine - Core prediction logic
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Core inference engine for RUL and health predictions.
    Handles preprocessing, prediction, and postprocessing.
    """

    def __init__(
        self,
        sequence_length: int = 50,
        n_features: int = 150,
        normalization: str = "standard",
    ):
        """
        Initialize inference engine

        Args:
            sequence_length: Length of input sequences for LSTM
            n_features: Number of features
            normalization: Normalization method ('standard', 'minmax', 'robust')
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.normalization = normalization
        self.scaler = StandardScaler()  # Will be loaded from saved scaler

        # Health status thresholds
        self.rul_thresholds = {
            "healthy": 100,
            "warning": 50,
            "critical": 30,
            "imminent": 10,
        }

        self.health_classes = ["healthy", "warning", "critical", "imminent_failure"]

    def preprocess_sequence(
        self, sequence: List[Dict[str, float]], equipment_id: str
    ) -> np.ndarray:
        """
        Preprocess input sequence for LSTM

        Args:
            sequence: List of feature dictionaries
            equipment_id: Equipment identifier

        Returns:
            Preprocessed numpy array (1, sequence_length, n_features)
        """
        try:
            # Convert to numpy array
            features = []
            for reading in sequence:
                # Extract feature values in consistent order
                feature_vector = [
                    reading.get(f"feature_{i}", 0.0) for i in range(self.n_features)
                ]
                features.append(feature_vector)

            features = np.array(features)

            # Handle sequence length
            if len(features) < self.sequence_length:
                # Pad with zeros at the beginning
                padding = np.zeros(
                    (self.sequence_length - len(features), self.n_features)
                )
                features = np.vstack([padding, features])
            elif len(features) > self.sequence_length:
                # Take last sequence_length readings
                features = features[-self.sequence_length :]

            # Normalize
            if self.normalization == "standard":
                features = self.scaler.transform(features)

            # Add batch dimension
            features = features.reshape(1, self.sequence_length, self.n_features)

            return features

        except Exception as e:
            logger.error(f"Preprocessing failed for {equipment_id}: {e}")
            raise

    def preprocess_features(
        self, features: Dict[str, float], equipment_id: str
    ) -> np.ndarray:
        """
        Preprocess features for Random Forest

        Args:
            features: Feature dictionary
            equipment_id: Equipment identifier

        Returns:
            Preprocessed numpy array (1, n_features)
        """
        try:
            # Extract features in consistent order
            feature_vector = [
                features.get(f"feature_{i}", 0.0) for i in range(self.n_features)
            ]
            feature_vector = np.array(feature_vector).reshape(1, -1)

            # Normalize
            if self.normalization == "standard":
                feature_vector = self.scaler.transform(feature_vector)

            return feature_vector

        except Exception as e:
            logger.error(f"Preprocessing failed for {equipment_id}: {e}")
            raise

    def predict_rul(
        self,
        model: tf.keras.Model,
        sequence: np.ndarray,
        return_confidence: bool = False,
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Predict RUL using LSTM model

        Args:
            model: Loaded LSTM model
            sequence: Preprocessed sequence (1, sequence_length, n_features)
            return_confidence: Whether to return confidence interval

        Returns:
            Tuple of (predicted_rul, confidence_interval)
        """
        try:
            # Make prediction
            prediction = model.predict(sequence, verbose=0)
            rul = float(prediction[0][0])

            # Clip to reasonable range
            rul = max(0, min(rul, 200))

            confidence_interval = None
            if return_confidence:
                # Monte Carlo dropout for uncertainty estimation
                predictions = []
                for _ in range(10):
                    pred = model.predict(sequence, verbose=0)
                    predictions.append(float(pred[0][0]))

                predictions = np.array(predictions)
                confidence_interval = {
                    "lower": float(np.percentile(predictions, 2.5)),
                    "upper": float(np.percentile(predictions, 97.5)),
                    "std": float(np.std(predictions)),
                }

            return rul, confidence_interval

        except Exception as e:
            logger.error(f"RUL prediction failed: {e}")
            raise

    def predict_health(
        self, model: Any, features: np.ndarray, return_probabilities: bool = False
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        Predict health status using Random Forest

        Args:
            model: Loaded Random Forest model
            features: Preprocessed features (1, n_features)
            return_probabilities: Whether to return class probabilities

        Returns:
            Tuple of (predicted_class, confidence, probabilities)
        """
        try:
            # Make prediction
            prediction = model.predict(features)
            predicted_class = self.health_classes[int(prediction[0])]

            # Get probabilities
            probabilities_array = model.predict_proba(features)[0]
            confidence = float(np.max(probabilities_array))

            probabilities = None
            if return_probabilities:
                probabilities = {
                    cls: float(prob)
                    for cls, prob in zip(self.health_classes, probabilities_array)
                }

            return predicted_class, confidence, probabilities

        except Exception as e:
            logger.error(f"Health prediction failed: {e}")
            raise

    def get_health_status_from_rul(self, rul: float) -> str:
        """
        Determine health status from RUL

        Args:
            rul: Predicted remaining useful life

        Returns:
            Health status string
        """
        if rul >= self.rul_thresholds["healthy"]:
            return "healthy"
        elif rul >= self.rul_thresholds["warning"]:
            return "warning"
        elif rul >= self.rul_thresholds["critical"]:
            return "critical"
        else:
            return "imminent_failure"

    def batch_predict_rul(
        self, model: tf.keras.Model, sequences: List[np.ndarray]
    ) -> List[float]:
        """
        Batch RUL prediction

        Args:
            model: Loaded LSTM model
            sequences: List of preprocessed sequences

        Returns:
            List of RUL predictions
        """
        try:
            # Stack sequences
            batch = np.vstack(sequences)

            # Batch prediction
            predictions = model.predict(batch, verbose=0)

            # Clip and convert
            ruls = [max(0, min(float(pred[0]), 200)) for pred in predictions]

            return ruls

        except Exception as e:
            logger.error(f"Batch RUL prediction failed: {e}")
            raise

    def validate_input(
        self, features: Dict[str, float], equipment_id: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input features

        Args:
            features: Feature dictionary
            equipment_id: Equipment identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check equipment_id
        if not equipment_id or not isinstance(equipment_id, str):
            return False, "Invalid equipment_id"

        # Check feature count
        if len(features) != self.n_features:
            return False, f"Expected {self.n_features} features, got {len(features)}"

        # Check for NaN/Inf
        for key, value in features.items():
            if not isinstance(value, (int, float)):
                return False, f"Feature {key} must be numeric"
            if np.isnan(value) or np.isinf(value):
                return False, f"Feature {key} contains NaN or Inf"

        return True, None
