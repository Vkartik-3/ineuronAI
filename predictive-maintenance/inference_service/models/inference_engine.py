"""
Inference Engine — Core prediction logic.

Supports two model backends:
  1. **PyTorch MLP** (primary) — flat feature vector input, explicit GPU
     placement via torch.device('cuda'). Used for turbofan RUL prediction
     on NASA C-MAPSS temporal features.
  2. **TensorFlow LSTM** (legacy) — sequence input, kept for backwards compat.

GPU acceleration is applied automatically for the PyTorch path when a CUDA
device is available.
"""

from __future__ import annotations  # lazy annotations: `tf.keras.Model` in a
# signature must not be evaluated at import when tensorflow is not installed
# (the TF/LSTM backend is optional; the primary path is PyTorch).

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# PyTorch — primary inference backend
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    _TORCH_DEVICE = None
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# TensorFlow — legacy / LSTM backend
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
    _TF_AVAILABLE = True
except ImportError:
    tf = None  # type: ignore
    _TF_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.info(
    "InferenceEngine backends — PyTorch: %s (device=%s), TensorFlow: %s",
    _TORCH_AVAILABLE, _TORCH_DEVICE, _TF_AVAILABLE,
)


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

    def _scaler_is_fitted(self) -> bool:
        return getattr(self.scaler, "mean_", None) is not None

    def _warn_scaler_unfitted(self, equipment_id: str) -> None:
        if not getattr(self, "_scaler_unfitted_warned", False):
            logger.warning(
                "normalization='standard' requested but no fitted scaler is "
                "loaded — serving %s on unnormalized features. Load a checkpoint "
                "with an embedded scaler.", equipment_id,
            )
            self._scaler_unfitted_warned = True

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

            # Normalize only with a FITTED scaler. A freshly constructed
            # StandardScaler has no statistics, and calling transform() on it
            # raises NotFittedError — crashing the request. The scaler is
            # populated when a checkpoint with an embedded scaler is loaded; if
            # none has been loaded we skip normalization and warn once rather
            # than fail. (Canonical MLP inference does not use this LSTM/RF
            # helper; it goes through OnlineFeatureEngineer + the canonical
            # checkpoint's own transform().)
            if self.normalization == "standard" and self._scaler_is_fitted():
                features = self.scaler.transform(features)
            elif self.normalization == "standard":
                self._warn_scaler_unfitted(equipment_id)

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

            # Normalize only with a fitted scaler (see preprocess_sequence).
            if self.normalization == "standard" and self._scaler_is_fitted():
                feature_vector = self.scaler.transform(feature_vector)
            elif self.normalization == "standard":
                self._warn_scaler_unfitted(equipment_id)

            return feature_vector

        except Exception as e:
            logger.error(f"Preprocessing failed for {equipment_id}: {e}")
            raise

    # ------------------------------------------------------------------
    # PyTorch MLP path — primary, GPU-accelerated
    # ------------------------------------------------------------------

    def predict_rul_torch(
        self,
        model: Any,
        features: np.ndarray,
        return_confidence: bool = False,
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Predict RUL using the PyTorch MLP model with GPU acceleration.

        Inputs are moved to the same device as the model (CUDA if available).
        Monte-Carlo dropout is used for uncertainty estimation when
        return_confidence=True.

        Args:
            model:            Loaded PyTorch nn.Module
            features:         Flat feature vector (1, n_features) — produced from
                              engineered temporal features (lag, rolling, EMA)
            return_confidence: Return 95% CI from MC-dropout sampling

        Returns:
            (predicted_rul, confidence_interval_or_None)
        """
        if not _TORCH_AVAILABLE or torch is None:
            raise RuntimeError("torch is not installed")

        device = getattr(model, "_device", _TORCH_DEVICE)

        # Apply scaler if saved in checkpoint (normalizes to training distribution).
        # Skipping this step silently would cause wrong-scale inference — the
        # model was trained on scaled features, so raw features shift every
        # prediction.  Warn loudly (once per model object) instead.
        scaler = self._get_scaler(model)
        if scaler is not None:
            try:
                features = scaler.transform(features)
            except Exception as exc:
                if not getattr(model, "_scaler_warning_logged", False):
                    logger.warning(
                        "Scaler.transform() failed (%s) — falling back to raw "
                        "features. Predictions may be wrong-scale until the "
                        "model is reloaded with a working scaler.", exc,
                    )
                    model._scaler_warning_logged = True  # type: ignore[attr-defined]
        else:
            if not getattr(model, "_scaler_warning_logged", False):
                logger.warning(
                    "No scaler attached to model (model._scaler is None) — "
                    "running inference on raw, unscaled features. If this "
                    "model was trained on scaled features, predictions will "
                    "be systematically wrong. Retrain with scaler persistence "
                    "enabled or reload a checkpoint that embeds 'scaler'."
                )
                model._scaler_warning_logged = True  # type: ignore[attr-defined]

        # Move input to target device (GPU or CPU)
        x = torch.tensor(features, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            out = model(x)
            rul = float(out.squeeze().item())

        rul = max(0.0, min(rul, 200.0))

        confidence_interval: Optional[Dict[str, float]] = None
        if return_confidence:
            # Monte-Carlo dropout — enable dropout at inference time
            def _enable_dropout(m: Any) -> None:
                if m.__class__.__name__.startswith("Dropout"):
                    m.train()

            model.apply(_enable_dropout)
            mc_preds: List[float] = []
            with torch.no_grad():
                for _ in range(30):
                    p = model(x)
                    mc_preds.append(float(p.squeeze().item()))
            model.eval()

            mc_arr = np.array(mc_preds)
            confidence_interval = {
                "lower": float(np.percentile(mc_arr, 2.5)),
                "upper": float(np.percentile(mc_arr, 97.5)),
                "std": float(np.std(mc_arr)),
                "confidence": float(1.0 - np.std(mc_arr) / (np.mean(np.abs(mc_arr)) + 1e-8)),
            }

        return rul, confidence_interval

    def preprocess_flat_features(
        self,
        sensor_readings: Dict[str, float],
        equipment_id: str,
    ) -> np.ndarray:
        """
        Preprocess a single sensor reading dict into a flat feature vector
        for the PyTorch MLP.

        Accepts any dict of {feature_name: float} — values are extracted in
        sorted key order so the vector is deterministic across calls. This
        matches the API request format where keys are sensor names like
        'temperature', 'vibration', 'pressure', etc.

        If only a subset of features is provided, the vector is zero-padded
        or truncated to self.n_features.

        Args:
            sensor_readings: Dict mapping sensor/feature names to float values
            equipment_id:    Equipment identifier (used for logging)

        Returns:
            np.ndarray of shape (1, n_features)
        """
        try:
            if not sensor_readings:
                logger.warning(
                    "Empty sensor_readings for %s — returning zero vector", equipment_id
                )
                return np.zeros((1, self.n_features), dtype=np.float32)

            # Extract values in sorted key order for determinism
            sorted_keys = sorted(sensor_readings.keys())
            values = [float(sensor_readings[k]) for k in sorted_keys]

            # Pad or truncate to expected input dimension
            if len(values) < self.n_features:
                values += [0.0] * (self.n_features - len(values))
            elif len(values) > self.n_features:
                values = values[: self.n_features]

            feature_vector = np.array(values, dtype=np.float32).reshape(1, -1)

            return feature_vector

        except Exception as e:
            logger.error("Flat feature preprocessing failed for %s: %s", equipment_id, e)
            raise

    def _get_scaler(self, model: Any):
        """Return the best available scaler: model-attached first, instance fallback."""
        model_scaler = getattr(model, "_scaler", None)
        if model_scaler is not None:
            return model_scaler
        return self.scaler if getattr(self.scaler, "mean_", None) is not None else None

    def predict_rul(
        self,
        model: Any,
        sequence: np.ndarray,
        return_confidence: bool = False,
    ) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Predict RUL — routes to PyTorch MLP (primary) or TensorFlow LSTM (legacy)
        based on what framework the model belongs to.

        Predict RUL using LSTM model

        Args:
            model: Loaded LSTM model
            sequence: Preprocessed sequence (1, sequence_length, n_features)
            return_confidence: Whether to return confidence interval

        Returns:
            Tuple of (predicted_rul, confidence_interval)
        """
        try:
            # Route to PyTorch MLP if the model carries the pytorch marker
            if _TORCH_AVAILABLE and torch is not None and isinstance(
                model, torch.nn.Module
            ):
                return self.predict_rul_torch(model, sequence, return_confidence)

            # TensorFlow / Keras LSTM (legacy path)
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

    @staticmethod
    def get_recommendations(rul: float, health_status: str) -> List[str]:
        """
        Generate actionable recommendations based on RUL and health status.

        Args:
            rul: Predicted remaining useful life (cycles).
            health_status: Derived health status string.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if health_status == "imminent_failure":
            recommendations.append("URGENT: Schedule immediate maintenance or shutdown")
            recommendations.append(f"Equipment has only ~{rul:.0f} cycles remaining")
        elif health_status == "critical":
            hours = rul * 0.5
            recommendations.append(f"Schedule maintenance within {hours:.0f} hours")
            recommendations.append(
                "Increase monitoring frequency to 5-minute intervals"
            )
        elif health_status == "warning":
            recommendations.append("Plan maintenance in the next scheduled window")
            recommendations.append("Monitor sensor trends for accelerating degradation")
        else:
            recommendations.append("No action required — equipment operating normally")

        return recommendations
