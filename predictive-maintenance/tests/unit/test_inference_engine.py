"""Unit tests for InferenceEngine — preprocessing and prediction helpers."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from models.inference_engine import InferenceEngine


def _make_feature_dict(n_features: int = 10) -> dict:
    """Return a dict of feature_0..feature_N with random values."""
    return {f"feature_{i}": float(np.random.randn()) for i in range(n_features)}


def _make_sequence(length: int = 50, n_features: int = 10):
    """Return a list of feature dicts (one per timestep)."""
    return [_make_feature_dict(n_features) for _ in range(length)]


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPreprocessSequence:
    def test_output_shape(self):
        engine = InferenceEngine(sequence_length=10, n_features=5)
        seq = _make_sequence(length=10, n_features=5)
        arr = engine.preprocess_sequence(seq, "eq1")
        assert arr.shape == (1, 10, 5)

    def test_pads_short_sequence(self):
        engine = InferenceEngine(sequence_length=10, n_features=5)
        seq = _make_sequence(length=3, n_features=5)
        arr = engine.preprocess_sequence(seq, "eq1")
        assert arr.shape == (1, 10, 5)

    def test_truncates_long_sequence(self):
        engine = InferenceEngine(sequence_length=10, n_features=5)
        seq = _make_sequence(length=20, n_features=5)
        arr = engine.preprocess_sequence(seq, "eq1")
        assert arr.shape == (1, 10, 5)


@pytest.mark.unit
class TestPreprocessFeatures:
    def test_output_shape(self):
        engine = InferenceEngine(n_features=8)
        feats = _make_feature_dict(8)
        arr = engine.preprocess_features(feats, "eq1")
        assert arr.shape == (1, 8)


# ═══════════════════════════════════════════════════════════════════════════
# Health status from RUL
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestHealthStatusFromRUL:
    def test_healthy(self):
        engine = InferenceEngine()
        assert engine.get_health_status_from_rul(150) == "healthy"

    def test_warning(self):
        engine = InferenceEngine()
        assert engine.get_health_status_from_rul(60) == "warning"

    def test_critical(self):
        engine = InferenceEngine()
        assert engine.get_health_status_from_rul(35) == "critical"

    def test_imminent_failure(self):
        engine = InferenceEngine()
        assert engine.get_health_status_from_rul(5) == "imminent_failure"


# ═══════════════════════════════════════════════════════════════════════════
# Input validation
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestValidateInput:
    def test_valid_input(self):
        engine = InferenceEngine(n_features=5)
        feats = _make_feature_dict(5)
        valid, msg = engine.validate_input(feats, "eq1")
        assert valid is True

    def test_nan_rejected(self):
        engine = InferenceEngine(n_features=5)
        feats = _make_feature_dict(5)
        feats["feature_0"] = float("nan")
        valid, msg = engine.validate_input(feats, "eq1")
        assert valid is False

    def test_inf_rejected(self):
        engine = InferenceEngine(n_features=5)
        feats = _make_feature_dict(5)
        feats["feature_0"] = float("inf")
        valid, msg = engine.validate_input(feats, "eq1")
        assert valid is False


# ═══════════════════════════════════════════════════════════════════════════
# Predict RUL (with mocked model)
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPredictRUL:
    def test_rul_clipped_to_range(self):
        engine = InferenceEngine(sequence_length=10, n_features=5)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[250.0]])  # above clip

        seq = np.random.randn(1, 10, 5).astype(np.float32)
        rul, _ = engine.predict_rul(mock_model, seq)
        assert 0 <= rul <= 200

    def test_negative_prediction_clipped(self):
        engine = InferenceEngine(sequence_length=10, n_features=5)
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[-50.0]])

        seq = np.random.randn(1, 10, 5).astype(np.float32)
        rul, _ = engine.predict_rul(mock_model, seq)
        assert rul >= 0
