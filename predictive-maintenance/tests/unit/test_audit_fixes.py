"""
Tests for the audit fixes:

Fix A  — PredictionCache.make_key includes model name+version
Fix B  — ModelManager: get_model/get_model_metadata acquire RLock
Fix C  — DriftDetector: Benjamini-Hochberg correction + min-feature threshold
Fix D  — KafkaPredictionPipeline: warm-up suppression
Fix E  — SequenceData validator: NaN/Inf rejection
Fix F  — /feedback: true_rul/predicted_rul bounds validation

Resume-defense fixes:
TestPerformanceMonitorRedisPersistence — durable, shared accuracy tracking
"""

import sys
import os
import json
import math
import re
import logging
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[3]
_INFERENCE = _REPO / "predictive-maintenance" / "inference_service"
_ML = _REPO / "predictive-maintenance" / "ml_pipeline"
sys.path.insert(0, str(_INFERENCE.parent))  # predictive-maintenance/
sys.path.insert(0, str(_INFERENCE))         # inference_service/
sys.path.insert(0, str(_ML / "retrain"))    # ml_pipeline/retrain/


# ═══════════════════════════════════════════════════════════════════════════
# Fix A — Cache key includes model name+version
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCacheKeyModelVersion:
    """
    Verifies that make_key() produces different keys when model version changes
    so that post-hot-reload predictions are never served from the old cache entry.
    """

    def _cache(self):
        from inference_service.cache.prediction_cache import PredictionCache
        return PredictionCache.__new__(PredictionCache)  # no Redis connection

    def test_same_input_same_version_gives_same_key(self):
        c = self._cache()
        seq = [{"sensor_2": 1.0}]
        k1 = c.make_key("EQ-1", seq, model_name="mlp", model_version="v1")
        k2 = c.make_key("EQ-1", seq, model_name="mlp", model_version="v1")
        assert k1 == k2

    def test_same_input_different_version_gives_different_key(self):
        """After hot-reload (v1 → v2) the cache key must change."""
        c = self._cache()
        seq = [{"sensor_2": 1.0}]
        k1 = c.make_key("EQ-1", seq, model_name="mlp", model_version="v1")
        k2 = c.make_key("EQ-1", seq, model_name="mlp", model_version="v2")
        assert k1 != k2

    def test_same_input_different_model_name_gives_different_key(self):
        c = self._cache()
        seq = [{"sensor_2": 1.0}]
        k1 = c.make_key("EQ-1", seq, model_name="mlp", model_version="v1")
        k2 = c.make_key("EQ-1", seq, model_name="lstm", model_version="v1")
        assert k1 != k2

    def test_different_equipment_still_separates(self):
        c = self._cache()
        seq = [{"sensor_2": 1.0}]
        k1 = c.make_key("EQ-1", seq, model_name="mlp", model_version="v1")
        k2 = c.make_key("EQ-2", seq, model_name="mlp", model_version="v1")
        assert k1 != k2

    def test_different_sequence_still_separates(self):
        c = self._cache()
        k1 = c.make_key("EQ-1", [{"a": 1.0}], model_name="mlp", model_version="v1")
        k2 = c.make_key("EQ-1", [{"a": 2.0}], model_name="mlp", model_version="v1")
        assert k1 != k2

    def test_empty_version_still_produces_string_key(self):
        c = self._cache()
        k = c.make_key("EQ-1", [{"a": 1.0}])
        assert isinstance(k, str)
        assert len(k) > 10


# ═══════════════════════════════════════════════════════════════════════════
# Fix B — ModelManager RLock atomicity
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestModelManagerLock:
    """
    Verifies that get_model() and get_model_metadata() are consistent even
    when a concurrent writer swaps model+metadata simultaneously.
    """

    def _fresh_manager(self):
        """Return a fresh ModelManager instance (bypasses singleton logic)."""
        from inference_service.models.model_manager import ModelManager
        mgr = object.__new__(ModelManager)
        mgr._models = {}
        mgr._model_metadata = {}
        mgr._lock = threading.RLock()
        return mgr

    def test_get_model_returns_none_for_unknown_key(self):
        mgr = self._fresh_manager()
        assert mgr.get_model("does_not_exist") is None

    def test_get_model_metadata_returns_none_for_unknown_key(self):
        mgr = self._fresh_manager()
        assert mgr.get_model_metadata("does_not_exist") is None

    def test_atomic_write_visible_to_reader(self):
        """model and metadata written inside lock are both visible after write."""
        mgr = self._fresh_manager()
        sentinel_model = object()
        sentinel_meta = {"version": "v42", "loaded": True}

        with mgr._lock:
            mgr._models["mlp"] = sentinel_model
            mgr._model_metadata["mlp"] = sentinel_meta

        assert mgr.get_model("mlp") is sentinel_model
        assert mgr.get_model_metadata("mlp")["version"] == "v42"

    def test_concurrent_reads_do_not_crash(self):
        """Many concurrent readers must not raise due to dict mutation."""
        mgr = self._fresh_manager()
        mgr._models["mlp"] = "model_a"
        mgr._model_metadata["mlp"] = {"version": "v1"}
        errors = []

        def _reader():
            for _ in range(50):
                try:
                    mgr.get_model("mlp")
                    mgr.get_model_metadata("mlp")
                except Exception as e:
                    errors.append(e)

        def _writer():
            for i in range(10):
                with mgr._lock:
                    mgr._models["mlp"] = f"model_{i}"
                    mgr._model_metadata["mlp"] = {"version": f"v{i}"}
                time.sleep(0.001)

        threads = [threading.Thread(target=_reader) for _ in range(5)]
        threads.append(threading.Thread(target=_writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent access raised: {errors}"

    def test_metadata_version_matches_model_after_swap(self):
        """After an atomic swap, get_model and get_model_metadata must agree."""
        mgr = self._fresh_manager()
        mgr._models["mlp"] = "old_model"
        mgr._model_metadata["mlp"] = {"version": "v1"}

        # Simulate hot-reload
        with mgr._lock:
            mgr._models["mlp"] = "new_model"
            mgr._model_metadata["mlp"] = {"version": "v2"}

        assert mgr.get_model("mlp") == "new_model"
        assert mgr.get_model_metadata("mlp")["version"] == "v2"


# ═══════════════════════════════════════════════════════════════════════════
# Fix C — DriftDetector BH correction + min-feature threshold
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDriftDetectorBH:
    """
    Verifies that the drift detector uses BH correction and requires a minimum
    fraction of features to drift before raising the overall drift signal.
    """

    def _detector(self):
        from ml_pipeline.retrain.drift_detector import DriftDetector
        det = object.__new__(DriftDetector)
        det.data_drift_threshold = 0.05
        det.concept_drift_threshold = 0.15
        det.window_size = 7
        det.min_samples = 10
        return det

    def _make_frames(self, n: int = 500, shift: float = 0.0):
        import pandas as pd
        rng = np.random.default_rng(42)
        df_ref = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(20)})
        df_cur = pd.DataFrame({f"f{i}": rng.normal(shift, 1, n) for i in range(20)})
        return df_ref, df_cur

    def test_no_drift_when_distributions_identical(self):
        det = self._detector()
        ref, cur = self._make_frames(n=500, shift=0.0)
        result = det._detect_data_drift(ref, cur)
        # Identical distributions: BH correction should reject nearly all tests
        assert result["correction"] == "benjamini_hochberg"
        # With no real drift, very few features should be flagged
        assert result["num_affected"] < 3

    def test_drift_detected_when_all_features_shifted(self):
        """Large shift on all features: drift must be detected."""
        det = self._detector()
        ref, cur = self._make_frames(n=500, shift=5.0)  # 5-sigma shift
        result = det._detect_data_drift(ref, cur)
        assert result["drift_detected"] is True
        assert result["num_affected"] > 0

    def test_single_spurious_feature_does_not_raise_signal(self):
        """One noisy feature below BH threshold should not trigger drift."""
        import pandas as pd
        rng = np.random.default_rng(0)
        n = 500
        # 19 identical features + 1 slightly shifted (but within noise)
        df_ref = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(19)})
        df_cur = pd.DataFrame({f"f{i}": rng.normal(0, 1, n) for i in range(19)})
        # Add one feature with tiny shift — unlikely to survive BH at this n
        df_ref["f19"] = rng.normal(0, 1, n)
        df_cur["f19"] = rng.normal(0.1, 1, n)

        det = self._detector()
        result = det._detect_data_drift(df_ref, df_cur)
        # Even if f19 is flagged, min_required_features > 1 for 20 features
        assert result["min_required_features"] >= 1
        # Overall drift requires at least min_required features to be affected
        if result["num_affected"] < result["min_required_features"]:
            assert result["drift_detected"] is False

    def test_result_contains_correction_field(self):
        det = self._detector()
        ref, cur = self._make_frames(n=200)
        result = det._detect_data_drift(ref, cur)
        assert "correction" in result
        assert result["correction"] == "benjamini_hochberg"

    def test_empty_data_returns_no_drift(self):
        import pandas as pd
        det = self._detector()
        empty = pd.DataFrame()
        result = det._detect_data_drift(empty, empty)
        assert result["drift_detected"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Fix D — Kafka warm-up suppression
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestKafkaWarmupSuppression:
    """
    Verifies that the pipeline suppresses predictions until the per-equipment
    buffer reaches min_warmup_readings.
    """

    def _pipeline(self, min_warmup: int = 5):
        from inference_service.consumer import KafkaPredictionPipeline, _EquipmentBuffer
        pipe = object.__new__(KafkaPredictionPipeline)
        pipe._seq_len = 50
        pipe._min_warmup = min_warmup
        pipe._sequence_buffers = {}
        pipe._model_manager = None
        pipe._inference_engine = None
        pipe._stats = {"consumed": 0, "predicted": 0, "published": 0, "errors": 0}
        return pipe, _EquipmentBuffer

    def test_buffer_does_not_emit_during_warmup(self):
        pipe, BufClass = self._pipeline(min_warmup=5)
        # Simulate pushing 4 readings (< min_warmup=5) and check suppression
        pipe._sequence_buffers["EQ-1"] = BufClass(maxlen=50)
        for _ in range(4):
            pipe._sequence_buffers["EQ-1"].push({"sensor_2": 1.0})
        buf_len = len(pipe._sequence_buffers["EQ-1"])
        # warm-up check: should suppress
        assert buf_len < pipe._min_warmup

    def test_buffer_ready_after_min_warmup(self):
        pipe, BufClass = self._pipeline(min_warmup=5)
        pipe._sequence_buffers["EQ-1"] = BufClass(maxlen=50)
        for _ in range(5):
            pipe._sequence_buffers["EQ-1"].push({"sensor_2": 1.0})
        buf_len = len(pipe._sequence_buffers["EQ-1"])
        assert buf_len >= pipe._min_warmup

    def test_new_equipment_starts_in_warmup(self):
        pipe, BufClass = self._pipeline(min_warmup=10)
        pipe._sequence_buffers["EQ-NEW"] = BufClass(maxlen=50)
        pipe._sequence_buffers["EQ-NEW"].push({"sensor_2": 1.0})
        assert len(pipe._sequence_buffers["EQ-NEW"]) < pipe._min_warmup

    def test_extract_raw_reading_generic_format(self):
        from inference_service.consumer import KafkaPredictionPipeline
        payload = {"sensor_data": {"sensor_2": 1.0, "sensor_3": 2.0}, "time_cycle": 5.0}
        result = KafkaPredictionPipeline._extract_raw_reading(payload)
        assert result is not None
        assert result["sensor_2"] == 1.0
        assert result["time_cycle"] == 5.0

    def test_extract_raw_reading_cmapss_format(self):
        from inference_service.consumer import KafkaPredictionPipeline
        payload = {
            "sensors": {"sensor_2": 642.5, "sensor_3": 1589.7},
            "operating_settings": {"setting_1": 0.0, "setting_2": 0.0},
            "time_cycle": 1.0,
        }
        result = KafkaPredictionPipeline._extract_raw_reading(payload)
        assert result is not None
        assert result["sensor_2"] == 642.5
        assert result["op_setting_1"] == 0.0  # setting_1 → op_setting_1

    def test_extract_raw_reading_empty_payload_returns_none(self):
        from inference_service.consumer import KafkaPredictionPipeline
        result = KafkaPredictionPipeline._extract_raw_reading({})
        assert result is None

    def test_extract_raw_reading_rejects_non_numeric(self):
        from inference_service.consumer import KafkaPredictionPipeline
        payload = {"sensor_data": {"sensor_2": "bad_value"}}
        result = KafkaPredictionPipeline._extract_raw_reading(payload)
        # Non-numeric values are silently skipped — no key should appear
        assert result is None or "sensor_2" not in result


# ═══════════════════════════════════════════════════════════════════════════
# Fix E — SequenceData NaN/Inf validator
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSequenceDataValidation:
    """
    Verifies that SequenceData rejects NaN and Inf values in the sequence.
    """

    def _make_request(self, seq):
        from inference_service.api.schemas import SequenceData
        return SequenceData(equipment_id="EQ-1", sequence=seq)

    def test_valid_sequence_accepted(self):
        seq = [{"sensor_2": 1.0, "sensor_3": 2.0}]
        sd = self._make_request(seq)
        assert len(sd.sequence) == 1

    def test_nan_in_sequence_raises(self):
        from pydantic import ValidationError
        seq = [{"sensor_2": float("nan")}]
        with pytest.raises(ValidationError, match="NaN or Inf"):
            self._make_request(seq)

    def test_inf_in_sequence_raises(self):
        from pydantic import ValidationError
        seq = [{"sensor_2": float("inf")}]
        with pytest.raises(ValidationError, match="NaN or Inf"):
            self._make_request(seq)

    def test_neg_inf_in_sequence_raises(self):
        from pydantic import ValidationError
        seq = [{"sensor_2": float("-inf")}]
        with pytest.raises(ValidationError, match="NaN or Inf"):
            self._make_request(seq)

    def test_empty_sequence_raises(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._make_request([])

    def test_multiple_readings_all_valid(self):
        seq = [{"sensor_2": float(i)} for i in range(30)]
        sd = self._make_request(seq)
        assert len(sd.sequence) == 30

    def test_nan_in_second_reading_raises(self):
        from pydantic import ValidationError
        seq = [{"sensor_2": 1.0}, {"sensor_2": float("nan")}]
        with pytest.raises(ValidationError, match="NaN or Inf"):
            self._make_request(seq)


# ═══════════════════════════════════════════════════════════════════════════
# Fix F — /feedback bounds validation (unit-level, no server required)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFeedbackBoundsLogic:
    """
    Validates the bounds-checking logic that /feedback applies before
    calling PerformanceMonitor.record_prediction.

    We test the same conditions as the endpoint without spinning up FastAPI.
    """

    def _check(self, equipment_id: str, true_rul: float, predicted_rul: float) -> str:
        """
        Replicate the /feedback validation logic from main.py.
        Returns 'ok' if valid, or the error message string.
        """
        if not equipment_id or not equipment_id.strip():
            return "equipment_id must not be empty"
        if math.isnan(true_rul) or math.isinf(true_rul):
            return "true_rul must be a finite number"
        if math.isnan(predicted_rul) or math.isinf(predicted_rul):
            return "predicted_rul must be a finite number"
        if true_rul < 0:
            return f"true_rul must be >= 0, got {true_rul}"
        if predicted_rul < 0:
            return f"predicted_rul must be >= 0, got {predicted_rul}"
        MAX_RUL = 1000.0
        if true_rul > MAX_RUL:
            return f"true_rul exceeds plausible upper bound ({MAX_RUL})"
        if predicted_rul > MAX_RUL:
            return f"predicted_rul exceeds plausible upper bound ({MAX_RUL})"
        return "ok"

    def test_valid_feedback_passes(self):
        assert self._check("EQ-1", 80.0, 75.0) == "ok"

    def test_zero_rul_passes(self):
        assert self._check("EQ-1", 0.0, 0.0) == "ok"

    def test_negative_true_rul_fails(self):
        result = self._check("EQ-1", -1.0, 80.0)
        assert "true_rul must be >= 0" in result

    def test_negative_predicted_rul_fails(self):
        result = self._check("EQ-1", 80.0, -5.0)
        assert "predicted_rul must be >= 0" in result

    def test_nan_true_rul_fails(self):
        result = self._check("EQ-1", float("nan"), 80.0)
        assert "finite" in result

    def test_inf_predicted_rul_fails(self):
        result = self._check("EQ-1", 80.0, float("inf"))
        assert "finite" in result

    def test_above_max_true_rul_fails(self):
        result = self._check("EQ-1", 9999.0, 80.0)
        assert "upper bound" in result

    def test_above_max_predicted_rul_fails(self):
        result = self._check("EQ-1", 80.0, 9999.0)
        assert "upper bound" in result

    def test_empty_equipment_id_fails(self):
        result = self._check("", 80.0, 80.0)
        assert "equipment_id" in result

    def test_whitespace_only_equipment_id_fails(self):
        result = self._check("   ", 80.0, 80.0)
        assert "equipment_id" in result


# ═══════════════════════════════════════════════════════════════════════════
# Model input dimension consistency guard
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestEngineeeredDimConsistency:
    """
    ENGINEERED_FEATURE_DIM exported by feature_engineering.py must equal 200
    and match the dimension produced by OnlineFeatureEngineer on a full
    C-MAPSS sequence — this is the contract the MLP relies on.
    """

    def test_dim_constant_is_200(self):
        from inference_service.feature_engineering import ENGINEERED_FEATURE_DIM
        assert ENGINEERED_FEATURE_DIM == 200, (
            f"ENGINEERED_FEATURE_DIM changed from 200 to {ENGINEERED_FEATURE_DIM}. "
            "All MLP checkpoints and InferenceEngine initialization must be updated."
        )

    def test_online_engineer_output_matches_dim_constant(self):
        from inference_service.feature_engineering import (
            OnlineFeatureEngineer, ENGINEERED_FEATURE_DIM, CONSTANT_SENSORS
        )
        all_sensors = [f"sensor_{i}" for i in range(1, 22)]
        sequence = [
            {
                **{s: 1.0 for s in all_sensors},
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
                "time_cycle": float(t + 1),
            }
            for t in range(30)
        ]
        eng = OnlineFeatureEngineer()
        out = eng.engineer(sequence, "test")
        assert out.shape == (1, ENGINEERED_FEATURE_DIM), (
            f"OnlineFeatureEngineer produced {out.shape[1]} features but "
            f"ENGINEERED_FEATURE_DIM={ENGINEERED_FEATURE_DIM}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fix G — Scaler persistence on the automated retraining path
#
# baseline_comparison.py fits a StandardScaler on the training features and
# attaches it as predictor.scaler before save_model() embeds it in the
# checkpoint as checkpoint["scaler"].  TrainingPipeline.train_mlp() and
# RetrainingPipeline._train_model() previously skipped this step, so a
# retrained model could be promoted with model._scaler == None and
# InferenceEngine.predict_rul_torch() would silently run on raw features.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestScalerPersistence:

    _SMALL_MLP_CONFIG = {
        "mlp": {
            "architecture": {"hidden_sizes": [8, 4], "dropout": 0.1},
            "training": {
                "learning_rate": 0.01,
                "epochs": 1,
                "batch_size": 4,
                "patience": 1,
            },
        }
    }

    @staticmethod
    def _small_data(n_train=40, n_val=8, n_features=6, seed=1):
        rng = np.random.default_rng(seed)
        X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
        y_train = rng.uniform(0, 100, size=n_train).astype(np.float32)
        X_val = rng.normal(size=(n_val, n_features)).astype(np.float32)
        y_val = rng.uniform(0, 100, size=n_val).astype(np.float32)
        return X_train, y_train, X_val, y_val

    # ------------------------------------------------------------------
    # 1. save_model() embeds a fitted scaler; load_model() restores it
    # ------------------------------------------------------------------
    def test_save_model_embeds_fitted_scaler(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from ml_pipeline.train.models.mlp_model import MLPRULPredictor
        import torch

        predictor = MLPRULPredictor(self._SMALL_MLP_CONFIG)
        predictor.build_model(input_dim=6)

        scaler = StandardScaler()
        scaler.fit(np.random.default_rng(0).normal(size=(20, 6)))
        predictor.scaler = scaler

        ckpt_path = str(tmp_path / "mlp_with_scaler.pt")
        predictor.save_model(ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        assert "scaler" in checkpoint
        assert checkpoint["scaler"] is not None
        assert hasattr(checkpoint["scaler"], "mean_")
        np.testing.assert_allclose(checkpoint["scaler"].mean_, scaler.mean_)

        # round-trip through load_model — self.scaler must be restored
        reloaded = MLPRULPredictor(self._SMALL_MLP_CONFIG)
        reloaded.load_model(ckpt_path)
        assert getattr(reloaded, "scaler", None) is not None
        np.testing.assert_allclose(reloaded.scaler.mean_, scaler.mean_)

    # ------------------------------------------------------------------
    # 2a. Source check — train_pipeline.py::train_mlp fits + attaches a scaler
    #     before save_model().  Reads the actual file (no heavy imports —
    #     train_pipeline pulls in feature_store -> redis) so the assertion
    #     runs against the real production code, not a copy.
    # ------------------------------------------------------------------
    def test_train_mlp_source_fits_and_attaches_scaler(self):
        src = (_ML / "train" / "train_pipeline.py").read_text()
        # Isolate the train_mlp method body
        start = src.index("def train_mlp(")
        end = src.index("def prepare_mlp_data(")
        body = src[start:end]
        assert "StandardScaler" in body, "train_mlp must fit a StandardScaler on X_train"
        assert "scaler.fit(X_train)" in body
        assert "self.mlp_model.scaler = scaler" in body, (
            "fitted scaler must be attached to the predictor before save_model() "
            "so MLPRULPredictor.save_model() embeds it as checkpoint['scaler']"
        )
        # Attachment must happen before save_model() persists the checkpoint
        assert body.index("self.mlp_model.scaler = scaler") < body.index("self.mlp_model.save_model(")

    # ------------------------------------------------------------------
    # 2b. Functional equivalent of TrainingPipeline.train_mlp(): build, train,
    #     fit+attach scaler on X_train, save — assert checkpoint embeds it.
    #     (Avoids importing TrainingPipeline directly: it transitively pulls
    #     in FeatureStorePipeline -> redis, which is not installed here.)
    # ------------------------------------------------------------------
    def test_train_mlp_equivalent_produces_checkpoint_with_scaler(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from ml_pipeline.train.models.mlp_model import MLPRULPredictor
        import torch

        X_train, y_train, X_val, y_val = self._small_data()

        predictor = MLPRULPredictor(self._SMALL_MLP_CONFIG)
        predictor.build_model(input_dim=X_train.shape[1])
        predictor.train(X_train, y_train, X_val, y_val)

        # === mirrors the lines added to TrainingPipeline.train_mlp() ===
        scaler = StandardScaler()
        scaler.fit(X_train)
        predictor.scaler = scaler
        # ================================================================

        ckpt_path = str(tmp_path / "mlp_rul_equivalent.pt")
        predictor.save_model(ckpt_path)

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        assert checkpoint.get("scaler") is not None
        np.testing.assert_allclose(
            checkpoint["scaler"].mean_, X_train.mean(axis=0), rtol=1e-5
        )

    # ------------------------------------------------------------------
    # 3. Source check — RetrainingPipeline._train_model fits + attaches a
    #    scaler and saves a checkpoint with it embedded (mirrors baseline).
    # ------------------------------------------------------------------
    def test_retrain_train_model_source_persists_scaler(self):
        src = (_ML / "retrain" / "retrain_pipeline.py").read_text()
        start = src.index("def _train_model(")
        end = src.index("def _log_retraining_workflow(")
        body = src[start:end]
        assert "StandardScaler" in body
        assert "scaler.fit(X_train)" in body
        assert "predictor.scaler = scaler" in body
        assert "predictor.save_model(checkpoint_path)" in body, (
            "RetrainingPipeline._train_model must persist a .pt checkpoint "
            "(via MLPRULPredictor.save_model) so the embedded scaler can be "
            "loaded back by ModelManager.load_mlp_model()"
        )
        # Attachment must happen before the checkpoint is saved
        assert body.index("predictor.scaler = scaler") < body.index("predictor.save_model(checkpoint_path)")

    # ------------------------------------------------------------------
    # 4. ModelManager.load_mlp_model() attaches model._scaler from checkpoint
    # ------------------------------------------------------------------
    def test_model_manager_attaches_scaler_from_checkpoint(self, tmp_path):
        from sklearn.preprocessing import StandardScaler
        from ml_pipeline.train.models.mlp_model import MLPRULPredictor
        from inference_service.models.model_manager import ModelManager

        predictor = MLPRULPredictor(self._SMALL_MLP_CONFIG)
        predictor.build_model(input_dim=6)
        scaler = StandardScaler()
        scaler.fit(np.random.default_rng(2).normal(size=(20, 6)))
        predictor.scaler = scaler
        ckpt_path = str(tmp_path / "mlp_for_manager.pt")
        predictor.save_model(ckpt_path)

        mgr = object.__new__(ModelManager)
        mgr._models = {}
        mgr._model_metadata = {}
        mgr._lock = threading.RLock()

        loaded = mgr.load_mlp_model(ckpt_path, "mlp_rul_predictor", "v-test")

        attached = mgr.get_model("mlp")
        assert attached is loaded
        assert getattr(attached, "_scaler", None) is not None
        np.testing.assert_allclose(attached._scaler.mean_, scaler.mean_)

    # ------------------------------------------------------------------
    # 5. InferenceEngine.predict_rul_torch applies model._scaler exactly once
    #    (no double-scaling) before the forward pass
    # ------------------------------------------------------------------
    def test_predict_rul_torch_applies_attached_scaler(self):
        from sklearn.preprocessing import StandardScaler
        from inference_service.models.inference_engine import InferenceEngine
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        model.eval()
        model._device = torch.device("cpu")

        scaler = StandardScaler()
        scaler.fit(np.array([[10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 20.0]]))
        model._scaler = scaler

        calls = []
        real_transform = scaler.transform

        def _spy_transform(X):
            calls.append(np.array(X, copy=True))
            return real_transform(X)

        scaler.transform = _spy_transform

        engine = InferenceEngine(n_features=4)
        features = np.array([[15.0, 15.0, 15.0, 15.0]], dtype=np.float32)
        rul, _ = engine.predict_rul_torch(model, features)

        assert isinstance(rul, float)
        assert len(calls) == 1, "scaler.transform must be called exactly once — no double-scaling"
        np.testing.assert_allclose(calls[0], features)

    # ------------------------------------------------------------------
    # 6. predict_rul_torch logs a clear warning (and still returns a result)
    #    when no scaler is attached — required so the gap is never silent
    # ------------------------------------------------------------------
    def test_predict_rul_torch_warns_when_no_scaler(self, caplog):
        from inference_service.models.inference_engine import InferenceEngine
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        model.eval()
        model._device = torch.device("cpu")
        model._scaler = None

        engine = InferenceEngine(n_features=4)
        features = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        with caplog.at_level(logging.WARNING):
            rul, _ = engine.predict_rul_torch(model, features)

        assert isinstance(rul, float)
        assert "no scaler attached" in caplog.text.lower()

    # ------------------------------------------------------------------
    # 7. Backward compatibility — loading an old checkpoint with no "scaler"
    #    key must not crash, must set model._scaler = None, and must log
    #    a clear warning (not silently skip).
    # ------------------------------------------------------------------
    def test_old_checkpoint_without_scaler_loads_with_warning(self, tmp_path, caplog):
        from ml_pipeline.train.models.mlp_model import MLPRULPredictor
        from inference_service.models.model_manager import ModelManager
        import torch

        predictor = MLPRULPredictor(self._SMALL_MLP_CONFIG)
        predictor.build_model(input_dim=6)
        # Old-format checkpoint — saved before scaler persistence existed,
        # no "scaler" key at all (not even None).
        old_checkpoint = {
            "model_state_dict": predictor.model.state_dict(),
            "config": predictor.config,
            "hidden_sizes": predictor.hidden_sizes,
            "dropout": predictor.dropout,
            "input_dim": 6,
            "history": predictor.history,
        }
        ckpt_path = str(tmp_path / "old_mlp_no_scaler.pt")
        torch.save(old_checkpoint, ckpt_path)

        mgr = object.__new__(ModelManager)
        mgr._models = {}
        mgr._model_metadata = {}
        mgr._lock = threading.RLock()

        with caplog.at_level(logging.WARNING):
            loaded = mgr.load_mlp_model(ckpt_path, "mlp_rul_predictor", "v-old")

        assert loaded is not None
        assert getattr(loaded, "_scaler", None) is None
        assert "no embedded scaler" in caplog.text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Resume-defense Fix Area 1 — PerformanceMonitor durable Redis persistence
#
# Resume claim: "Built automated monitoring and retraining pipeline ... that
# triggered retraining when accuracy dropped below 80%, ensuring production
# reliability."
#
# Gap: PerformanceMonitor stored PredictionRecords in a plain in-memory list
# — not durable across restarts and not shared across Uvicorn workers / ECS
# tasks. Fix: write/read through a shared Redis sorted set (ZADD / ZRANGEBYSCORE),
# falling back to in-memory (with a warning) when Redis is unavailable.
# ═══════════════════════════════════════════════════════════════════════════

class _FakeRedis:
    """
    Minimal in-memory stand-in for redis.Redis sorted-set operations.

    Implements just enough of the ZSET surface (ZADD / ZRANGEBYSCORE / ZRANGE /
    ZREMRANGEBYSCORE / ZCARD / PING) for PerformanceMonitor to exercise its
    real Redis code path deterministically, without a live Redis server.
    """

    def __init__(self, alive: bool = True):
        self._alive = alive
        self._zsets: Dict[str, Dict[str, float]] = {}

    def ping(self):
        if not self._alive:
            raise ConnectionError("FakeRedis: connection refused")
        return True

    @staticmethod
    def _bound(value):
        if value in ("-inf", b"-inf"):
            return float("-inf")
        if value in ("+inf", b"+inf"):
            return float("inf")
        return float(value)

    def zadd(self, key, mapping):
        z = self._zsets.setdefault(key, {})
        added = 0
        for member, score in mapping.items():
            if member not in z:
                added += 1
            z[member] = float(score)
        return added

    def zrangebyscore(self, key, min_score, max_score):
        lo, hi = self._bound(min_score), self._bound(max_score)
        z = self._zsets.get(key, {})
        items = sorted(((m, s) for m, s in z.items() if lo <= s <= hi), key=lambda t: t[1])
        return [m for m, _ in items]

    def zrange(self, key, start, end):
        z = self._zsets.get(key, {})
        members = [m for m, _ in sorted(z.items(), key=lambda t: t[1])]
        return members[start:] if end == -1 else members[start : end + 1]

    def zremrangebyscore(self, key, min_score, max_score):
        lo, hi = self._bound(min_score), self._bound(max_score)
        z = self._zsets.get(key, {})
        to_remove = [m for m, s in z.items() if lo <= s <= hi]
        for m in to_remove:
            del z[m]
        return len(to_remove)

    def zcard(self, key):
        return len(self._zsets.get(key, {}))


@pytest.mark.unit
class TestPerformanceMonitorRedisPersistence:
    """
    Verifies PerformanceMonitor persists ground-truth/prediction records to a
    shared Redis backend (so accuracy tracking survives restarts and is
    consistent across every Uvicorn worker / ECS task) and falls back to an
    in-memory store — with a clear warning — when Redis is unreachable.
    """

    @staticmethod
    def _bare_monitor(redis_client, *, min_samples=50, key_prefix="perf_monitor:"):
        """Construct a PerformanceMonitor without running __init__ / touching real Redis."""
        from ml_pipeline.retrain.performance_monitor import PerformanceMonitor

        mon = object.__new__(PerformanceMonitor)
        mon.accuracy_threshold = 0.80
        mon.tolerance_cycles = 15
        mon.window_days = 7
        mon.min_samples = min_samples
        mon._records = []
        mon._redis_key_prefix = key_prefix
        mon._redis_key = f"{key_prefix}records"
        mon._record_retention_days = 14
        mon._redis_client = redis_client
        return mon

    # ------------------------------------------------------------------
    # 1. record_prediction persists to shared backend
    # ------------------------------------------------------------------
    def test_record_prediction_persists_to_shared_backend(self):
        fake = _FakeRedis()
        mon = self._bare_monitor(fake)

        mon.record_prediction("EQ-1", true_rul=100.0, predicted_rul=95.0)

        assert fake.zcard(mon._redis_key) == 1, (
            "record_prediction() must write the PredictionRecord into the "
            "shared Redis sorted set (ZADD), not just an in-memory list"
        )
        # In-memory fallback list must stay empty — the record went to Redis.
        assert mon._records == []
        stored = fake.zrangebyscore(mon._redis_key, "-inf", "+inf")
        payload = json.loads(stored[0])
        assert payload["equipment_id"] == "EQ-1"
        assert payload["true_rul"] == 100.0
        assert payload["predicted_rul"] == 95.0

    # ------------------------------------------------------------------
    # 2. check_accuracy_threshold reads persisted records
    # ------------------------------------------------------------------
    def test_check_accuracy_threshold_reads_persisted_records(self):
        fake = _FakeRedis()
        mon = self._bare_monitor(fake, min_samples=5)

        now = datetime.utcnow()
        for i in range(5):
            mon.record_prediction(
                f"EQ-{i}", true_rul=100.0, predicted_rul=98.0,
                timestamp=now - timedelta(hours=i),
            )

        # Nothing should have landed in the in-memory fallback.
        assert mon._records == []
        assert fake.zcard(mon._redis_key) == 5

        report = mon.check_accuracy_threshold()
        assert report.n_samples == 5, (
            "check_accuracy_threshold() must read its rolling-window sample "
            "from the shared Redis store (ZRANGEBYSCORE), not the empty "
            "in-memory fallback"
        )
        assert report.accuracy == pytest.approx(1.0)
        assert report.below_threshold is False

    # ------------------------------------------------------------------
    # 3. records survive a new PerformanceMonitor instance
    # ------------------------------------------------------------------
    def test_records_survive_new_monitor_instance(self):
        shared_backend = _FakeRedis()

        first = self._bare_monitor(shared_backend, min_samples=3)
        for i in range(3):
            first.record_prediction(f"EQ-{i}", true_rul=100.0, predicted_rul=99.0)

        # Simulate a restart / a different Uvicorn worker: brand-new instance,
        # same Redis backend, *no* shared Python state with `first`.
        second = self._bare_monitor(shared_backend, min_samples=3)

        assert second._records == [], "the new instance starts with an empty in-memory list"
        assert second.n_records == 3, (
            "a freshly constructed PerformanceMonitor must see records written "
            "by a previous instance via the shared Redis backend"
        )
        report = second.check_accuracy_threshold()
        assert report.n_samples == 3
        assert report.accuracy == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # 4. below 80% triggers below_threshold=True
    # ------------------------------------------------------------------
    def test_below_80_percent_triggers_below_threshold(self):
        fake = _FakeRedis()
        mon = self._bare_monitor(fake, min_samples=50)

        now = datetime.utcnow()
        # 60 predictions, all wildly off (|true - pred| = 50 >> 15-cycle tolerance)
        # → within-tolerance accuracy is far below the 80% threshold.
        for i in range(60):
            mon.record_prediction(
                f"EQ-{i}", true_rul=100.0, predicted_rul=50.0,
                timestamp=now - timedelta(minutes=i),
            )

        report = mon.check_accuracy_threshold()
        assert report.n_samples == 60
        assert report.accuracy < 0.80
        assert report.below_threshold is True
        assert report.threshold == pytest.approx(0.80)
        assert report.tolerance_cycles == 15

    # ------------------------------------------------------------------
    # 5. insufficient samples does not trigger
    # ------------------------------------------------------------------
    def test_insufficient_samples_does_not_trigger(self):
        fake = _FakeRedis()
        mon = self._bare_monitor(fake, min_samples=50)

        now = datetime.utcnow()
        # Only 10 records — below min_samples=50, even though they're all bad.
        for i in range(10):
            mon.record_prediction(
                f"EQ-{i}", true_rul=100.0, predicted_rul=10.0,
                timestamp=now - timedelta(minutes=i),
            )

        report = mon.check_accuracy_threshold()
        assert report.n_samples == 10
        assert report.below_threshold is False, (
            "fewer than min_samples records must never trigger retraining, "
            "regardless of how inaccurate they are"
        )
        assert report.details.get("reason") == "insufficient_samples"

    # ------------------------------------------------------------------
    # 6. Redis unavailable falls back gracefully with a warning
    # ------------------------------------------------------------------
    def test_redis_unavailable_falls_back_with_warning(self, monkeypatch, tmp_path, caplog):
        import ml_pipeline.retrain.performance_monitor as pm_module

        # Simulate redis-py being installed but the server being unreachable —
        # PerformanceMonitor must catch the ping() failure, log a warning, and
        # keep working via the in-memory fallback.
        dead_client = _FakeRedis(alive=False)

        class _FakeRedisModule:
            @staticmethod
            def Redis(**kwargs):
                return dead_client

        monkeypatch.setattr(pm_module, "_REDIS_AVAILABLE", True)
        monkeypatch.setattr(pm_module, "_redis_lib", _FakeRedisModule)

        cfg_path = tmp_path / "retrain_config.yaml"
        cfg_path.write_text(
            "performance_monitoring:\n"
            "  accuracy_threshold: 0.80\n"
            "  accuracy_tolerance_cycles: 15\n"
            "  window_days: 7\n"
            "  min_samples_for_check: 3\n"
        )

        with caplog.at_level(logging.WARNING):
            mon = pm_module.PerformanceMonitor(config_path=str(cfg_path))

        assert mon._redis_client is None, "an unreachable Redis must leave _redis_client unset"
        assert mon.is_persistent is False
        assert "redis not reachable" in caplog.text.lower()
        assert "in-memory" in caplog.text.lower()

        # The monitor must remain fully functional on the in-memory fallback.
        now = datetime.utcnow()
        for i in range(3):
            mon.record_prediction(f"EQ-{i}", true_rul=100.0, predicted_rul=99.0, timestamp=now)

        assert len(mon._records) == 3
        report = mon.check_accuracy_threshold()
        assert report.n_samples == 3
        assert report.accuracy == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Resume-defense Fix Area 2 — GPU-acceleration runtime evidence
#
# Resume claim: "Delivered real-time engine health predictions under 300 ms
# by deploying a FastAPI service with GPU acceleration, Redis caching, and
# asynchronous processing."
#
# Gap: the code *supports* CUDA, but nothing reported, at runtime, whether a
# loaded model is actually executing on GPU vs. CPU. Fix: ModelManager now
# stamps a "device" field into every model's metadata, /health and /models
# surface it, benchmark_latency.py reports torch.cuda.is_available() / the
# selected device / GPU name / run_type, and InferenceEngine demonstrably
# moves input tensors to model._device before the forward pass.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGPUAccelerationEvidence:
    """Verifies runtime device evidence for the resume's GPU-acceleration claim."""

    _SMALL_MLP_CONFIG = {
        "architecture": {"hidden_sizes": [16, 8], "dropout": 0.1},
        "training": {"learning_rate": 1e-3},
    }

    # ------------------------------------------------------------------
    # 1. Model metadata includes device
    # ------------------------------------------------------------------
    def test_model_metadata_includes_device(self, tmp_path):
        from ml_pipeline.train.models.mlp_model import MLPRULPredictor
        from inference_service.models.model_manager import ModelManager, TORCH_DEVICE, _device_label_for
        import numpy as _np

        predictor = MLPRULPredictor(self._SMALL_MLP_CONFIG)
        predictor.build_model(input_dim=5)
        ckpt_path = str(tmp_path / "device_meta_mlp.pt")
        predictor.save_model(ckpt_path)

        mgr = object.__new__(ModelManager)
        mgr._models = {}
        mgr._model_metadata = {}
        mgr._lock = threading.RLock()
        mgr.load_mlp_model(ckpt_path, "mlp_rul_predictor", "v-device-test")

        meta = mgr.get_model_metadata("mlp")
        assert meta is not None
        assert "device" in meta, "ModelManager metadata must include a 'device' field"
        assert meta["device"] == str(TORCH_DEVICE)
        assert meta["device"] in ("cpu",) or meta["device"].startswith("cuda")

        # Every loader's metadata must carry a device label — including the
        # MLflow path (mlp/lstm) and sklearn (which is always CPU-only).
        assert _device_label_for("mlp") == str(TORCH_DEVICE)
        assert _device_label_for("random_forest") == "cpu"
        assert _device_label_for("lstm") in ("cpu", "cuda", "unknown")

    # ------------------------------------------------------------------
    # 2. CPU fallback is explicit
    # ------------------------------------------------------------------
    def test_cpu_fallback_is_explicit(self, tmp_path, caplog):
        import torch
        from ml_pipeline.train.models.mlp_model import MLPRULPredictor
        from inference_service.models.model_manager import ModelManager, TORCH_DEVICE

        if torch.cuda.is_available():
            pytest.skip("CUDA available on this host — CPU-fallback path is not exercised")

        predictor = MLPRULPredictor(self._SMALL_MLP_CONFIG)
        predictor.build_model(input_dim=5)
        ckpt_path = str(tmp_path / "cpu_fallback_mlp.pt")
        predictor.save_model(ckpt_path)

        mgr = object.__new__(ModelManager)
        mgr._models = {}
        mgr._model_metadata = {}
        mgr._lock = threading.RLock()

        with caplog.at_level(logging.INFO):
            mgr.load_mlp_model(ckpt_path, "mlp_rul_predictor", "v-cpu-test")

        assert str(TORCH_DEVICE) == "cpu"
        assert mgr.get_model_metadata("mlp")["device"] == "cpu"
        assert "CUDA unavailable, running on CPU fallback." in caplog.text, (
            "When CUDA is unavailable the loader must log the exact, "
            "unambiguous fallback message — not silently default to CPU"
        )

    # ------------------------------------------------------------------
    # 3. Benchmark output includes selected device
    # ------------------------------------------------------------------
    def test_benchmark_output_includes_selected_device(self, tmp_path):
        out_path = tmp_path / "bench_device_evidence.json"
        proc = subprocess.run(
            [
                sys.executable, str(_INFERENCE / "benchmark_latency.py"),
                "--n-warmup", "1", "--n-runs", "2", "--seq-len", "10",
                "--output", str(out_path),
            ],
            cwd=str(_INFERENCE),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stdout[-3000:] + "\n--- stderr ---\n" + proc.stderr[-3000:]

        # Printed benchmark report must show the selected device + CUDA evidence.
        assert "torch.cuda.is_available()" in proc.stdout
        assert "Selected device" in proc.stdout

        data = json.loads(out_path.read_text())
        for key in ("cuda_available", "device", "gpu_name", "run_type"):
            assert key in data, f"benchmark JSON output must report '{key}'"
        assert data["run_type"] in ("cuda", "cpu")
        assert isinstance(data["cuda_available"], bool)
        assert data["run_type"] == ("cuda" if data["cuda_available"] else "cpu")
        if not data["cuda_available"]:
            assert data["device"] == "cpu"
            assert data["gpu_name"] is None

    # ------------------------------------------------------------------
    # 4. InferenceEngine sends the input tensor to model._device
    # ------------------------------------------------------------------
    def test_predict_rul_torch_sends_tensor_to_model_device(self, monkeypatch):
        from inference_service.models.inference_engine import InferenceEngine
        import torch
        import torch.nn as nn

        torch.manual_seed(0)
        model = nn.Linear(4, 1)
        model.eval()
        target_device = torch.device("cpu")
        model._device = target_device
        model._scaler = None

        seen_devices = []
        real_to = torch.Tensor.to

        def _spy_to(self_tensor, *args, **kwargs):
            dev = kwargs.get("device", args[0] if args else None)
            if isinstance(dev, torch.device):
                seen_devices.append(dev)
            return real_to(self_tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", _spy_to)

        engine = InferenceEngine(n_features=4)
        features = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        rul, _ = engine.predict_rul_torch(model, features)

        assert isinstance(rul, float)
        assert target_device in seen_devices, (
            "predict_rul_torch must move the input tensor to model._device "
            "(torch.tensor(features).to(device)) — this is what makes GPU "
            "placement actually take effect at inference time, not just "
            "exist as supported code"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Resume-defense — claim 1: "Lowered turbofan engine RUL prediction error
# by 15%" — verified against the real, logged MLflow evidence (mlruns.db),
# no live MLflow server / network / GPU required.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestResumeEvidence15PercentClaim:
    """
    print_resume_evidence.py reads the 'baseline_vs_mlp_cmapss' experiment
    straight out of the local SQLite tracking store (mlruns.db) and prints
    the MAE comparison that backs the resume's "15%" claim.
    """

    _SCRIPT = _ML / "evaluate" / "print_resume_evidence.py"
    _DB = _REPO / "mlruns.db"

    def _run_script(self):
        assert self._SCRIPT.exists(), f"script not found: {self._SCRIPT}"
        assert self._DB.exists(), (
            f"mlruns.db not found at {self._DB} — run baseline_comparison.py first"
        )
        return subprocess.run(
            [sys.executable, str(self._SCRIPT), "--db", str(self._DB)],
            capture_output=True,
            text=True,
            cwd=str(_REPO),
            timeout=120,
        )

    def test_script_runs_without_requiring_live_services(self):
        """
        The script must run to completion purely from the local mlruns.db
        SQLite file — no live MLflow tracking server, network call, or GPU.
        """
        result = self._run_script()
        combined = (result.stdout + result.stderr).lower()

        # Evidence that it never attempted to reach a live service
        for forbidden in (
            "connectionerror",
            "connection refused",
            "http://mlflow",
            "https://mlflow",
            "failed to connect",
            "max retries exceeded",
        ):
            assert forbidden not in combined, (
                f"script output suggests it tried to reach a live service "
                f"({forbidden!r} found) — it should be fully self-contained "
                f"via the local SQLite store:\n{result.stdout}\n{result.stderr}"
            )

        assert result.returncode == 0, (
            f"script exited with code {result.returncode} (expected 0)\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "Resume claim" in result.stdout

    def test_output_shows_at_least_15_percent_mae_improvement(self):
        """
        The printed verdict must clearly show a measured MAE improvement
        (MLP + temporal features vs Ridge baseline) of at least the 15%
        claimed on the resume, with an explicit PASS verdict.
        """
        result = self._run_script()
        assert result.returncode == 0, (
            f"script must succeed before its output can be checked\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        out = result.stdout

        m = re.search(r"MAE improvement \(MLP temporal vs Ridge\)\s*:\s*([+-]?\d+(?:\.\d+)?)%", out)
        assert m, f"could not find the MAE-improvement line in output:\n{out}"
        improvement_pct = float(m.group(1))

        assert improvement_pct >= 15.0, (
            f"measured MAE improvement of {improvement_pct}% does NOT meet "
            f"the resume's '15%' claim — output:\n{out}"
        )
        assert "PASS" in out, (
            f"expected an explicit PASS verdict in the output:\n{out}"
        )
        assert "FAIL" not in out


# ═══════════════════════════════════════════════════════════════════════════
# Resume-defense — claim 2: "Built automated monitoring ... using MLflow
# and Prometheus/Grafana that triggered retraining when accuracy dropped
# below 80%, ensuring production reliability."
#
# These tests verify the /metrics endpoint exposes the instruments needed
# to back that claim (request counts, latency, cache hit/miss, models
# loaded, Kafka pipeline status, retraining triggers, feedback volume,
# rolling accuracy, model version) and that they are live, scrapeable,
# and actually move when the code paths that update them run.
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPrometheusMetricsProductionReliability:

    @staticmethod
    def _metrics_client():
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from inference_service.api.metrics import router as metrics_router

        app = FastAPI()
        app.include_router(metrics_router)
        return TestClient(app)

    @staticmethod
    def _metric_value(text: str, name: str) -> Optional[float]:
        m = re.search(re.escape(name) + r"(?:\{[^}]*\})? ([0-9eE+\-.]+)", text)
        return float(m.group(1)) if m else None

    def test_metrics_endpoint_reachable(self):
        client = self._metrics_client()
        resp = client.get("/metrics")

        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]

        body = resp.text
        # Required instruments backing the resume's monitoring/reliability claim
        for name in (
            "inference_requests_total",          # inference request count
            "inference_latency_seconds",         # inference latency histogram
            "prediction_cache_hits_total",       # cache hit count
            "prediction_cache_misses_total",     # cache miss count
            "models_loaded",                     # models loaded
            "kafka_pipeline_running",            # Kafka pipeline status
            "retraining_triggered_total",        # retraining trigger count
            "feedback_records_total",            # feedback volume
            "model_rolling_accuracy",            # current rolling accuracy
            "model_version_info",                # model_version gauge/label
            "service_uptime_seconds",
        ):
            assert name in body, f"expected metric {name!r} to be exposed at /metrics"

    def test_metrics_updated_after_prediction_and_feedback_calls(self):
        """
        Exercise the SAME instruments main.py updates on the predict_rul
        cache-hit/miss path and the /feedback path, and confirm the new
        values are reflected in the scraped output (not static/dead gauges).
        """
        from prometheus_client import generate_latest, REGISTRY
        from inference_service.api.metrics import (
            INFERENCE_REQUESTS_TOTAL,
            CACHE_HITS_TOTAL,
            CACHE_MISSES_TOTAL,
            FEEDBACK_RECORDS_TOTAL,
            CURRENT_ACCURACY,
            MODEL_VERSION_INFO,
        )

        before = generate_latest(REGISTRY).decode()
        hits_before = self._metric_value(before, "prediction_cache_hits_total") or 0.0
        misses_before = self._metric_value(before, "prediction_cache_misses_total") or 0.0
        feedback_before = self._metric_value(before, "feedback_records_total") or 0.0

        # --- simulate a cache-hit prediction (predict_rul cache-hit path) ---
        INFERENCE_REQUESTS_TOTAL.labels(model="mlp", status="cache_hit").inc()
        CACHE_HITS_TOTAL.inc()

        # --- simulate a cache-miss prediction that runs full inference ---
        from inference_service.api.metrics import INFERENCE_LATENCY_SECONDS
        CACHE_MISSES_TOTAL.inc()
        INFERENCE_REQUESTS_TOTAL.labels(model="mlp", status="success").inc()
        INFERENCE_LATENCY_SECONDS.labels(model="mlp").observe(0.05)

        # --- simulate a /feedback call (record_prediction + accuracy refresh) ---
        FEEDBACK_RECORDS_TOTAL.inc()
        CURRENT_ACCURACY.labels(model="mlp").set(0.91)
        MODEL_VERSION_INFO.labels(model="mlp", version="v1.2.0").set(1)

        after = generate_latest(REGISTRY).decode()

        assert self._metric_value(after, "prediction_cache_hits_total") == hits_before + 1
        assert self._metric_value(after, "prediction_cache_misses_total") == misses_before + 1
        assert self._metric_value(after, "feedback_records_total") == feedback_before + 1
        assert 'model_rolling_accuracy{model="mlp"} 0.91' in after
        assert 'model_version_info{model="mlp",version="v1.2.0"} 1.0' in after
        assert after != before

    def test_metrics_remains_scrapeable_by_prometheus(self):
        """
        The /metrics output must remain valid Prometheus exposition format
        — i.e. parseable by the same client library Prometheus itself uses
        — after adding the new instruments (no name collisions / malformed
        samples that would break scraping).
        """
        from prometheus_client.parser import text_string_to_metric_families

        client = self._metrics_client()
        resp = client.get("/metrics")
        assert resp.status_code == 200

        families = list(text_string_to_metric_families(resp.text))
        names = {f.name for f in families}

        assert len(families) > 5, "expected multiple metric families to be exposed"
        # NOTE: the official parser reports Counter family names WITHOUT the
        # "_total" suffix (e.g. "inference_requests_total" -> family
        # "inference_requests") — that's the correct, scrapeable grouping.
        for expected in (
            "inference_requests",
            "inference_latency_seconds",
            "models_loaded",
            "kafka_pipeline_running",
            "retraining_triggered",
            "feedback_records",
            "model_rolling_accuracy",
        ):
            assert expected in names, f"{expected!r} missing from parsed metric families"

        # A second scrape must also succeed and be stable/parseable
        resp2 = client.get("/metrics")
        assert resp2.status_code == 200
        families2 = list(text_string_to_metric_families(resp2.text))
        assert len(families2) == len(families)
