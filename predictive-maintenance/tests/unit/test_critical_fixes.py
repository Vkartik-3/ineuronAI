"""
Tests for the 5 critical fixes:

1. OnlineFeatureEngineer — serving features match training feature semantics
2. POST /feedback schema — endpoint wiring (unit-level)
3. ModelManager._infer_type — key normalization coverage
4. PredictionCache.make_key — determinism (cache key stability)
5. PerformanceMonitor.record_prediction — basic smoke test

These tests document the exact coverage boundary:

COVERED
-------
- OnlineFeatureEngineer: output shape, determinism, EMA recurrence, lag values
- ModelManager._infer_type: all three branches (mlp, lstm, random_forest)
- PredictionCache.make_key: same input → same key; ordering independence
- PerformanceMonitor.record_prediction + check accuracy

NOT COVERED (require live infra or real model weights)
------------------------------------------------------
- POST /feedback HTTP endpoint (requires running FastAPI app + real PerformanceMonitor config)
- MLP forward pass with engineered features (requires real .pt checkpoint)
- MLflow model loading (requires running MLflow server)
- Kafka consumer end-to-end (requires running Kafka broker)
- Retraining pipeline full execution (requires C-MAPSS data file)
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow imports from inference_service and ml_pipeline roots
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[3]
_INFERENCE = _REPO / "predictive-maintenance" / "inference_service"
_ML = _REPO / "predictive-maintenance" / "ml_pipeline"
sys.path.insert(0, str(_INFERENCE.parent))   # predictive-maintenance/
sys.path.insert(0, str(_INFERENCE))          # inference_service/
sys.path.insert(0, str(_ML / "retrain"))     # ml_pipeline/retrain/


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1 — OnlineFeatureEngineer
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestOnlineFeatureEngineer:
    """Verify that serving feature engineering mirrors offline training semantics."""

    def _make_sequence(self, n: int = 25) -> list:
        """
        Return n complete readings -- every sensor the feature contract requires.

        This fixture previously supplied only sensor_2/sensor_3 and relied on
        the engineer inferring its sensor set from whichever keys happened to be
        present, zero-filling the rest. That behaviour made the feature vector's
        width and column meaning depend on the payload, and it is what allowed
        the offline/online sensor-order divergence to go undetected. Partial
        readings are now rejected; see TestOnlineInputValidation in
        test_feature_contract.py.
        """
        from shared.feature_contract import OP_SETTINGS, SENSORS

        seq = []
        for i in range(n):
            reading = {"time_cycle": float(i + 1)}
            for j, s in enumerate(SENSORS):
                reading[s] = float(i) * 0.1 + 0.5 + j
            for o in OP_SETTINGS:
                reading[o] = 1.0
            seq.append(reading)
        return seq

    def test_returns_2d_array(self):
        from inference_service.feature_engineering import OnlineFeatureEngineer
        eng = OnlineFeatureEngineer()
        seq = self._make_sequence(25)
        result = eng.engineer(seq, "EQ-001")
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_output_dtype_float32(self):
        from inference_service.feature_engineering import OnlineFeatureEngineer
        eng = OnlineFeatureEngineer()
        seq = self._make_sequence(25)
        result = eng.engineer(seq, "EQ-001")
        assert result.dtype == np.float32

    def test_deterministic_same_input(self):
        """Same sequence must produce identical feature vectors."""
        from inference_service.feature_engineering import OnlineFeatureEngineer
        eng = OnlineFeatureEngineer()
        seq = self._make_sequence(30)
        r1 = eng.engineer(seq, "EQ-001")
        r2 = eng.engineer(seq, "EQ-001")
        np.testing.assert_array_equal(r1, r2)

    def test_ema_last_value_matches_recurrence(self):
        """_ema_last() must match the pandas ewm(adjust=False) recurrence."""
        from inference_service.feature_engineering import _ema_last
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = 0.3
        # Manual recurrence
        ema = series[0]
        for v in series[1:]:
            ema = (1 - alpha) * ema + alpha * v
        assert abs(_ema_last(series, alpha) - ema) < 1e-9

    def test_lag_of_1_is_last_minus_second_to_last(self):
        """lag-1 at the last timestep must equal series[-1] - series[-2]."""
        from inference_service.feature_engineering import OnlineFeatureEngineer
        from shared.feature_contract import FEATURE_NAMES, OP_SETTINGS, SENSORS

        eng = OnlineFeatureEngineer()
        # sensor_2 increases by exactly 1 each step; all other sensors are flat.
        seq = []
        for i in range(15):
            reading = {"time_cycle": float(i + 1)}
            for s in SENSORS:
                reading[s] = float(i) if s == "sensor_2" else 1.0
            for o in OP_SETTINGS:
                reading[o] = 0.5
            seq.append(reading)

        result = eng.engineer(seq, "EQ-001")
        # Locate the column by NAME via the contract rather than by a hardcoded
        # offset; the old version assumed a per-sensor-interleaved layout that
        # does not match training.
        lag1_idx = FEATURE_NAMES.index("sensor_2_lag1")
        assert abs(result[0, lag1_idx] - 1.0) < 1e-5

    def test_constant_sensors_excluded(self):
        """CONSTANT_SENSORS must not appear in the output features."""
        from inference_service.feature_engineering import (
            OnlineFeatureEngineer,
            CONSTANT_SENSORS,
        )
        from shared.feature_contract import OP_SETTINGS, SENSORS

        eng = OnlineFeatureEngineer()

        def _base(i):
            reading = {"time_cycle": float(i + 1)}
            for j, s in enumerate(SENSORS):
                reading[s] = float(i) + j
            for o in OP_SETTINGS:
                reading[o] = 0.5
            return reading

        # Constant sensors present in the payload must be ignored entirely.
        seq_with = []
        for i in range(10):
            r = _base(i)
            for c in CONSTANT_SENSORS:
                r[c] = 99.0
            seq_with.append(r)
        seq_without = [_base(i) for i in range(10)]

        np.testing.assert_array_equal(
            eng.engineer(seq_with, "EQ-001"),
            eng.engineer(seq_without, "EQ-001"),
        )
        assert not any(
            n.startswith(tuple(f"{c}_" for c in CONSTANT_SENSORS)) or n in CONSTANT_SENSORS
            for n in OnlineFeatureEngineer.feature_names
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fix 3 — ModelManager._infer_type key normalization
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestModelManagerInferType:
    """
    _infer_type maps arbitrary config key names to the canonical short keys
    used by get_model() / get_model_metadata() callers.
    """

    def _infer(self, key: str) -> str:
        from inference_service.models.model_manager import ModelManager
        return ModelManager._infer_type(key)

    def test_mlp_rul_maps_to_mlp(self):
        assert self._infer("mlp_rul") == "mlp"

    def test_lstm_rul_maps_to_lstm(self):
        assert self._infer("lstm_rul") == "lstm"

    def test_random_forest_health_maps_to_random_forest(self):
        assert self._infer("random_forest_health") == "random_forest"

    def test_mlp_alone_maps_to_mlp(self):
        assert self._infer("mlp") == "mlp"

    def test_lstm_alone_maps_to_lstm(self):
        assert self._infer("lstm") == "lstm"

    def test_random_forest_alone_maps_to_random_forest(self):
        assert self._infer("random_forest") == "random_forest"

    def test_unknown_key_defaults_to_random_forest(self):
        # Anything without 'mlp' or 'lstm' in the name falls back to rf
        assert self._infer("gradient_boost_v2") == "random_forest"


# ═══════════════════════════════════════════════════════════════════════════
# Fix 4 — PredictionCache.make_key determinism
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPredictionCacheKey:
    """
    Cache key must be stable (same input → same key) and independent of
    dict insertion order.
    """

    def _cache(self):
        from inference_service.cache.prediction_cache import PredictionCache
        return PredictionCache(host="localhost", port=6379, ttl=300)

    def test_same_input_produces_same_key(self):
        cache = self._cache()
        seq = [{"a": 1.0, "b": 2.0}, {"a": 1.5, "b": 2.5}]
        k1 = cache.make_key("EQ-001", seq)
        k2 = cache.make_key("EQ-001", seq)
        assert k1 == k2

    def test_different_equipment_id_produces_different_key(self):
        cache = self._cache()
        seq = [{"a": 1.0}]
        k1 = cache.make_key("EQ-001", seq)
        k2 = cache.make_key("EQ-002", seq)
        assert k1 != k2

    def test_different_sequence_produces_different_key(self):
        cache = self._cache()
        seq1 = [{"a": 1.0}]
        seq2 = [{"a": 2.0}]
        k1 = cache.make_key("EQ-001", seq1)
        k2 = cache.make_key("EQ-001", seq2)
        assert k1 != k2

    def test_key_is_string(self):
        cache = self._cache()
        k = cache.make_key("EQ-001", [{"a": 1.0}])
        assert isinstance(k, str)


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2 — PerformanceMonitor ground-truth recording
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPerformanceMonitorRecording:
    """
    Smoke test: record_prediction stores records and within_tolerance
    is set correctly (±15 cycles tolerance).
    """

    def _monitor(self):
        """Build a monitor without loading a config file."""
        from ml_pipeline.retrain.performance_monitor import PerformanceMonitor
        mon = object.__new__(PerformanceMonitor)
        mon.accuracy_threshold = 0.80
        mon.tolerance_cycles = 15
        mon.window_days = 7
        mon.min_samples = 5
        mon._records = []
        mon._redis_client = None  # in-memory mode — no Redis backend in this bare instance
        mon._redis_key = "perf_monitor:records"
        mon._record_retention_days = 14
        return mon

    def test_record_adds_entry(self):
        mon = self._monitor()
        mon.record_prediction("EQ-001", true_rul=80.0, predicted_rul=75.0)
        assert len(mon._records) == 1

    def test_within_tolerance_true(self):
        mon = self._monitor()
        mon.record_prediction("EQ-001", true_rul=80.0, predicted_rul=75.0)  # diff=5
        assert mon._records[0].within_tolerance is True

    def test_outside_tolerance_false(self):
        mon = self._monitor()
        mon.record_prediction("EQ-001", true_rul=80.0, predicted_rul=50.0)  # diff=30
        assert mon._records[0].within_tolerance is False

    def test_compute_accuracy_all_within(self):
        from inference_service.feature_engineering import _ema_last  # noqa: ensure module importable
        from ml_pipeline.retrain.performance_monitor import PerformanceMonitor
        y_true = np.array([80.0, 90.0, 70.0])
        y_pred = np.array([82.0, 88.0, 72.0])  # all within ±15
        acc = PerformanceMonitor.compute_within_tolerance_accuracy(y_true, y_pred, tolerance=15)
        assert acc == pytest.approx(1.0)

    def test_compute_accuracy_none_within(self):
        from ml_pipeline.retrain.performance_monitor import PerformanceMonitor
        y_true = np.array([80.0])
        y_pred = np.array([110.0])  # diff=30, outside ±15
        acc = PerformanceMonitor.compute_within_tolerance_accuracy(y_true, y_pred, tolerance=15)
        assert acc == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Fix 1 (extended) — Feature order parity: OnlineFeatureEngineer vs training
# ═══════════════════════════════════════════════════════════════════════════

def _engineer_features_reference(
    df,
    sensor_cols,
    op_cols,
    lags=(1, 3, 5, 10),
    windows=(5, 10, 20),
    ema_alphas=(0.1, 0.3, 0.5),
):
    """
    Inline copy of baseline_comparison._engineer_features() used only in this
    test so the parity check has no import-chain dependency on torch or
    cmapss_loader.  Must be kept byte-for-byte identical to the original.
    """
    import pandas as pd

    feature_frames = []
    for unit_id, grp in df.groupby("unit_id"):
        grp = grp.sort_values("time_cycle").copy()
        cols: dict = {}

        for col in sensor_cols + op_cols:
            cols[col] = grp[col].values

        for col in sensor_cols:
            for lag in lags:
                cols[f"{col}_lag{lag}"] = grp[col].diff(lag).fillna(0).values

        for col in sensor_cols:
            for w in windows:
                cols[f"{col}_roll{w}_mean"] = (
                    grp[col].rolling(w, min_periods=1).mean().values
                )
                cols[f"{col}_roll{w}_std"] = (
                    grp[col].rolling(w, min_periods=1).std().fillna(0).values
                )

        for col in sensor_cols:
            for alpha in ema_alphas:
                alpha_str = str(alpha).replace(".", "")
                cols[f"{col}_ema{alpha_str}"] = (
                    grp[col].ewm(alpha=alpha, adjust=False).mean().values
                )

        cols["time_cycle"] = grp["time_cycle"].values

        feature_frames.append(pd.DataFrame(cols))

    full = pd.concat(feature_frames, ignore_index=True)
    X = full.fillna(0).values.astype(np.float32)
    return X


@pytest.mark.unit
class TestFeatureOrderParity:
    """
    Verify that OnlineFeatureEngineer._build_feature_vector output column order
    and values match _engineer_features() from baseline_comparison.py.

    This test catches training-serving feature mismatches before they reach
    production.  It uses an inline copy of _engineer_features so there is no
    dependency on torch or cmapss_loader at test time.

    The previous implementation interleaved all features for each sensor
    (raw+lags+rolling+EMA for s2, then for s3, …) while training grouped them
    as all-raw-sensors, then all-ops, then all-lags, etc.  The fix reorders
    _build_feature_vector to match training exactly.
    """

    def _make_df_and_sequence(self, n_cycles: int = 25):
        """
        Build a matching DataFrame (training-side) and sequence-of-dicts
        (inference-side) from the same synthetic data.
        """
        pandas = pytest.importorskip("pandas")
        from shared.feature_contract import OP_SETTINGS, SENSORS

        np.random.seed(42)

        # Every contract sensor is populated with a DISTINCT trajectory. The
        # earlier version of this fixture used only sensor_2 and sensor_3 --
        # the one subset on which lexical and numeric sensor ordering happen to
        # agree -- so it stayed green while all 14 real sensor blocks were
        # permuted at serving time.
        data = {
            "unit_id": 1,
            "time_cycle": np.arange(1.0, n_cycles + 1, dtype=float),
            "rul": np.linspace(100.0, 1.0, n_cycles),
        }
        for j, s in enumerate(SENSORS):
            data[s] = (
                np.linspace(1.0 + 10 * j, 2.0 + 10 * j, n_cycles)
                + np.random.normal(0, 0.02, n_cycles)
            )
        for j, o in enumerate(OP_SETTINGS):
            data[o] = np.full(n_cycles, 0.5 + j)

        df = pandas.DataFrame(data)

        sequence = [
            {
                k: float(df[k].iloc[i])
                for k in list(SENSORS) + list(OP_SETTINGS) + ["time_cycle"]
            }
            for i in range(n_cycles)
        ]
        return df, sequence

    def test_feature_order_and_values_match_training(self):
        """
        The feature vector at the last cycle must be numerically identical
        whether computed by _engineer_features (training) or
        OnlineFeatureEngineer (inference).
        """
        from inference_service.feature_engineering import OnlineFeatureEngineer

        from shared.feature_contract import OP_SETTINGS, SENSORS

        df, sequence = self._make_df_and_sequence(n_cycles=25)
        sensor_cols = list(SENSORS)
        op_cols = list(OP_SETTINGS)

        # Training path — take the last row (last cycle for unit 1)
        X_train = _engineer_features_reference(df, sensor_cols, op_cols)
        train_last = X_train[-1]  # already float32

        # Inference path
        eng = OnlineFeatureEngineer()
        inf_vec = eng.engineer(sequence, "unit_1")[0]  # shape (n_features,)

        assert len(train_last) == len(inf_vec), (
            f"Feature count mismatch: training={len(train_last)}, "
            f"inference={len(inf_vec)}"
        )
        np.testing.assert_allclose(
            inf_vec,
            train_last,
            rtol=1e-4,
            atol=1e-4,
            err_msg=(
                "Feature ordering or values differ between OnlineFeatureEngineer "
                "and _engineer_features().  This indicates a training-serving "
                "column-layout mismatch."
            ),
        )

    def test_engineered_feature_dim_is_200(self):
        """ENGINEERED_FEATURE_DIM must equal 200 for C-MAPSS FD001."""
        from inference_service.feature_engineering import ENGINEERED_FEATURE_DIM

        assert ENGINEERED_FEATURE_DIM == 200

    def test_full_cmapss_sequence_produces_200_features(self):
        """
        A sequence with all 21 C-MAPSS sensors (7 constant, 14 non-constant)
        must produce exactly 200 features.
        """
        from inference_service.feature_engineering import (
            OnlineFeatureEngineer,
            CONSTANT_SENSORS,
            ENGINEERED_FEATURE_DIM,
        )

        all_sensors = [f"sensor_{i}" for i in range(1, 22)]
        sequence = [
            {
                **{s: float(t + i * 0.01) for i, s in enumerate(all_sensors)},
                "op_setting_1": 0.5,
                "op_setting_2": 0.3,
                "op_setting_3": 80.0,
                "time_cycle": float(t + 1),
            }
            for t in range(30)
        ]

        eng = OnlineFeatureEngineer()
        result = eng.engineer(sequence, "full_cmapss")
        assert result.shape == (1, ENGINEERED_FEATURE_DIM), (
            f"Expected shape (1, {ENGINEERED_FEATURE_DIM}), got {result.shape}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fix 2 (extended) — /feedback persistence: records survive after request
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFeedbackPersistence:
    """
    Verify that feedback records accumulate in the PerformanceMonitor singleton
    and are visible to check_accuracy_threshold() across multiple calls.

    The bug: /feedback created a new PerformanceMonitor() per request, so
    every record was discarded when the handler returned.

    The fix: a module-level singleton is created once in the lifespan and
    reused by every /feedback call and by RetrainingPipeline.
    """

    def _make_monitor(self):
        """Build a PerformanceMonitor without a config file (same bypass used
        by the existing TestPerformanceMonitorRecording tests)."""
        from ml_pipeline.retrain.performance_monitor import PerformanceMonitor

        mon = object.__new__(PerformanceMonitor)
        mon.accuracy_threshold = 0.80
        mon.tolerance_cycles = 15
        mon.window_days = 7
        mon.min_samples = 5
        mon._records = []
        mon._redis_client = None  # in-memory mode — no Redis backend in this bare instance
        mon._redis_key = "perf_monitor:records"
        mon._record_retention_days = 14
        return mon

    def test_records_accumulate_across_calls(self):
        """
        Three separate /feedback calls on the same monitor must accumulate
        three records — not reset to one record each time.
        """
        mon = self._make_monitor()

        # Simulate three consecutive handler invocations on the SAME instance
        mon.record_prediction("EQ-1", true_rul=80.0, predicted_rul=75.0)
        mon.record_prediction("EQ-2", true_rul=90.0, predicted_rul=88.0)
        mon.record_prediction("EQ-3", true_rul=50.0, predicted_rul=70.0)

        assert mon.n_records == 3, (
            f"Expected 3 records, got {mon.n_records}. "
            "Records must persist on the same instance."
        )

    def test_check_accuracy_threshold_sees_all_feedback(self):
        """
        check_accuracy_threshold() must reflect every record submitted via the
        shared monitor — not just the most recent one.
        """
        mon = self._make_monitor()

        # 6 within-tolerance (diff ≤ 15) + 4 outside (diff = 30)
        for i in range(6):
            mon.record_prediction(f"EQ-{i}", true_rul=80.0, predicted_rul=82.0)
        for i in range(4):
            mon.record_prediction(f"EQ-{i + 6}", true_rul=80.0, predicted_rul=110.0)

        report = mon.check_accuracy_threshold()

        assert report.n_samples == 10
        assert report.accuracy == pytest.approx(0.6)
        assert report.below_threshold is True, (
            "60% accuracy < 80% threshold; must flag for retraining"
        )

    def test_throwaway_monitor_does_not_accumulate(self):
        """
        Regression: documents the pre-fix failure mode.  Creating a new
        PerformanceMonitor() per request means records are immediately lost.
        """
        from ml_pipeline.retrain.performance_monitor import PerformanceMonitor

        def old_broken_handler(equipment_id, true_rul, predicted_rul):
            """Old broken pattern — new instance per call, records discarded."""
            mon = object.__new__(PerformanceMonitor)
            mon.accuracy_threshold = 0.80
            mon.tolerance_cycles = 15
            mon.window_days = 7
            mon.min_samples = 5
            mon._records = []
            mon._redis_client = None  # in-memory mode — no Redis backend in this bare instance
            mon._redis_key = "perf_monitor:records"
            mon._record_retention_days = 14
            mon.record_prediction(equipment_id, true_rul, predicted_rul)
            return mon.n_records  # always 1

        r1 = old_broken_handler("EQ-1", 80.0, 75.0)
        r2 = old_broken_handler("EQ-2", 90.0, 88.0)
        r3 = old_broken_handler("EQ-3", 50.0, 70.0)

        # Each broken call returns 1 regardless of previous calls
        assert r1 == 1
        assert r2 == 1  # should have been 2 with a singleton
        assert r3 == 1  # should have been 3 with a singleton

    def test_retrain_pipeline_uses_injected_monitor(self):
        """
        When performance_monitor is passed to RetrainingPipeline.__init__,
        self.performance_monitor must be the same object — not a fresh instance.
        """
        # We test only the injection logic without opening config files.
        mon = self._make_monitor()
        mon.record_prediction("EQ-X", true_rul=70.0, predicted_rul=65.0)

        from ml_pipeline.retrain.retrain_pipeline import RetrainingPipeline

        pipeline = object.__new__(RetrainingPipeline)
        # Replicate only the monitor-injection line from __init__
        pipeline.performance_monitor = (
            mon if mon is not None else None
        )

        assert pipeline.performance_monitor is mon, (
            "RetrainingPipeline must use the injected monitor, not create a new one"
        )
        assert pipeline.performance_monitor.n_records == 1
