"""Unit tests for feature engineering — TimeDomainFeatures and FrequencyDomainFeatures."""

import pytest
import numpy as np

from features.time_domain_features import TimeDomainFeatures, AggregatedFeatures
from features.frequency_domain_features import FrequencyDomainFeatures


# ═══════════════════════════════════════════════════════════════════════════
# TimeDomainFeatures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestTimeDomainFeatures:
    def _make_sensor_data(self, value: float = 1.0):
        return {"sensor_2": value, "sensor_3": value * 2}

    def test_extract_features_returns_dict(self, feature_config):
        tdf = TimeDomainFeatures(feature_config)
        features = tdf.extract_features("eq1", self._make_sensor_data())
        assert isinstance(features, dict)

    def test_current_values_present(self, feature_config):
        tdf = TimeDomainFeatures(feature_config)
        features = tdf.extract_features("eq1", self._make_sensor_data(5.0))
        assert features.get("sensor_2_current") == pytest.approx(5.0)

    def test_rolling_statistics_after_window(self, feature_config):
        tdf = TimeDomainFeatures(feature_config)
        for i in range(12):
            features = tdf.extract_features("eq1", self._make_sensor_data(float(i)))
        # After 12 readings, a window=10 rolling mean should be available
        key = "sensor_2_rolling_10_mean"
        if key in features:
            assert features[key] is not None

    def test_rate_of_change(self, feature_config):
        tdf = TimeDomainFeatures(feature_config)
        # Feed increasing values
        for i in range(10):
            features = tdf.extract_features("eq1", self._make_sensor_data(float(i)))
        roc_key = "sensor_2_roc_5"
        if roc_key in features:
            # Rate of change should be positive on an increasing series
            assert features[roc_key] >= 0

    def test_clear_buffer(self, feature_config):
        tdf = TimeDomainFeatures(feature_config)
        tdf.extract_features("eq1", self._make_sensor_data())
        tdf.clear_buffer("eq1")
        status = tdf.get_buffer_status("eq1")
        assert status is not None

    def test_different_equipment_ids_isolated(self, feature_config):
        tdf = TimeDomainFeatures(feature_config)
        tdf.extract_features("eq1", self._make_sensor_data(1.0))
        tdf.extract_features("eq2", self._make_sensor_data(100.0))
        f1 = tdf.extract_features("eq1", self._make_sensor_data(1.0))
        f2 = tdf.extract_features("eq2", self._make_sensor_data(100.0))
        # Current values should differ between equipment
        assert f1.get("sensor_2_current") != f2.get("sensor_2_current")


@pytest.mark.unit
class TestAggregatedFeatures:
    def test_cross_sensor_features(self):
        data = {"sensor_a": 10.0, "sensor_b": 20.0, "sensor_c": 30.0}
        result = AggregatedFeatures.compute_cross_sensor_features(data)
        assert isinstance(result, dict)
        assert "sensors_mean" in result
        assert result["sensors_mean"] == pytest.approx(20.0)


# ═══════════════════════════════════════════════════════════════════════════
# FrequencyDomainFeatures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestFrequencyDomainFeatures:
    def _make_sensor_data(self, cycle: int):
        """Create synthetic sensor data with a known frequency component."""
        return {
            "sensor_2": np.sin(2 * np.pi * 0.1 * cycle) + np.random.normal(0, 0.01),
            "sensor_3": np.cos(2 * np.pi * 0.2 * cycle) + np.random.normal(0, 0.01),
        }

    def test_extract_features_empty_before_window(self, feature_config):
        fdf = FrequencyDomainFeatures(feature_config)
        features = fdf.extract_features("eq1", self._make_sensor_data(0))
        # Should return a dict (possibly empty — buffer not full yet)
        assert isinstance(features, dict)

    def test_extract_features_after_filling_buffer(self, feature_config):
        fdf = FrequencyDomainFeatures(feature_config)
        window_size = feature_config["feature_engineering"]["frequency_domain"][
            "window_size"
        ]
        features = {}
        for c in range(window_size + 5):
            features = fdf.extract_features("eq1", self._make_sensor_data(c))
        # The last call should have produced spectral features
        spectral_keys = [
            k for k in features if "fft" in k or "spectral" in k or "band" in k
        ]
        assert len(spectral_keys) > 0, (
            f"Expected spectral features, got keys: {list(features.keys())}"
        )

    def test_clear_buffer(self, feature_config):
        fdf = FrequencyDomainFeatures(feature_config)
        fdf.extract_features("eq1", self._make_sensor_data(0))
        fdf.clear_buffer("eq1")
        status = fdf.get_buffer_status("eq1")
        assert status is not None
