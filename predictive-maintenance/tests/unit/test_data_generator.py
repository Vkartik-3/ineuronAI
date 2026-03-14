"""Unit tests for data_generator — DegradationEngine and SensorSimulator."""

import pytest
import numpy as np

from simulator.degradation_engine import (
    DegradationEngine,
    DegradationPattern,
    FailureMode,
    LinearDegradation,
    ExponentialDegradation,
    StepDegradation,
    OscillatingDegradation,
)
from simulator.sensor_simulator import SensorSimulator, SensorDataGenerator


# ═══════════════════════════════════════════════════════════════════════════
# DegradationEngine tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestLinearDegradation:
    def test_returns_zero_before_start(self):
        val = LinearDegradation.calculate(
            cycle=0, start_cycle=10, rate=0.01, rul_remaining=200, total_rul=200
        )
        assert val == 0.0

    def test_increases_over_time(self):
        vals = [
            LinearDegradation.calculate(
                cycle=c, start_cycle=0, rate=0.01, rul_remaining=200 - c, total_rul=200
            )
            for c in range(0, 201, 50)
        ]
        # Should be monotonically non-decreasing
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    def test_clipped_to_one(self):
        val = LinearDegradation.calculate(
            cycle=10000, start_cycle=0, rate=0.01, rul_remaining=0, total_rul=200
        )
        assert val <= 1.0


@pytest.mark.unit
class TestExponentialDegradation:
    def test_starts_near_zero(self):
        val = ExponentialDegradation.calculate(
            cycle=1, start_cycle=0, growth_rate=0.5, rul_remaining=200, total_rul=200
        )
        assert val < 0.1

    def test_grows_faster_than_linear(self):
        exp_vals = [
            ExponentialDegradation.calculate(
                cycle=c,
                start_cycle=0,
                growth_rate=0.5,
                rul_remaining=200 - c,
                total_rul=200,
            )
            for c in [50, 100, 150, 200]
        ]
        lin_vals = [
            LinearDegradation.calculate(
                cycle=c,
                start_cycle=0,
                rate=0.01,
                rul_remaining=200 - c,
                total_rul=200,
            )
            for c in [50, 100, 150, 200]
        ]
        # Exponential should exceed linear at some point
        assert any(e > l for e, l in zip(exp_vals, lin_vals))


@pytest.mark.unit
class TestStepDegradation:
    def test_zero_before_occurrence(self):
        val = StepDegradation.calculate(cycle=5, occurrence_cycle=10, magnitude=0.6)
        assert val == 0.0

    def test_magnitude_after_occurrence(self):
        val = StepDegradation.calculate(cycle=15, occurrence_cycle=10, magnitude=0.6)
        assert val == pytest.approx(0.6)


@pytest.mark.unit
class TestDegradationEngine:
    def test_create_failure_mode(self, degradation_config):
        engine = DegradationEngine(degradation_config)
        fm = engine.create_failure_mode("turbofan", "fan_degradation", current_cycle=0)
        assert fm is not None
        assert fm.equipment_type == "turbofan"
        assert fm.name == "fan_degradation"
        assert fm.rul_remaining > 0

    def test_create_failure_mode_invalid_type(self, degradation_config):
        engine = DegradationEngine(degradation_config)
        fm = engine.create_failure_mode("nonexistent", "fan_degradation")
        assert fm is None

    def test_calculate_degradation_factors(self, degradation_config):
        engine = DegradationEngine(degradation_config)
        fm = engine.create_failure_mode("turbofan", "fan_degradation", current_cycle=0)
        factors = engine.calculate_degradation_factors(fm, current_cycle=50)
        assert isinstance(factors, dict)
        for sensor in fm.primary_sensors:
            assert sensor in factors
            assert 0.0 <= factors[sensor] <= 1.0

    def test_update_rul_decrements(self, degradation_config):
        engine = DegradationEngine(degradation_config)
        fm = engine.create_failure_mode("turbofan", "fan_degradation", current_cycle=0)
        original_rul = fm.rul_remaining
        fm = engine.update_rul(fm)
        assert fm.rul_remaining == original_rul - 1

    def test_is_failed(self, degradation_config):
        engine = DegradationEngine(degradation_config)
        fm = engine.create_failure_mode("turbofan", "fan_degradation", current_cycle=0)
        fm.rul_remaining = 0
        assert engine.is_failed(fm) is True

    def test_failure_mode_stages(self, degradation_config):
        engine = DegradationEngine(degradation_config)
        fm = engine.create_failure_mode("turbofan", "fan_degradation", current_cycle=0)
        total = fm.total_rul

        fm.rul_remaining = int(total * 0.9)
        assert fm.get_degradation_stage() == "early"

        fm.rul_remaining = int(total * 0.5)
        assert fm.get_degradation_stage() == "middle"

        fm.rul_remaining = int(total * 0.2)
        assert fm.get_degradation_stage() == "late"

        fm.rul_remaining = int(total * 0.05)
        assert fm.get_degradation_stage() == "critical"


# ═══════════════════════════════════════════════════════════════════════════
# SensorSimulator tests
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSensorSimulator:
    def test_generate_reading_within_bounds(self, sensor_config):
        sim = SensorSimulator("turbofan", sensor_config)
        for _ in range(50):
            val = sim.generate_reading("temperature", cycle=10)
            assert sensor_config["sensor_baseline"]["temperature"]["min"] <= val
            assert val <= sensor_config["sensor_baseline"]["temperature"]["max"]

    def test_degradation_shifts_value(self, sensor_config):
        sim = SensorSimulator("turbofan", sensor_config)
        np.random.seed(42)
        baseline = np.mean(
            [
                sim.generate_reading("temperature", cycle=10, degradation_factor=0.0)
                for _ in range(200)
            ]
        )
        np.random.seed(42)
        degraded = np.mean(
            [
                sim.generate_reading("temperature", cycle=10, degradation_factor=0.8)
                for _ in range(200)
            ]
        )
        # With degradation the value should shift
        assert abs(degraded - baseline) > 0.5

    def test_generate_all_sensors(self, sensor_config):
        sim = SensorSimulator("turbofan", sensor_config)
        readings = sim.generate_all_sensors(cycle=1)
        assert isinstance(readings, dict)
        for sensor_name in sensor_config["sensor_baseline"]:
            assert sensor_name in readings

    def test_noise_can_be_disabled(self, sensor_config):
        sim = SensorSimulator("turbofan", sensor_config)
        sim.set_noise_level(enabled=False)
        vals = [sim.generate_reading("temperature", cycle=1) for _ in range(10)]
        # With noise disabled values should be very close together
        assert np.std(vals) < 1.0


@pytest.mark.unit
class TestSensorDataGenerator:
    def test_generate_returns_dict(self, equipment_config):
        gen = SensorDataGenerator("turbofan", equipment_config)
        result = gen.generate("engine_001", cycle=5)
        assert isinstance(result, dict)
        assert "equipment_id" in result
        assert result["equipment_id"] == "engine_001"
        assert "sensor_readings" in result or "readings" in result or len(result) > 2
