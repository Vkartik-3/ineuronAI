"""
Shared pytest fixtures for the Predictive Maintenance test suite.
"""

import sys
import os

import pytest

# ---------------------------------------------------------------------------
# Ensure module directories are importable from tests
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_DIRS = [
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "data_generator"),
    os.path.join(PROJECT_ROOT, "stream_processor"),
    os.path.join(PROJECT_ROOT, "feature_store"),
    os.path.join(PROJECT_ROOT, "inference_service"),
    os.path.join(PROJECT_ROOT, "alerting"),
    os.path.join(PROJECT_ROOT, "ml_pipeline"),
]
for d in MODULE_DIRS:
    if d not in sys.path:
        sys.path.insert(0, d)


# ---------------------------------------------------------------------------
# Degradation / equipment config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def degradation_config():
    """Minimal degradation config understood by DegradationEngine."""
    return {
        "degradation_patterns": {
            "linear": {"type": "linear", "rate": 0.01},
            "exponential": {"type": "exponential", "growth_rate": 0.5},
            "step": {"type": "step", "magnitude": 0.6},
            "oscillating": {"type": "oscillating", "amplitude": 0.3, "frequency": 0.1},
        },
        "failure_modes": {
            "turbofan": {
                "fan_degradation": {
                    "pattern": "linear",
                    "primary_sensors": ["sensor_2", "sensor_3"],
                    "rul_range": [100, 300],
                    "severity_multiplier": 1.0,
                    "pattern_params": {"rate": 0.01},
                },
                "compressor_stall": {
                    "pattern": "exponential",
                    "primary_sensors": ["sensor_7", "sensor_8"],
                    "rul_range": [80, 200],
                    "severity_multiplier": 1.2,
                    "pattern_params": {"growth_rate": 0.5},
                },
            }
        },
        "progression": {
            "early_stage": {"sensor_deviation": 0.05, "noise_increase": 1.0},
            "middle_stage": {"sensor_deviation": 0.15, "noise_increase": 1.5},
            "late_stage": {"sensor_deviation": 0.30, "noise_increase": 2.0},
            "critical_stage": {"sensor_deviation": 0.50, "noise_increase": 3.0},
        },
    }


@pytest.fixture
def sensor_config():
    """Minimal sensor baseline config for SensorSimulator."""
    return {
        "sensor_baseline": {
            "temperature": {"mean": 500.0, "std": 5.0, "min": 400.0, "max": 650.0},
            "vibration": {"mean": 0.5, "std": 0.05, "min": 0.0, "max": 2.0},
            "pressure": {"mean": 30.0, "std": 1.0, "min": 10.0, "max": 50.0},
        }
    }


@pytest.fixture
def equipment_config(sensor_config):
    """Config for SensorDataGenerator."""
    return {
        "equipment_types": {
            "turbofan": sensor_config,
        }
    }


# ---------------------------------------------------------------------------
# Feature engineering config
# ---------------------------------------------------------------------------


@pytest.fixture
def feature_config():
    """Config dict for TimeDomainFeatures and FrequencyDomainFeatures."""
    return {
        "feature_engineering": {
            "time_domain": {
                "rolling_windows": [5, 10],
                "statistics": ["mean", "std", "min", "max"],
                "rate_of_change": {"enabled": True, "windows": [5]},
            },
            "frequency_domain": {
                "sampling_rate": 1.0,
                "window_size": 16,
                "overlap": 0.5,
                "target_sensors": ["sensor_2", "sensor_3"],
                "frequency_bands": {
                    "low": [0.0, 0.1],
                    "medium": [0.1, 0.3],
                    "high": [0.3, 0.5],
                },
            },
        }
    }


# ---------------------------------------------------------------------------
# Label generation config
# ---------------------------------------------------------------------------


@pytest.fixture
def label_config():
    """Config for LabelGenerator."""
    return {
        "label_generation": {
            "rul": {
                "method": "cycle_based",
                "max_rul": 300,
                "clip_rul": True,
                "piecewise_linear": {
                    "enabled": True,
                    "early_life_rul": 125,
                },
            },
            "binary_failure": {
                "window_before_failure": 30,
            },
            "health_status": {
                "categories": {
                    "healthy": [0.7, 1.0],
                    "warning": [0.3, 0.7],
                    "critical": [0.1, 0.3],
                    "imminent_failure": [0.0, 0.1],
                },
            },
            "degradation_rate": {
                "window_size": 5,
            },
        }
    }


# ---------------------------------------------------------------------------
# Alert engine config
# ---------------------------------------------------------------------------


@pytest.fixture
def alert_engine_config():
    """Config for AlertRuleEngine."""
    return {
        "rules": {
            "high_rul_warning": {
                "condition": "rul < 50",
                "severity": "WARNING",
                "message": "RUL below 50 for equipment {equipment_id}",
                "enabled": True,
                "cooldown": 0,  # disable cooldown for tests
            },
            "critical_rul": {
                "condition": "rul < 20",
                "severity": "CRITICAL",
                "message": "RUL critically low for equipment {equipment_id}",
                "enabled": True,
                "cooldown": 0,
            },
            "disabled_rule": {
                "condition": "rul < 100",
                "severity": "INFO",
                "message": "Disabled rule for {equipment_id}",
                "enabled": False,
                "cooldown": 0,
            },
        },
        "suppression": {
            "enabled": False,
            "maintenance_windows": [],
            "incident_mode": {"active": False},
        },
    }


# ---------------------------------------------------------------------------
# TimescaleDB writer config (mock-friendly)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_writer_config():
    """Config consumed by MockTimescaleDBWriter."""
    return {
        "timescaledb": {
            "host": "localhost",
            "port": 5432,
            "database": "predictive_maintenance",
            "user": "pmuser",
            "password": "pmpassword",
            "connection_pool": {"min_size": 1, "max_size": 2},
            "batch_write": {"batch_size": 100, "timeout_seconds": 5},
        }
    }
