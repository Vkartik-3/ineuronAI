"""
Sensor Data Simulator
Generates realistic sensor readings for manufacturing equipment
"""

import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SensorSimulator:
    """Simulates sensor readings with realistic noise and variations"""

    def __init__(self, equipment_type: str, sensor_config: Dict):
        """
        Initialize sensor simulator

        Args:
            equipment_type: Type of equipment (e.g., 'turbofan_engine', 'pump')
            sensor_config: Configuration dictionary with sensor baselines
        """
        self.equipment_type = equipment_type
        self.sensor_config = sensor_config
        self.sensor_baseline = sensor_config.get("sensor_baseline", {})
        self.noise_enabled = True
        self.noise_multiplier = 0.5

    def generate_reading(
        self,
        sensor_name: str,
        cycle: int,
        degradation_factor: float = 0.0,
        noise_multiplier: float = 1.0,
    ) -> float:
        """
        Generate a single sensor reading

        Args:
            sensor_name: Name of the sensor
            cycle: Current operational cycle
            degradation_factor: Factor representing degradation (0=healthy, 1=failed)
            noise_multiplier: Multiplier for noise level

        Returns:
            Simulated sensor value
        """
        if sensor_name not in self.sensor_baseline:
            logger.warning(f"Sensor {sensor_name} not found in configuration")
            return 0.0

        sensor_spec = self.sensor_baseline[sensor_name]
        base_mean = sensor_spec["mean"]
        base_std = sensor_spec["std"]
        sensor_min = sensor_spec.get("min", base_mean - 3 * base_std)
        sensor_max = sensor_spec.get("max", base_mean + 3 * base_std)

        # Base value with small random variation
        value = base_mean

        # Add Gaussian noise
        if self.noise_enabled and base_std > 0:
            noise = np.random.normal(
                0, base_std * self.noise_multiplier * noise_multiplier
            )
            value += noise

        # Add slight temporal variation (simulates operating condition changes)
        temporal_variation = 0.01 * base_mean * np.sin(cycle * 0.01)
        value += temporal_variation

        # Apply degradation impact
        if degradation_factor > 0:
            # Degradation affects different sensors differently
            if "temperature" in sensor_name.lower():
                # Temperature typically increases with degradation
                value += degradation_factor * 0.15 * base_mean
            elif "vibration" in sensor_name.lower():
                # Vibration increases significantly
                value += degradation_factor * 0.5 * base_mean
            elif "pressure" in sensor_name.lower():
                # Pressure may decrease with degradation
                value -= degradation_factor * 0.1 * base_mean
            elif "flow" in sensor_name.lower():
                # Flow rate decreases
                value -= degradation_factor * 0.12 * base_mean
            elif "power" in sensor_name.lower():
                # Power consumption may increase
                value += degradation_factor * 0.2 * base_mean
            elif "fuel" in sensor_name.lower():
                # Fuel consumption increases with degradation
                value += degradation_factor * 0.18 * base_mean
            else:
                # Generic degradation effect
                value += (
                    degradation_factor * 0.1 * np.random.choice([-1, 1]) * base_mean
                )

        # Clip to realistic sensor ranges
        value = np.clip(value, sensor_min, sensor_max)

        return round(float(value), 4)

    def generate_all_sensors(
        self,
        cycle: int,
        degradation_factors: Optional[Dict[str, float]] = None,
        noise_multiplier: float = 1.0,
    ) -> Dict[str, float]:
        """
        Generate readings for all sensors

        Args:
            cycle: Current operational cycle
            degradation_factors: Dictionary mapping sensor names to degradation factors
            noise_multiplier: Global noise multiplier

        Returns:
            Dictionary of sensor readings
        """
        if degradation_factors is None:
            degradation_factors = {}

        readings = {}
        for sensor_name in self.sensor_baseline.keys():
            degradation = degradation_factors.get(sensor_name, 0.0)
            readings[sensor_name] = self.generate_reading(
                sensor_name, cycle, degradation, noise_multiplier
            )

        return readings

    def set_noise_level(self, enabled: bool, multiplier: float = 0.5):
        """Configure noise generation"""
        self.noise_enabled = enabled
        self.noise_multiplier = multiplier


class OperationalSettingSimulator:
    """Simulates operational settings that affect equipment behavior"""

    def __init__(self, equipment_type: str, sensor_config: Dict):
        self.equipment_type = equipment_type
        self.sensor_config = sensor_config

    def generate_settings(self, cycle: int) -> Dict[str, float]:
        """
        Generate operational settings

        Args:
            cycle: Current operational cycle

        Returns:
            Dictionary of operational settings
        """
        settings = {}
        sensor_baseline = self.sensor_config.get("sensor_baseline", {})

        # Extract operational settings from config
        for key in [
            "operational_setting_1",
            "operational_setting_2",
            "operational_setting_3",
        ]:
            if key in sensor_baseline:
                spec = sensor_baseline[key]
                mean = spec["mean"]
                std = spec["std"]

                if std > 0:
                    value = np.random.normal(mean, std)
                else:
                    value = mean

                settings[key] = round(float(value), 4)

        return settings


class SensorDataGenerator:
    """High-level interface for generating complete sensor data packages"""

    def __init__(self, equipment_type: str, equipment_config: Dict):
        """
        Initialize sensor data generator

        Args:
            equipment_type: Type of equipment
            equipment_config: Complete equipment configuration
        """
        self.equipment_type = equipment_type
        self.equipment_config = equipment_config

        sensor_config = equipment_config["equipment_types"].get(equipment_type, {})
        self.sensor_simulator = SensorSimulator(equipment_type, sensor_config)
        self.settings_simulator = OperationalSettingSimulator(
            equipment_type, sensor_config
        )

    def generate(
        self,
        equipment_id: str,
        cycle: int,
        degradation_factors: Optional[Dict[str, float]] = None,
        noise_multiplier: float = 1.0,
    ) -> Dict:
        """
        Generate complete sensor data package

        Args:
            equipment_id: Unique equipment identifier
            cycle: Current operational cycle
            degradation_factors: Sensor-specific degradation factors
            noise_multiplier: Noise level multiplier

        Returns:
            Complete sensor data dictionary ready for Kafka
        """
        timestamp = datetime.utcnow().isoformat() + "Z"

        sensor_readings = self.sensor_simulator.generate_all_sensors(
            cycle, degradation_factors, noise_multiplier
        )

        operational_settings = self.settings_simulator.generate_settings(cycle)

        data_package = {
            "equipment_id": equipment_id,
            "equipment_type": self.equipment_type,
            "timestamp": timestamp,
            "cycle": cycle,
            "operational_settings": operational_settings,
            "sensor_readings": sensor_readings,
            "metadata": {
                "location": "Factory_Floor_1",
                "model": f"{self.equipment_type.upper()}_v1",
                "install_date": "2024-01-01T00:00:00Z",
            },
        }

        return data_package
