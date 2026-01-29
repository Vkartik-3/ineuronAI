"""
Time-Domain Feature Engineering
Extracts statistical and temporal features from sensor data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TimeDomainFeatures:
    """Extracts time-domain features from sensor data"""

    def __init__(self, config: Dict):
        """
        Initialize feature extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        fe_config = config.get("feature_engineering", {})
        td_config = fe_config.get("time_domain", {})

        self.rolling_windows = td_config.get("rolling_windows", [10, 30, 60])
        self.statistics = td_config.get("statistics", ["mean", "std", "min", "max"])

        # Rate of change configuration
        roc_config = td_config.get("rate_of_change", {})
        self.roc_enabled = roc_config.get("enabled", True)
        self.roc_windows = roc_config.get("windows", [1, 5, 10])

        # Maintain rolling buffers per equipment
        self.rolling_buffers: Dict[str, Dict[str, deque]] = {}

        logger.info(
            f"Time-domain feature extractor initialized. Windows: {self.rolling_windows}"
        )

    def extract_features(self, equipment_id: str, sensor_data: Dict) -> Dict:
        """
        Extract time-domain features from sensor data

        Args:
            equipment_id: Unique equipment identifier
            sensor_data: Dictionary of sensor readings

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Initialize buffers for this equipment if needed
        if equipment_id not in self.rolling_buffers:
            self.rolling_buffers[equipment_id] = {}

        equipment_buffers = self.rolling_buffers[equipment_id]

        # Process each sensor
        for sensor_name, value in sensor_data.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)):
                continue

            # Initialize buffer for this sensor if needed
            if sensor_name not in equipment_buffers:
                max_window = max(self.rolling_windows)
                equipment_buffers[sensor_name] = deque(maxlen=max_window)

            # Add current value to buffer
            equipment_buffers[sensor_name].append(value)
            buffer = equipment_buffers[sensor_name]

            # Extract instantaneous features
            features[f"{sensor_name}_current"] = value

            # Extract rolling window features
            for window in self.rolling_windows:
                if len(buffer) >= window:
                    window_data = list(buffer)[-window:]
                    window_features = self._compute_statistics(
                        window_data, self.statistics
                    )

                    # Add window features with naming convention
                    for stat_name, stat_value in window_features.items():
                        feature_name = f"{sensor_name}_rolling_{window}_{stat_name}"
                        features[feature_name] = stat_value

            # Extract rate of change features
            if self.roc_enabled:
                roc_features = self._compute_rate_of_change(buffer, self.roc_windows)
                for roc_window, roc_value in roc_features.items():
                    features[f"{sensor_name}_roc_{roc_window}"] = roc_value

        return features

    def _compute_statistics(
        self, data: List[float], statistics: List[str]
    ) -> Dict[str, float]:
        """
        Compute statistical features

        Args:
            data: List of values
            statistics: List of statistic names to compute

        Returns:
            Dictionary of computed statistics
        """
        if not data:
            return {}

        arr = np.array(data)
        stats = {}

        for stat_name in statistics:
            try:
                if stat_name == "mean":
                    stats["mean"] = float(np.mean(arr))
                elif stat_name == "std":
                    stats["std"] = float(np.std(arr))
                elif stat_name == "min":
                    stats["min"] = float(np.min(arr))
                elif stat_name == "max":
                    stats["max"] = float(np.max(arr))
                elif stat_name == "median":
                    stats["median"] = float(np.median(arr))
                elif stat_name == "range":
                    stats["range"] = float(np.ptp(arr))  # Peak-to-peak
                elif stat_name == "rms":
                    stats["rms"] = float(np.sqrt(np.mean(arr**2)))
                elif stat_name == "skewness":
                    stats["skewness"] = float(pd.Series(arr).skew())
                elif stat_name == "kurtosis":
                    stats["kurtosis"] = float(pd.Series(arr).kurtosis())
            except Exception as e:
                logger.warning(f"Error computing {stat_name}: {e}")

        return stats

    def _compute_rate_of_change(
        self, buffer: deque, windows: List[int]
    ) -> Dict[int, float]:
        """
        Compute rate of change over different windows

        Args:
            buffer: Deque of historical values
            windows: List of window sizes

        Returns:
            Dictionary mapping window size to rate of change
        """
        roc_features = {}

        for window in windows:
            if len(buffer) >= window + 1:
                current = buffer[-1]
                previous = buffer[-(window + 1)]

                # Compute percentage change
                if previous != 0:
                    roc = ((current - previous) / abs(previous)) * 100
                else:
                    roc = 0.0

                roc_features[window] = round(float(roc), 4)

        return roc_features

    def clear_buffer(self, equipment_id: str):
        """Clear buffers for specific equipment"""
        if equipment_id in self.rolling_buffers:
            del self.rolling_buffers[equipment_id]
            logger.debug(f"Cleared buffers for equipment {equipment_id}")

    def get_buffer_status(self, equipment_id: str) -> Dict:
        """Get status of buffers for equipment"""
        if equipment_id not in self.rolling_buffers:
            return {"status": "not_initialized"}

        equipment_buffers = self.rolling_buffers[equipment_id]
        buffer_status = {}

        for sensor_name, buffer in equipment_buffers.items():
            buffer_status[sensor_name] = {
                "size": len(buffer),
                "capacity": buffer.maxlen,
            }

        return buffer_status


class AggregatedFeatures:
    """Computes aggregated features across multiple sensors"""

    @staticmethod
    def compute_cross_sensor_features(sensor_data: Dict) -> Dict:
        """
        Compute features across multiple sensors

        Args:
            sensor_data: Dictionary of sensor readings

        Returns:
            Dictionary of cross-sensor features
        """
        features = {}

        # Extract numeric values
        numeric_sensors = {
            k: v for k, v in sensor_data.items() if isinstance(v, (int, float))
        }

        if len(numeric_sensors) < 2:
            return features

        values = list(numeric_sensors.values())

        try:
            # Overall statistics
            features["sensors_mean"] = float(np.mean(values))
            features["sensors_std"] = float(np.std(values))
            features["sensors_range"] = float(np.ptp(values))

            # Coefficient of variation
            if features["sensors_mean"] != 0:
                features["sensors_cv"] = (
                    features["sensors_std"] / features["sensors_mean"]
                )

            # Count of sensors above/below mean
            mean_val = features["sensors_mean"]
            features["sensors_above_mean"] = sum(1 for v in values if v > mean_val)
            features["sensors_below_mean"] = sum(1 for v in values if v < mean_val)

        except Exception as e:
            logger.warning(f"Error computing cross-sensor features: {e}")

        return features

    @staticmethod
    def compute_sensor_correlations(
        sensor_buffers: Dict[str, deque], sensor_pairs: List[tuple]
    ) -> Dict:
        """
        Compute correlations between sensor pairs

        Args:
            sensor_buffers: Dictionary of sensor buffers
            sensor_pairs: List of (sensor1, sensor2) tuples

        Returns:
            Dictionary of correlation features
        """
        features = {}

        for sensor1, sensor2 in sensor_pairs:
            if sensor1 in sensor_buffers and sensor2 in sensor_buffers:
                buffer1 = list(sensor_buffers[sensor1])
                buffer2 = list(sensor_buffers[sensor2])

                # Need at least 10 points for meaningful correlation
                min_len = min(len(buffer1), len(buffer2))
                if min_len >= 10:
                    try:
                        arr1 = np.array(buffer1[-min_len:])
                        arr2 = np.array(buffer2[-min_len:])

                        correlation = np.corrcoef(arr1, arr2)[0, 1]
                        features[f"corr_{sensor1}_{sensor2}"] = float(correlation)
                    except Exception as e:
                        logger.warning(f"Error computing correlation: {e}")

        return features
