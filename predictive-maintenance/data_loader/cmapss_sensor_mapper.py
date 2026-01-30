"""
C-MAPSS Sensor Mapping

Maps NASA C-MAPSS sensor readings to standard feature names
used by the predictive maintenance system.
"""

from typing import Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CMAPSSSensorMapper:
    """
    Maps C-MAPSS sensor columns to standard feature names.

    C-MAPSS has 21 sensors + 3 operational settings.
    This mapper converts them to the feature names expected
    by downstream processing (feature engineering, ML pipeline).
    """

    # Mapping from C-MAPSS sensors to standard feature names
    SENSOR_MAPPING = {
        # Temperature sensors (°R to °F conversion: °F = °R - 459.67)
        "sensor_2": "temperature",  # T24 - LPC outlet temperature (primary temp sensor)
        "sensor_3": "temp_hpc_outlet",  # T30 - HPC outlet temperature
        "sensor_4": "temp_lpt_outlet",  # T50 - LPT outlet temperature
        # Pressure sensors (psia)
        "sensor_7": "pressure",  # P30 - HPC outlet pressure (primary pressure sensor)
        "sensor_11": "pressure_static_hpc",  # Ps30 - Static pressure at HPC outlet
        # Speed sensors (rpm)
        "sensor_8": "rpm",  # Nf - Physical fan speed (primary rpm sensor)
        "sensor_9": "rpm_core",  # Nc - Physical core speed
        "sensor_13": "rpm_corrected_fan",  # NRf - Corrected fan speed
        "sensor_14": "rpm_corrected_core",  # NRc - Corrected core speed
        # Performance ratios
        "sensor_10": "engine_pressure_ratio",  # epr - P50/P2
        "sensor_15": "bypass_ratio",  # BPR
        "sensor_16": "fuel_air_ratio",  # farB - Burner fuel-air ratio
        "sensor_12": "fuel_flow_ratio",  # phi - Fuel flow to Ps30 ratio
        # Bleed and enthalpy
        "sensor_17": "bleed_enthalpy",  # htBleed
        "sensor_20": "coolant_bleed_hpt",  # W31 - HPT coolant bleed
        "sensor_21": "coolant_bleed_lpt",  # W32 - LPT coolant bleed
        # Additional sensors (for completeness)
        "sensor_1": "temp_fan_inlet",  # T2
        "sensor_5": "pressure_fan_inlet",  # P2
        "sensor_6": "pressure_bypass",  # P15
        "sensor_18": "fan_speed_demand",  # Nf_dmd
        "sensor_19": "corrected_fan_speed_demand",  # PCNfR_dmd
    }

    # Primary sensors for basic monitoring (subset of all sensors)
    PRIMARY_SENSORS = [
        "temperature",  # sensor_2
        "pressure",  # sensor_7
        "rpm",  # sensor_8
        "engine_pressure_ratio",  # sensor_10
        "fuel_flow_ratio",  # sensor_12
    ]

    # Sensors that are typically constant and can be removed
    CONSTANT_SENSORS = [
        "sensor_1",  # T2 - often constant at sea level
        "sensor_5",  # P2 - often constant
        "sensor_6",  # P15 - bypass duct pressure
        "sensor_18",  # Nf_dmd - demand signals
        "sensor_19",  # PCNfR_dmd
    ]

    @classmethod
    def map_cmapss_to_standard(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Map C-MAPSS sensor columns to standard feature names.

        Args:
            data: DataFrame with C-MAPSS sensor columns (sensor_1 to sensor_21)

        Returns:
            DataFrame with mapped feature names
        """
        mapped_data = data.copy()

        # Rename sensor columns to standard names
        rename_dict = {}
        for cmapss_col, standard_col in cls.SENSOR_MAPPING.items():
            if cmapss_col in mapped_data.columns:
                rename_dict[cmapss_col] = standard_col

        mapped_data = mapped_data.rename(columns=rename_dict)

        # Convert temperature from Rankine to Fahrenheit if needed
        temp_cols = [
            "temperature",
            "temp_hpc_outlet",
            "temp_lpt_outlet",
            "temp_fan_inlet",
        ]
        for col in temp_cols:
            if col in mapped_data.columns:
                # C-MAPSS temperatures are in Rankine, convert to Fahrenheit
                # Only if values are in Rankine range (> 400)
                if mapped_data[col].mean() > 400:
                    mapped_data[col] = mapped_data[col] - 459.67

        logger.info(f"Mapped {len(rename_dict)} C-MAPSS sensors to standard names")

        return mapped_data

    @classmethod
    def get_feature_names(cls) -> List[str]:
        """
        Get list of all standard feature names.

        Returns:
            List of feature names
        """
        return list(cls.SENSOR_MAPPING.values())

    @classmethod
    def get_primary_features(cls) -> List[str]:
        """
        Get list of primary features for basic monitoring.

        Returns:
            List of primary feature names
        """
        return cls.PRIMARY_SENSORS

    @classmethod
    def extract_operating_settings(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract operating settings from C-MAPSS data.

        Args:
            data: DataFrame with op_setting_1, op_setting_2, op_setting_3

        Returns:
            DataFrame with operating settings
        """
        op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

        if all(col in data.columns for col in op_cols):
            return data[op_cols]
        else:
            logger.warning("Operating setting columns not found")
            return pd.DataFrame()

    @classmethod
    def create_unified_message(cls, cmapss_message: Dict) -> Dict:
        """
        Convert C-MAPSS Kafka message to unified format.

        Args:
            cmapss_message: Message from CMAPSSKafkaStreamer

        Returns:
            Unified message format compatible with existing pipeline
        """
        # Extract sensor data
        sensors = cmapss_message.get("sensors", {})
        op_settings = cmapss_message.get("operating_settings", {})

        # Map primary sensors
        unified_message = {
            "equipment_id": cmapss_message.get("equipment_id"),
            "unit_id": cmapss_message.get("unit_id"),  # Preserve C-MAPSS unit_id
            "equipment_type": "turbofan_engine",
            "timestamp": cmapss_message.get("timestamp"),
            "time_cycle": cmapss_message.get("time_cycle"),
            "data_source": "cmapss",
            "dataset": cmapss_message.get("dataset", "FD001"),
        }

        # Map sensors to standard names
        if "sensor_2" in sensors:  # T24
            unified_message["temperature"] = sensors["sensor_2"] - 459.67  # R to F
        if "sensor_7" in sensors:  # P30
            unified_message["pressure"] = sensors["sensor_7"]
        if "sensor_8" in sensors:  # Nf
            unified_message["rpm"] = sensors["sensor_8"]
        if "sensor_3" in sensors:  # T30 - can be used as vibration proxy
            # Use temperature variation as proxy for vibration
            unified_message["vibration_rms"] = (sensors["sensor_3"] - 1500) / 100.0
        if "sensor_12" in sensors:  # phi - fuel flow ratio
            # Can be used as power proxy
            unified_message["power"] = sensors["sensor_12"] * 1000.0

        # Add all C-MAPSS sensors for feature engineering
        unified_message["cmapss_sensors"] = sensors
        unified_message["operating_settings"] = op_settings

        # Add RUL if available (for evaluation)
        if "rul" in cmapss_message:
            unified_message["rul"] = cmapss_message["rul"]

        return unified_message


def map_cmapss_data_for_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map C-MAPSS DataFrame to format expected by ML pipeline.

    Args:
        df: DataFrame with C-MAPSS data (from CMAPSSLoader)

    Returns:
        DataFrame in standard format for ML pipeline
    """
    # Map sensors to standard names
    mapped_df = CMAPSSSensorMapper.map_cmapss_to_standard(df)

    # Ensure required columns exist
    required_cols = ["unit_id", "time_cycle", "rul"]
    for col in required_cols:
        if col not in mapped_df.columns:
            logger.warning(f"Required column '{col}' not found in data")

    return mapped_df


def main():
    """Test sensor mapping"""
    import pandas as pd

    print("\n=== C-MAPSS Sensor Mapper ===\n")

    # Create sample C-MAPSS data
    sample_data = {
        "unit_id": [1, 1, 1],
        "time_cycle": [1, 2, 3],
        "sensor_1": [518.67, 518.67, 518.67],  # T2 (Rankine)
        "sensor_2": [643.0, 644.0, 645.0],  # T24 (Rankine)
        "sensor_7": [552.0, 553.0, 554.0],  # P30 (psia)
        "sensor_8": [2388.0, 2389.0, 2390.0],  # Nf (rpm)
        "sensor_10": [1.3, 1.3, 1.3],  # EPR
        "rul": [100, 99, 98],
    }

    df = pd.DataFrame(sample_data)

    print("Original C-MAPSS Data:")
    print(df.head())

    # Map to standard format
    mapped_df = CMAPSSSensorMapper.map_cmapss_to_standard(df)

    print("\nMapped Data:")
    print(mapped_df.head())

    print("\nAvailable Feature Names:")
    print(CMAPSSSensorMapper.get_feature_names())

    print("\nPrimary Features:")
    print(CMAPSSSensorMapper.get_primary_features())

    # Test message conversion
    cmapss_message = {
        "equipment_id": "ENGINE_0001",
        "unit_id": 1,
        "timestamp": "2026-01-30T10:00:00",
        "time_cycle": 1,
        "dataset": "FD001",
        "sensors": {"sensor_2": 643.0, "sensor_7": 552.0, "sensor_8": 2388.0},
        "operating_settings": {
            "op_setting_1": -0.0007,
            "op_setting_2": -0.0004,
            "op_setting_3": 100.0,
        },
        "rul": 100,
    }

    unified_msg = CMAPSSSensorMapper.create_unified_message(cmapss_message)

    print("\nUnified Message:")
    import json

    print(json.dumps(unified_msg, indent=2))


if __name__ == "__main__":
    main()
