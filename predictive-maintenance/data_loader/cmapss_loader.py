"""
NASA C-MAPSS Dataset Loader

Loads and preprocesses the NASA Commercial Modular Aero-Propulsion System
Simulation (C-MAPSS) Turbofan Engine Degradation Dataset.

Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation
           Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08, 2008.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CMAPSSLoader:
    """
    Loads and preprocesses NASA C-MAPSS turbofan engine degradation dataset.

    Dataset Details:
    - FD001: 100 train engines, 100 test engines, 1 operating condition, 1 fault mode
    - FD002: 260 train engines, 259 test engines, 6 operating conditions, 1 fault mode
    - FD003: 100 train engines, 100 test engines, 1 operating condition, 2 fault modes
    - FD004: 248 train engines, 249 test engines, 6 operating conditions, 2 fault modes

    Each row represents one operational cycle with:
    - Unit ID (engine identifier)
    - Time cycle
    - 3 operational settings
    - 21 sensor measurements
    """

    # Column names for the dataset
    COLUMN_NAMES = [
        "unit_id",
        "time_cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        "sensor_1",
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_5",
        "sensor_6",
        "sensor_7",
        "sensor_8",
        "sensor_9",
        "sensor_10",
        "sensor_11",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
        "sensor_16",
        "sensor_17",
        "sensor_18",
        "sensor_19",
        "sensor_20",
        "sensor_21",
    ]

    # Sensor name mapping (based on NASA documentation)
    SENSOR_DESCRIPTIONS = {
        "sensor_1": "T2 (Total temperature at fan inlet, °R)",
        "sensor_2": "T24 (Total temperature at LPC outlet, °R)",
        "sensor_3": "T30 (Total temperature at HPC outlet, °R)",
        "sensor_4": "T50 (Total temperature at LPT outlet, °R)",
        "sensor_5": "P2 (Pressure at fan inlet, psia)",
        "sensor_6": "P15 (Total pressure in bypass-duct, psia)",
        "sensor_7": "P30 (Total pressure at HPC outlet, psia)",
        "sensor_8": "Nf (Physical fan speed, rpm)",
        "sensor_9": "Nc (Physical core speed, rpm)",
        "sensor_10": "epr (Engine pressure ratio, P50/P2)",
        "sensor_11": "Ps30 (Static pressure at HPC outlet, psia)",
        "sensor_12": "phi (Ratio of fuel flow to Ps30, pps/psi)",
        "sensor_13": "NRf (Corrected fan speed, rpm)",
        "sensor_14": "NRc (Corrected core speed, rpm)",
        "sensor_15": "BPR (Bypass Ratio)",
        "sensor_16": "farB (Burner fuel-air ratio)",
        "sensor_17": "htBleed (Bleed Enthalpy)",
        "sensor_18": "Nf_dmd (Demanded fan speed, rpm)",
        "sensor_19": "PCNfR_dmd (Demanded corrected fan speed, rpm)",
        "sensor_20": "W31 (HPT coolant bleed, lbm/s)",
        "sensor_21": "W32 (LPT coolant bleed, lbm/s)",
    }

    def __init__(
        self, dataset_path: str = "../../archive/CMaps", dataset_id: str = "FD001"
    ):
        """
        Initialize C-MAPSS loader.

        Args:
            dataset_path: Path to C-MAPSS dataset directory
            dataset_id: Dataset identifier (FD001, FD002, FD003, FD004)
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_id = dataset_id

        # File paths
        self.train_file = self.dataset_path / f"train_{dataset_id}.txt"
        self.test_file = self.dataset_path / f"test_{dataset_id}.txt"
        self.rul_file = self.dataset_path / f"RUL_{dataset_id}.txt"

        # Validate files exist
        self._validate_files()

        logger.info(f"CMAPSSLoader initialized for dataset {dataset_id}")

    def _validate_files(self) -> None:
        """Validate that all required files exist"""
        for file_path in [self.train_file, self.test_file, self.rul_file]:
            if not file_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

    def load_train_data(self) -> pd.DataFrame:
        """
        Load training data and compute RUL labels.

        Training data contains complete run-to-failure trajectories.
        RUL is computed as (max_cycle - current_cycle) for each engine.

        Returns:
            DataFrame with columns: unit_id, time_cycle, op_settings, sensors, rul
        """
        logger.info(f"Loading training data from {self.train_file}")

        # Read data
        train_df = pd.read_csv(
            self.train_file,
            sep="\s+",  # Space-separated
            header=None,
            names=self.COLUMN_NAMES,
        )

        # Compute RUL for training data
        # RUL = max_cycle - current_cycle for each engine
        train_df = self._add_rul_to_train(train_df)

        logger.info(
            f"Loaded training data: {len(train_df)} records, "
            f"{train_df['unit_id'].nunique()} engines"
        )

        return train_df

    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data and add RUL labels from RUL file.

        Test data ends before failure. True RUL values are provided
        in a separate file.

        Returns:
            DataFrame with columns: unit_id, time_cycle, op_settings, sensors, rul
        """
        logger.info(f"Loading test data from {self.test_file}")

        # Read test data
        test_df = pd.read_csv(
            self.test_file, sep="\s+", header=None, names=self.COLUMN_NAMES
        )

        # Read true RUL values
        rul_values = pd.read_csv(self.rul_file, sep="\s+", header=None, names=["rul"])

        # Add RUL to test data
        test_df = self._add_rul_to_test(test_df, rul_values)

        logger.info(
            f"Loaded test data: {len(test_df)} records, "
            f"{test_df['unit_id'].nunique()} engines"
        )

        return test_df

    def _add_rul_to_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RUL column to training data.

        For training data, RUL = max_cycle - current_cycle for each engine.

        Args:
            df: Training dataframe

        Returns:
            DataFrame with RUL column added
        """
        # Get max cycle for each engine
        max_cycles = df.groupby("unit_id")["time_cycle"].max().reset_index()
        max_cycles.columns = ["unit_id", "max_cycle"]

        # Merge and compute RUL
        df = df.merge(max_cycles, on="unit_id", how="left")
        df["rul"] = df["max_cycle"] - df["time_cycle"]
        df = df.drop("max_cycle", axis=1)

        return df

    def _add_rul_to_test(self, df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add RUL column to test data.

        For test data, RUL is provided in separate file. The value represents
        RUL at the last observed cycle for each engine.

        Args:
            df: Test dataframe
            rul_df: DataFrame with true RUL values (one per engine)

        Returns:
            DataFrame with RUL column added
        """
        # Add unit_id to RUL dataframe (sequential 1, 2, 3...)
        rul_df["unit_id"] = range(1, len(rul_df) + 1)

        # Get max cycle for each engine in test data
        max_cycles = df.groupby("unit_id")["time_cycle"].max().reset_index()
        max_cycles.columns = ["unit_id", "max_cycle"]

        # Merge RUL values
        df = df.merge(max_cycles, on="unit_id", how="left")
        df = df.merge(rul_df, on="unit_id", how="left")

        # Compute RUL for each cycle: rul_at_last_cycle + (max_cycle - current_cycle)
        df["rul"] = df["rul"] + (df["max_cycle"] - df["time_cycle"])
        df = df.drop("max_cycle", axis=1)

        return df

    def get_sensor_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistical summary of sensor measurements.

        Args:
            df: DataFrame with sensor data

        Returns:
            DataFrame with sensor statistics
        """
        sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
        return df[sensor_cols].describe()

    def identify_constant_sensors(
        self, df: pd.DataFrame, threshold: float = 0.001
    ) -> list:
        """
        Identify sensors with near-constant values (low variance).

        These sensors may not be useful for modeling.

        Args:
            df: DataFrame with sensor data
            threshold: Variance threshold (sensors below this are considered constant)

        Returns:
            List of constant sensor column names
        """
        sensor_cols = [col for col in df.columns if col.startswith("sensor_")]
        constant_sensors = []

        for col in sensor_cols:
            if df[col].std() < threshold:
                constant_sensors.append(col)

        if constant_sensors:
            logger.info(f"Constant sensors detected: {constant_sensors}")

        return constant_sensors

    def normalize_sensors(
        self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
        """
        Normalize sensor values using training set statistics.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe (optional)

        Returns:
            Tuple of (normalized_train, normalized_test, normalization_params)
        """
        sensor_cols = [col for col in train_df.columns if col.startswith("sensor_")]
        op_setting_cols = [
            col for col in train_df.columns if col.startswith("op_setting_")
        ]

        # Compute normalization parameters from training data
        norm_params = {}
        for col in sensor_cols + op_setting_cols:
            norm_params[col] = {
                "mean": train_df[col].mean(),
                "std": train_df[col].std(),
            }

        # Normalize training data
        train_normalized = train_df.copy()
        for col in sensor_cols + op_setting_cols:
            mean = norm_params[col]["mean"]
            std = norm_params[col]["std"]
            if std > 0:
                train_normalized[col] = (train_df[col] - mean) / std

        # Normalize test data if provided
        test_normalized = None
        if test_df is not None:
            test_normalized = test_df.copy()
            for col in sensor_cols + op_setting_cols:
                mean = norm_params[col]["mean"]
                std = norm_params[col]["std"]
                if std > 0:
                    test_normalized[col] = (test_df[col] - mean) / std

        logger.info("Sensor normalization complete")

        return train_normalized, test_normalized, norm_params

    def get_dataset_info(self) -> Dict:
        """
        Get information about the loaded dataset.

        Returns:
            Dictionary with dataset metadata
        """
        train_df = self.load_train_data()
        test_df = self.load_test_data()

        return {
            "dataset_id": self.dataset_id,
            "train_engines": train_df["unit_id"].nunique(),
            "train_records": len(train_df),
            "test_engines": test_df["unit_id"].nunique(),
            "test_records": len(test_df),
            "sensors": 21,
            "operating_settings": 3,
            "avg_train_cycles_per_engine": train_df.groupby("unit_id")["time_cycle"]
            .max()
            .mean(),
            "avg_test_cycles_per_engine": test_df.groupby("unit_id")["time_cycle"]
            .max()
            .mean(),
        }


def main():
    """Test C-MAPSS loader"""
    print("\n=== NASA C-MAPSS Dataset Loader ===\n")

    # Initialize loader
    loader = CMAPSSLoader(dataset_path="../../archive/CMaps", dataset_id="FD001")

    # Get dataset info
    info = loader.get_dataset_info()
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Load data
    train_df = loader.load_train_data()
    test_df = loader.load_test_data()

    print(f"\nTraining Data Shape: {train_df.shape}")
    print(f"Test Data Shape: {test_df.shape}")

    print("\nTraining Data Sample:")
    print(train_df.head())

    print("\nSensor Statistics:")
    print(loader.get_sensor_stats(train_df))

    # Check for constant sensors
    constant_sensors = loader.identify_constant_sensors(train_df)
    print(f"\nConstant Sensors: {constant_sensors}")

    # Normalize data
    train_norm, test_norm, norm_params = loader.normalize_sensors(train_df, test_df)
    print(f"\nNormalized Training Data Shape: {train_norm.shape}")
    print(f"Normalized Test Data Shape: {test_norm.shape}")

    print("\n✅ C-MAPSS data loaded successfully!")


if __name__ == "__main__":
    main()
