"""
Time-Series Feature Engineering
Lag features, rolling windows, exponential moving averages
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesFeatureExtractor:
    """Extracts time-series features from sensor data"""

    def __init__(self, config: Dict):
        """
        Initialize time-series feature extractor

        Args:
            config: Configuration dictionary
        """
        self.config = config
        ts_config = config.get("feature_engineering", {}).get("time_series", {})

        # Lag features
        lag_config = ts_config.get("lag_features", {})
        self.lag_enabled = lag_config.get("enabled", True)
        self.lag_periods = lag_config.get("lag_periods", [1, 3, 5, 10])
        self.lag_sensors = lag_config.get("sensors", [])

        # Rolling window features
        window_config = ts_config.get("rolling_windows", {})
        self.window_enabled = window_config.get("enabled", True)
        self.rolling_windows = window_config.get("windows", [5, 10, 20])
        self.window_statistics = window_config.get("statistics", ["mean", "std"])
        self.window_sensors = window_config.get("sensors", [])

        # Exponential moving average
        ema_config = ts_config.get("ema", {})
        self.ema_enabled = ema_config.get("enabled", True)
        self.ema_alphas = ema_config.get("alpha_values", [0.1, 0.3, 0.5])
        self.ema_sensors = ema_config.get("sensors", [])

        # Cumulative features
        cum_config = ts_config.get("cumulative", {})
        self.cumulative_enabled = cum_config.get("enabled", True)
        self.cumulative_features = cum_config.get("features", ["cumulative_sum"])
        self.cumulative_sensors = cum_config.get("sensors", [])

        logger.info(
            f"Time-series feature extractor initialized. "
            f"Lag periods: {self.lag_periods}, Windows: {self.rolling_windows}"
        )

    def extract_features(
        self, df: pd.DataFrame, equipment_id: str = None
    ) -> pd.DataFrame:
        """
        Extract all time-series features

        Args:
            df: DataFrame with sensor readings (sorted by time)
            equipment_id: Optional equipment identifier for filtering

        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df

        # Filter by equipment if specified
        if equipment_id and "equipment_id" in df.columns:
            df = df[df["equipment_id"] == equipment_id].copy()

        # Ensure sorted by time
        if "time" in df.columns:
            df = df.sort_values("time")
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        result_df = df.copy()

        # Extract lag features
        if self.lag_enabled:
            result_df = self._add_lag_features(result_df)

        # Extract rolling window features
        if self.window_enabled:
            result_df = self._add_rolling_features(result_df)

        # Extract EMA features
        if self.ema_enabled:
            result_df = self._add_ema_features(result_df)

        # Extract cumulative features
        if self.cumulative_enabled:
            result_df = self._add_cumulative_features(result_df)

        # Add rate of change features
        result_df = self._add_rate_of_change(result_df)

        # Add time-based features
        result_df = self._add_time_features(result_df)

        logger.debug(
            f"Extracted {len(result_df.columns) - len(df.columns)} time-series features"
        )

        return result_df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features"""
        for sensor in self.lag_sensors:
            if sensor not in df.columns:
                continue

            for lag in self.lag_periods:
                feature_name = f"{sensor}_lag_{lag}"
                df[feature_name] = df[sensor].shift(lag)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        for sensor in self.window_sensors:
            if sensor not in df.columns:
                continue

            for window in self.rolling_windows:
                for stat in self.window_statistics:
                    feature_name = f"{sensor}_rolling_{window}_{stat}"

                    if stat == "mean":
                        df[feature_name] = (
                            df[sensor].rolling(window=window, min_periods=1).mean()
                        )
                    elif stat == "std":
                        df[feature_name] = (
                            df[sensor].rolling(window=window, min_periods=1).std()
                        )
                    elif stat == "min":
                        df[feature_name] = (
                            df[sensor].rolling(window=window, min_periods=1).min()
                        )
                    elif stat == "max":
                        df[feature_name] = (
                            df[sensor].rolling(window=window, min_periods=1).max()
                        )
                    elif stat == "median":
                        df[feature_name] = (
                            df[sensor].rolling(window=window, min_periods=1).median()
                        )
                    elif stat == "skewness":
                        df[feature_name] = (
                            df[sensor].rolling(window=window, min_periods=3).skew()
                        )
                    elif stat == "kurtosis":
                        df[feature_name] = (
                            df[sensor].rolling(window=window, min_periods=4).kurt()
                        )

        return df

    def _add_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add exponential moving average features"""
        for sensor in self.ema_sensors:
            if sensor not in df.columns:
                continue

            for alpha in self.ema_alphas:
                feature_name = f"{sensor}_ema_{int(alpha * 100)}"
                df[feature_name] = df[sensor].ewm(alpha=alpha, adjust=False).mean()

        return df

    def _add_cumulative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative features"""
        for sensor in self.cumulative_sensors:
            if sensor not in df.columns:
                continue

            for feature_type in self.cumulative_features:
                feature_name = f"{sensor}_{feature_type}"

                if feature_type == "cumulative_sum":
                    df[feature_name] = df[sensor].cumsum()
                elif feature_type == "cumulative_max":
                    df[feature_name] = df[sensor].cummax()
                elif feature_type == "cumulative_min":
                    df[feature_name] = df[sensor].cummin()

        return df

    def _add_rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate of change features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col.startswith("cycle") or col.endswith("_id"):
                continue

            # First-order difference (rate of change)
            df[f"{col}_diff"] = df[col].diff()

            # Percentage change
            df[f"{col}_pct_change"] = df[col].pct_change()

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        if "cycle" in df.columns:
            # Normalize cycle number
            df["cycle_normalized"] = (
                df["cycle"] / df["cycle"].max() if df["cycle"].max() > 0 else 0
            )

            # Cycle bins (early, mid, late life)
            df["cycle_bin"] = pd.cut(
                df["cycle"],
                bins=5,
                labels=["early", "mid_early", "mid", "mid_late", "late"],
            )

        if "time" in df.columns or "timestamp" in df.columns:
            time_col = "time" if "time" in df.columns else "timestamp"

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])

            # Time since start
            df["time_since_start"] = (
                df[time_col] - df[time_col].iloc[0]
            ).dt.total_seconds()

        return df


class SequenceGenerator:
    """Generates sequences for LSTM training"""

    def __init__(self, sequence_length: int = 50, stride: int = 1):
        """
        Initialize sequence generator

        Args:
            sequence_length: Length of each sequence
            stride: Step size between sequences
        """
        self.sequence_length = sequence_length
        self.stride = stride
        logger.info(
            f"Sequence generator initialized. Length: {sequence_length}, Stride: {stride}"
        )

    def generate_sequences(
        self, df: pd.DataFrame, feature_cols: List[str], target_col: str = "rul"
    ) -> tuple:
        """
        Generate sequences for LSTM

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Tuple of (X, y) where X is sequences, y is targets
        """
        X_sequences = []
        y_targets = []

        # Generate sequences
        for i in range(0, len(df) - self.sequence_length + 1, self.stride):
            # Extract sequence
            sequence = df.iloc[i : i + self.sequence_length][feature_cols].values

            # Target is the last value in the sequence
            target = df.iloc[i + self.sequence_length - 1][target_col]

            X_sequences.append(sequence)
            y_targets.append(target)

        X = np.array(X_sequences)
        y = np.array(y_targets)

        logger.info(f"Generated {len(X)} sequences. Shape: {X.shape}")

        return X, y

    def generate_sequences_per_equipment(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "rul",
        equipment_col: str = "equipment_id",
    ) -> tuple:
        """
        Generate sequences grouped by equipment

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name
            equipment_col: Equipment identifier column

        Returns:
            Tuple of (X, y, equipment_ids)
        """
        X_all = []
        y_all = []
        equipment_ids = []

        # Group by equipment
        for equipment_id, group in df.groupby(equipment_col):
            # Sort by time/cycle
            if "cycle" in group.columns:
                group = group.sort_values("cycle")
            elif "time" in group.columns:
                group = group.sort_values("time")

            # Generate sequences for this equipment
            if len(group) >= self.sequence_length:
                X, y = self.generate_sequences(group, feature_cols, target_col)

                X_all.append(X)
                y_all.append(y)
                equipment_ids.extend([equipment_id] * len(X))

        # Concatenate all sequences
        if X_all:
            X_combined = np.vstack(X_all)
            y_combined = np.concatenate(y_all)
        else:
            X_combined = np.array([])
            y_combined = np.array([])

        logger.info(
            f"Generated {len(X_combined)} sequences from {len(df[equipment_col].unique())} equipment. "
            f"Shape: {X_combined.shape}"
        )

        return X_combined, y_combined, equipment_ids
