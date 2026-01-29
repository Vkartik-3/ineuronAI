"""
Label Generation for ML Training
RUL calculation, binary failure labels, health status classification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates labels for supervised learning"""

    def __init__(self, config: Dict):
        """
        Initialize label generator

        Args:
            config: Configuration dictionary
        """
        self.config = config
        label_config = config.get("label_generation", {})

        # RUL configuration
        rul_config = label_config.get("rul", {})
        self.rul_method = rul_config.get("method", "cycle_based")
        self.max_rul = rul_config.get("max_rul", 300)
        self.clip_rul = rul_config.get("clip_rul", True)

        # Piecewise linear RUL
        piecewise_config = rul_config.get("piecewise_linear", {})
        self.piecewise_enabled = piecewise_config.get("enabled", True)
        self.early_life_rul = piecewise_config.get("early_life_rul", 125)

        # Binary failure labels
        binary_config = label_config.get("binary_failure", {})
        self.binary_enabled = binary_config.get("enabled", True)
        self.failure_window = binary_config.get("window_before_failure", 30)

        # Health status
        health_config = label_config.get("health_status", {})
        self.health_enabled = health_config.get("enabled", True)
        self.health_categories = health_config.get("categories", {})

        # Degradation rate
        deg_config = label_config.get("degradation_rate", {})
        self.degradation_enabled = deg_config.get("enabled", True)
        self.degradation_window = deg_config.get("window_size", 10)

        logger.info(
            f"Label generator initialized. Method: {self.rul_method}, "
            f"Max RUL: {self.max_rul}"
        )

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all labels

        Args:
            df: DataFrame with sensor data

        Returns:
            DataFrame with labels
        """
        if df.empty:
            logger.warning("Empty dataframe provided")
            return df

        result_df = df.copy()

        # Generate RUL labels
        result_df = self._generate_rul_labels(result_df)

        # Generate binary failure labels
        if self.binary_enabled:
            result_df = self._generate_binary_labels(result_df)

        # Generate health status labels
        if self.health_enabled:
            result_df = self._generate_health_status(result_df)

        # Generate degradation rate
        if self.degradation_enabled:
            result_df = self._generate_degradation_rate(result_df)

        logger.debug(f"Generated labels for {len(result_df)} records")

        return result_df

    def _generate_rul_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate RUL (Remaining Useful Life) labels

        Args:
            df: DataFrame with sensor data

        Returns:
            DataFrame with RUL column
        """
        if "equipment_id" not in df.columns:
            logger.error("equipment_id column not found")
            return df

        # Group by equipment
        rul_list = []

        for equipment_id, group in df.groupby("equipment_id"):
            # Sort by cycle or time
            if "cycle" in group.columns:
                group = group.sort_values("cycle")
                time_col = "cycle"
            elif "time" in group.columns:
                group = group.sort_values("time")
                time_col = "time"
            else:
                logger.warning(f"No time column found for equipment {equipment_id}")
                rul_list.extend([np.nan] * len(group))
                continue

            # Calculate RUL
            max_time = group[time_col].max()

            if self.rul_method == "cycle_based":
                # RUL = max_cycle - current_cycle
                rul = max_time - group[time_col]

                # Apply piecewise linear if enabled
                if self.piecewise_enabled:
                    # Early life: constant RUL
                    rul = rul.where(rul <= self.early_life_rul, self.early_life_rul)

                # Clip to max_rul if enabled
                if self.clip_rul:
                    rul = rul.clip(upper=self.max_rul)

            elif self.rul_method == "time_based":
                # RUL based on time remaining
                rul = max_time - group[time_col]

                if self.clip_rul:
                    rul = rul.clip(upper=self.max_rul)

            else:
                logger.warning(f"Unknown RUL method: {self.rul_method}")
                rul = pd.Series([np.nan] * len(group))

            rul_list.extend(rul.values)

        df["rul"] = rul_list

        # Normalized RUL (0-1)
        if self.max_rul > 0:
            df["rul_normalized"] = df["rul"] / self.max_rul
        else:
            df["rul_normalized"] = 0

        return df

    def _generate_binary_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate binary failure labels

        Args:
            df: DataFrame with RUL

        Returns:
            DataFrame with binary failure column
        """
        if "rul" not in df.columns:
            logger.warning("RUL column not found, generating first")
            df = self._generate_rul_labels(df)

        # Label as 1 if within failure window, 0 otherwise
        df["failure_imminent"] = (df["rul"] <= self.failure_window).astype(int)

        return df

    def _generate_health_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate multi-class health status labels

        Args:
            df: DataFrame with normalized RUL

        Returns:
            DataFrame with health status column
        """
        if "rul_normalized" not in df.columns:
            logger.warning("Normalized RUL not found, generating first")
            df = self._generate_rul_labels(df)

        def categorize_health(rul_norm):
            if pd.isna(rul_norm):
                return "unknown"

            for category, (lower, upper) in self.health_categories.items():
                if lower <= rul_norm < upper:
                    return category

            return "unknown"

        df["health_status"] = df["rul_normalized"].apply(categorize_health)

        # Encode as integers
        health_mapping = {
            "healthy": 0,
            "warning": 1,
            "critical": 2,
            "imminent_failure": 3,
            "unknown": -1,
        }
        df["health_status_code"] = df["health_status"].map(health_mapping)

        return df

    def _generate_degradation_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate degradation rate (change in health over time)

        Args:
            df: DataFrame with sensor data

        Returns:
            DataFrame with degradation rate column
        """
        if "equipment_id" not in df.columns:
            return df

        # Group by equipment
        deg_rate_list = []

        for equipment_id, group in df.groupby("equipment_id"):
            # Sort by cycle
            if "cycle" in group.columns:
                group = group.sort_values("cycle")

            # Calculate degradation indicator (could be vibration, temperature, etc.)
            # For now, use RUL change rate
            if "rul" in group.columns:
                # Rate of RUL decrease
                rul_change = group["rul"].diff(periods=self.degradation_window)
                degradation_rate = (
                    -rul_change / self.degradation_window
                )  # Negative because RUL decreases
            else:
                degradation_rate = pd.Series([np.nan] * len(group))

            deg_rate_list.extend(degradation_rate.values)

        df["degradation_rate"] = deg_rate_list

        return df

    def generate_failure_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate failure event markers

        Args:
            df: DataFrame with equipment data

        Returns:
            DataFrame with failure event column
        """
        if "equipment_id" not in df.columns:
            return df

        # Mark last cycle for each equipment as failure event
        failure_events = []

        for equipment_id, group in df.groupby("equipment_id"):
            if "cycle" in group.columns:
                max_cycle = group["cycle"].max()
                is_failure = (group["cycle"] == max_cycle).astype(int)
            else:
                is_failure = [0] * len(group)

            failure_events.extend(is_failure)

        df["failure_event"] = failure_events

        return df

    def calculate_time_to_failure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time to failure for each record

        Args:
            df: DataFrame with time/cycle information

        Returns:
            DataFrame with time_to_failure column
        """
        if "equipment_id" not in df.columns:
            return df

        ttf_list = []

        for equipment_id, group in df.groupby("equipment_id"):
            if "time" in group.columns:
                group = group.sort_values("time")
                max_time = group["time"].max()
                ttf = max_time - group["time"]
            elif "cycle" in group.columns:
                group = group.sort_values("cycle")
                max_cycle = group["cycle"].max()
                ttf = max_cycle - group["cycle"]
            else:
                ttf = pd.Series([np.nan] * len(group))

            ttf_list.extend(ttf.values)

        df["time_to_failure"] = ttf_list

        return df


class DatasetSplitter:
    """Splits data into train/validation/test sets"""

    @staticmethod
    def split_by_equipment(
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data by equipment to avoid data leakage

        Args:
            df: DataFrame with equipment_id column
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_state: Random seed

        Returns:
            Dictionary with train/val/test DataFrames
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
            "Ratios must sum to 1.0"
        )

        # Get unique equipment IDs
        equipment_ids = df["equipment_id"].unique()
        np.random.seed(random_state)
        np.random.shuffle(equipment_ids)

        # Calculate split indices
        n_total = len(equipment_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split equipment IDs
        train_ids = equipment_ids[:n_train]
        val_ids = equipment_ids[n_train : n_train + n_val]
        test_ids = equipment_ids[n_train + n_val :]

        # Split dataframe
        train_df = df[df["equipment_id"].isin(train_ids)]
        val_df = df[df["equipment_id"].isin(val_ids)]
        test_df = df[df["equipment_id"].isin(test_ids)]

        logger.info(
            f"Split data: Train={len(train_df)} ({len(train_ids)} equipment), "
            f"Val={len(val_df)} ({len(val_ids)} equipment), "
            f"Test={len(test_df)} ({len(test_ids)} equipment)"
        )

        return {"train": train_df, "val": val_df, "test": test_df}

    @staticmethod
    def split_by_time(
        df: pd.DataFrame, train_end_cycle: int, val_end_cycle: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data by cycle/time

        Args:
            df: DataFrame with cycle column
            train_end_cycle: Last cycle for training
            val_end_cycle: Last cycle for validation

        Returns:
            Dictionary with train/val/test DataFrames
        """
        train_df = df[df["cycle"] <= train_end_cycle]
        val_df = df[(df["cycle"] > train_end_cycle) & (df["cycle"] <= val_end_cycle)]
        test_df = df[df["cycle"] > val_end_cycle]

        logger.info(
            f"Split by time: Train={len(train_df)}, "
            f"Val={len(val_df)}, Test={len(test_df)}"
        )

        return {"train": train_df, "val": val_df, "test": test_df}
