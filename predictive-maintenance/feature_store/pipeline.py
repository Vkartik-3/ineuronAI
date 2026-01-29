"""
Feature Store Pipeline
Orchestrates feature extraction and storage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import yaml

from features.time_series_features import TimeSeriesFeatureExtractor, SequenceGenerator
from features.frequency_features import FrequencyFeatureExtractor
from features.label_generator import LabelGenerator, DatasetSplitter
from storage.feature_store_db import FeatureStoreDB
from storage.feature_cache import FeatureCache

logger = logging.getLogger(__name__)


class FeatureStorePipeline:
    """Main pipeline for feature engineering and storage"""

    def __init__(self, config_path: str):
        """
        Initialize feature store pipeline

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.ts_extractor = TimeSeriesFeatureExtractor(self.config)
        self.freq_extractor = FrequencyFeatureExtractor(self.config)
        self.label_generator = LabelGenerator(self.config)

        # Storage components
        self.db = FeatureStoreDB(self.config)
        self.cache = FeatureCache(self.config)

        # Sequence generator for LSTM
        self.sequence_generator = SequenceGenerator(sequence_length=50, stride=1)

        logger.info("Feature store pipeline initialized")

    def process_equipment_data(
        self, df: pd.DataFrame, equipment_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process sensor data and extract features

        Args:
            df: DataFrame with raw sensor data
            equipment_id: Optional equipment identifier

        Returns:
            DataFrame with engineered features and labels
        """
        logger.info(f"Processing data for equipment: {equipment_id or 'all'}")

        # Filter by equipment if specified
        if equipment_id and "equipment_id" in df.columns:
            df = df[df["equipment_id"] == equipment_id].copy()

        # Sort by time/cycle
        if "time" in df.columns:
            df = df.sort_values("time")
        elif "timestamp" in df.columns:
            df = df.sort_values("timestamp")
        elif "cycle" in df.columns:
            df = df.sort_values("cycle")

        # Extract time-series features
        logger.debug("Extracting time-series features...")
        df = self.ts_extractor.extract_features(df, equipment_id)

        # Extract frequency-domain features
        logger.debug("Extracting frequency-domain features...")
        df = self.freq_extractor.extract_features(df, equipment_id)

        # Generate labels
        logger.debug("Generating labels...")
        df = self.label_generator.generate_labels(df)

        logger.info(f"Processed {len(df)} records with {len(df.columns)} features")

        return df

    def process_and_store(self, df: pd.DataFrame, feature_version: str = "v1.0.0"):
        """
        Process features and store in database

        Args:
            df: DataFrame with raw sensor data
            feature_version: Version identifier
        """
        # Process features
        processed_df = self.process_equipment_data(df)

        # Store in database
        logger.info("Storing features in database...")
        self.db.insert_features(processed_df, feature_version)

        # Cache recent features
        if "equipment_id" in processed_df.columns and "cycle" in processed_df.columns:
            for equipment_id in processed_df["equipment_id"].unique():
                eq_data = processed_df[processed_df["equipment_id"] == equipment_id]
                latest_cycle = eq_data["cycle"].max()
                latest_features = (
                    eq_data[eq_data["cycle"] == latest_cycle].iloc[0].to_dict()
                )
                self.cache.cache_features(equipment_id, latest_cycle, latest_features)

        logger.info("Feature processing and storage complete")

    def load_from_timescaledb(
        self,
        start_cycle: Optional[int] = None,
        end_cycle: Optional[int] = None,
        equipment_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load sensor data from TimescaleDB

        Args:
            start_cycle: Starting cycle
            end_cycle: Ending cycle
            equipment_ids: List of equipment IDs

        Returns:
            DataFrame with sensor readings
        """
        logger.info("Loading data from TimescaleDB...")

        # Fetch from sensor_readings table
        df = self.db.fetch_features(equipment_ids=equipment_ids)

        # Filter by cycle
        if start_cycle is not None:
            df = df[df["cycle"] >= start_cycle]
        if end_cycle is not None:
            df = df[df["cycle"] <= end_cycle]

        logger.info(f"Loaded {len(df)} records from TimescaleDB")

        return df

    def create_training_dataset(
        self,
        equipment_ids: List[str],
        sequence_length: int = 50,
        target_col: str = "rul",
    ) -> Dict:
        """
        Create training dataset with sequences for LSTM

        Args:
            equipment_ids: List of equipment IDs for training
            sequence_length: Length of LSTM sequences
            target_col: Target column name

        Returns:
            Dictionary with X_train, y_train, feature_names
        """
        logger.info(f"Creating training dataset for {len(equipment_ids)} equipment...")

        # Fetch features from database
        df = self.db.fetch_features(equipment_ids=equipment_ids)

        if df.empty:
            logger.error("No data found for training")
            return {}

        # Get feature columns (exclude metadata)
        exclude_cols = [
            "time",
            "timestamp",
            "equipment_id",
            "cycle",
            "rul",
            "rul_normalized",
            "failure_imminent",
            "health_status",
            "health_status_code",
            "degradation_rate",
            "feature_version",
            "created_at",
            "sensor_features",
            "time_series_features",
            "frequency_features",
            "statistical_features",
        ]

        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and not col.startswith("cycle_bin")
        ]

        # Generate sequences
        self.sequence_generator.sequence_length = sequence_length
        X, y, eq_ids = self.sequence_generator.generate_sequences_per_equipment(
            df, feature_cols, target_col
        )

        logger.info(f"Created training dataset: X shape {X.shape}, y shape {y.shape}")

        return {
            "X": X,
            "y": y,
            "equipment_ids": eq_ids,
            "feature_names": feature_cols,
            "sequence_length": sequence_length,
        }

    def split_train_test(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/val/test sets

        Args:
            df: DataFrame with features
            train_ratio: Training proportion
            val_ratio: Validation proportion
            test_ratio: Test proportion

        Returns:
            Dictionary with train/val/test DataFrames
        """
        splitter = DatasetSplitter()
        return splitter.split_by_equipment(df, train_ratio, val_ratio, test_ratio)

    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get feature statistics

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with statistics
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        stats = {
            "n_records": len(df),
            "n_features": len(numeric_cols),
            "n_equipment": df["equipment_id"].nunique()
            if "equipment_id" in df.columns
            else 0,
            "feature_means": df[numeric_cols].mean().to_dict(),
            "feature_stds": df[numeric_cols].std().to_dict(),
            "missing_values": df[numeric_cols].isnull().sum().to_dict(),
        }

        return stats

    def close(self):
        """Close connections"""
        self.db.close()
        logger.info("Feature store pipeline closed")
