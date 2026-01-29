"""
Feature Store Database Manager
Manages feature storage and retrieval from TimescaleDB
"""

import psycopg2
from psycopg2 import pool, extras
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class FeatureStoreDB:
    """Manages feature storage in TimescaleDB"""

    def __init__(self, config: Dict):
        """
        Initialize feature store database manager

        Args:
            config: Configuration dictionary
        """
        self.config = config
        db_config = config.get("timescaledb", {})

        # Connection parameters
        self.host = db_config.get("host", "localhost")
        self.port = db_config.get("port", 5432)
        self.database = db_config.get("database", "predictive_maintenance")
        self.user = db_config.get("user", "pmuser")
        self.password = db_config.get("password", "pmpassword")

        # Connection pool
        pool_config = db_config.get("connection_pool", {})
        self.pool_min = pool_config.get("min_size", 2)
        self.pool_max = pool_config.get("max_size", 10)

        self.connection_pool = None
        self._init_connection_pool()

        logger.info(f"Feature store DB initialized. Host: {self.host}:{self.port}")

    def _init_connection_pool(self):
        """Initialize connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                self.pool_min,
                self.pool_max,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            logger.info("Connection pool created")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

    def get_connection(self):
        """Get connection from pool"""
        return self.connection_pool.getconn()

    def return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.putconn(conn)

    def create_feature_store_table(self):
        """Create feature store table"""
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            create_table_sql = """
            CREATE TABLE IF NOT EXISTS feature_store (
                time TIMESTAMPTZ NOT NULL,
                equipment_id VARCHAR(50) NOT NULL,
                cycle INTEGER,

                -- Raw sensor features
                sensor_features JSONB,

                -- Engineered features
                time_series_features JSONB,
                frequency_features JSONB,
                statistical_features JSONB,

                -- Labels
                rul DOUBLE PRECISION,
                rul_normalized DOUBLE PRECISION,
                failure_imminent INTEGER,
                health_status VARCHAR(20),
                health_status_code INTEGER,
                degradation_rate DOUBLE PRECISION,

                -- Metadata
                feature_version VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW(),

                PRIMARY KEY (time, equipment_id)
            );
            """

            cursor.execute(create_table_sql)

            # Convert to hypertable
            hypertable_sql = """
            SELECT create_hypertable('feature_store', 'time',
                                    if_not_exists => TRUE);
            """
            cursor.execute(hypertable_sql)

            # Create indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_feature_store_equipment ON feature_store(equipment_id, time DESC);",
                "CREATE INDEX IF NOT EXISTS idx_feature_store_cycle ON feature_store(cycle);",
                "CREATE INDEX IF NOT EXISTS idx_feature_store_health ON feature_store(health_status, time DESC);",
            ]

            for index_sql in indexes:
                cursor.execute(index_sql)

            conn.commit()
            logger.info("Feature store table created")

        except Exception as e:
            logger.error(f"Error creating feature store table: {e}")
            if conn:
                conn.rollback()

        finally:
            if conn:
                self.return_connection(conn)

    def insert_features(
        self, features_df: pd.DataFrame, feature_version: str = "v1.0.0"
    ):
        """
        Insert features into feature store

        Args:
            features_df: DataFrame with features
            feature_version: Version identifier
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_sql = """
            INSERT INTO feature_store (
                time, equipment_id, cycle,
                sensor_features, time_series_features, frequency_features,
                rul, rul_normalized, failure_imminent,
                health_status, health_status_code, degradation_rate,
                feature_version
            ) VALUES %s
            ON CONFLICT (time, equipment_id)
            DO UPDATE SET
                cycle = EXCLUDED.cycle,
                sensor_features = EXCLUDED.sensor_features,
                time_series_features = EXCLUDED.time_series_features,
                frequency_features = EXCLUDED.frequency_features,
                rul = EXCLUDED.rul,
                rul_normalized = EXCLUDED.rul_normalized,
                failure_imminent = EXCLUDED.failure_imminent,
                health_status = EXCLUDED.health_status,
                health_status_code = EXCLUDED.health_status_code,
                degradation_rate = EXCLUDED.degradation_rate,
                feature_version = EXCLUDED.feature_version
            """

            # Prepare data for insertion
            records = []
            for _, row in features_df.iterrows():
                record = (
                    row.get("time") or row.get("timestamp"),
                    row.get("equipment_id"),
                    row.get("cycle"),
                    json.dumps({}),  # sensor_features
                    json.dumps({}),  # time_series_features
                    json.dumps({}),  # frequency_features
                    row.get("rul"),
                    row.get("rul_normalized"),
                    row.get("failure_imminent"),
                    row.get("health_status"),
                    row.get("health_status_code"),
                    row.get("degradation_rate"),
                    feature_version,
                )
                records.append(record)

            # Batch insert
            extras.execute_values(cursor, insert_sql, records, page_size=1000)

            conn.commit()
            logger.info(f"Inserted {len(records)} feature records")

        except Exception as e:
            logger.error(f"Error inserting features: {e}")
            if conn:
                conn.rollback()

        finally:
            if conn:
                self.return_connection(conn)

    def fetch_features(
        self,
        equipment_ids: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000,
    ) -> pd.DataFrame:
        """
        Fetch features from feature store

        Args:
            equipment_ids: List of equipment IDs to filter
            start_time: Start time for filtering
            end_time: End time for filtering
            limit: Maximum number of records

        Returns:
            DataFrame with features
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)

            # Build query
            query = "SELECT * FROM feature_store WHERE 1=1"
            params = []

            if equipment_ids:
                query += f" AND equipment_id = ANY(%s)"
                params.append(equipment_ids)

            if start_time:
                query += " AND time >= %s"
                params.append(start_time)

            if end_time:
                query += " AND time <= %s"
                params.append(end_time)

            query += " ORDER BY time DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()

            # Convert to DataFrame
            df = pd.DataFrame(results)

            logger.info(f"Fetched {len(df)} feature records")

            return df

        except Exception as e:
            logger.error(f"Error fetching features: {e}")
            return pd.DataFrame()

        finally:
            if conn:
                self.return_connection(conn)

    def fetch_training_data(
        self,
        equipment_ids: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "rul",
    ) -> tuple:
        """
        Fetch training data (X, y)

        Args:
            equipment_ids: List of equipment IDs
            feature_cols: List of feature column names
            target_col: Target column name

        Returns:
            Tuple of (X, y) as DataFrames
        """
        df = self.fetch_features(equipment_ids=equipment_ids)

        if df.empty:
            return pd.DataFrame(), pd.Series()

        # Extract features and target
        if feature_cols:
            X = df[feature_cols]
        else:
            # Exclude metadata columns
            exclude_cols = [
                "time",
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
            ]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            X = df[feature_cols]

        y = df[target_col]

        return X, y

    def close(self):
        """Close connection pool"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Connection pool closed")
