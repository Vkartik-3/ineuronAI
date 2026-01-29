"""
TimescaleDB Writer
Handles batch writes to TimescaleDB hypertables
"""

import psycopg2
from psycopg2 import pool, extras
from typing import Dict, List, Optional
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TimescaleDBWriter:
    """Writes sensor data and features to TimescaleDB"""

    def __init__(self, config: Dict):
        """
        Initialize database writer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        db_config = config.get("timescaledb", {})

        # Connection parameters
        self.host = db_config.get("host", "localhost")
        self.port = db_config.get("port", 5432)
        self.database = db_config.get("database", "predictive_maintenance")
        self.user = db_config.get("user", "postgres")
        self.password = db_config.get("password", "postgres")

        # Connection pool settings
        pool_config = db_config.get("connection_pool", {})
        self.pool_min_conn = pool_config.get("min_size", 2)
        self.pool_max_conn = pool_config.get("max_size", 10)

        # Batch settings
        batch_config = db_config.get("batch_write", {})
        self.batch_size = batch_config.get("batch_size", 100)
        self.batch_timeout = batch_config.get("timeout_seconds", 5)

        # Initialize connection pool
        self.connection_pool = None
        self._init_connection_pool()

        # Batch buffers
        self.sensor_readings_buffer: List[tuple] = []
        self.processed_features_buffer: List[tuple] = []

        logger.info(
            f"TimescaleDB writer initialized. "
            f"Host: {self.host}:{self.port}, Database: {self.database}"
        )

    def _init_connection_pool(self):
        """Initialize connection pool"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                self.pool_min_conn,
                self.pool_max_conn,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            logger.info("Connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise

    def get_connection(self):
        """Get connection from pool"""
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {e}")
            raise

    def return_connection(self, conn):
        """Return connection to pool"""
        try:
            self.connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"Failed to return connection to pool: {e}")

    def write_sensor_reading(
        self,
        equipment_id: str,
        timestamp: datetime,
        sensor_data: Dict,
        metadata: Optional[Dict] = None,
    ):
        """
        Write sensor reading to buffer

        Args:
            equipment_id: Equipment identifier
            timestamp: Reading timestamp
            sensor_data: Dictionary of sensor values
            metadata: Optional metadata
        """
        reading = (
            equipment_id,
            timestamp,
            json.dumps(sensor_data),
            json.dumps(metadata) if metadata else None,
        )

        self.sensor_readings_buffer.append(reading)

        # Flush if batch size reached
        if len(self.sensor_readings_buffer) >= self.batch_size:
            self.flush_sensor_readings()

    def write_processed_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_type: str = "time_domain",
    ):
        """
        Write processed features to buffer

        Args:
            equipment_id: Equipment identifier
            timestamp: Feature timestamp
            features: Dictionary of feature values
            feature_type: Type of features (time_domain, frequency_domain, etc.)
        """
        feature_record = (equipment_id, timestamp, feature_type, json.dumps(features))

        self.processed_features_buffer.append(feature_record)

        # Flush if batch size reached
        if len(self.processed_features_buffer) >= self.batch_size:
            self.flush_processed_features()

    def flush_sensor_readings(self):
        """Flush sensor readings buffer to database"""
        if not self.sensor_readings_buffer:
            return

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Use execute_values for efficient batch insert
            insert_query = """
                INSERT INTO sensor_readings
                (equipment_id, timestamp, sensor_data, metadata)
                VALUES %s
                ON CONFLICT (equipment_id, timestamp)
                DO UPDATE SET
                    sensor_data = EXCLUDED.sensor_data,
                    metadata = EXCLUDED.metadata
            """

            extras.execute_values(
                cursor,
                insert_query,
                self.sensor_readings_buffer,
                page_size=self.batch_size,
            )

            conn.commit()
            logger.debug(f"Flushed {len(self.sensor_readings_buffer)} sensor readings")

            # Clear buffer
            self.sensor_readings_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing sensor readings: {e}")
            if conn:
                conn.rollback()

        finally:
            if conn:
                self.return_connection(conn)

    def flush_processed_features(self):
        """Flush processed features buffer to database"""
        if not self.processed_features_buffer:
            return

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO processed_features
                (equipment_id, timestamp, feature_type, features)
                VALUES %s
                ON CONFLICT (equipment_id, timestamp, feature_type)
                DO UPDATE SET
                    features = EXCLUDED.features
            """

            extras.execute_values(
                cursor,
                insert_query,
                self.processed_features_buffer,
                page_size=self.batch_size,
            )

            conn.commit()
            logger.debug(
                f"Flushed {len(self.processed_features_buffer)} processed features"
            )

            # Clear buffer
            self.processed_features_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing processed features: {e}")
            if conn:
                conn.rollback()

        finally:
            if conn:
                self.return_connection(conn)

    def flush_all(self):
        """Flush all buffers"""
        self.flush_sensor_readings()
        self.flush_processed_features()

    def write_prediction(
        self,
        equipment_id: str,
        timestamp: datetime,
        model_id: str,
        prediction_type: str,
        predicted_value: float,
        confidence: float,
        rul_estimate: Optional[int] = None,
    ):
        """
        Write prediction to database

        Args:
            equipment_id: Equipment identifier
            timestamp: Prediction timestamp
            model_id: Model identifier
            prediction_type: Type of prediction
            predicted_value: Predicted value
            confidence: Confidence score
            rul_estimate: Remaining useful life estimate
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO predictions
                (equipment_id, timestamp, model_id, prediction_type,
                 predicted_value, confidence, rul_estimate)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """

            cursor.execute(
                insert_query,
                (
                    equipment_id,
                    timestamp,
                    model_id,
                    prediction_type,
                    predicted_value,
                    confidence,
                    rul_estimate,
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error writing prediction: {e}")
            if conn:
                conn.rollback()

        finally:
            if conn:
                self.return_connection(conn)

    def write_alert(
        self,
        equipment_id: str,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None,
    ):
        """
        Write maintenance alert

        Args:
            equipment_id: Equipment identifier
            alert_type: Type of alert
            severity: Severity level
            message: Alert message
            details: Additional details
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO maintenance_alerts
                (equipment_id, alert_type, severity, message, details)
                VALUES (%s, %s, %s, %s, %s)
            """

            cursor.execute(
                insert_query,
                (
                    equipment_id,
                    alert_type,
                    severity,
                    message,
                    json.dumps(details) if details else None,
                ),
            )

            conn.commit()

        except Exception as e:
            logger.error(f"Error writing alert: {e}")
            if conn:
                conn.rollback()

        finally:
            if conn:
                self.return_connection(conn)

    def query_latest_features(
        self, equipment_id: str, feature_type: str, limit: int = 10
    ) -> List[Dict]:
        """
        Query latest features for equipment

        Args:
            equipment_id: Equipment identifier
            feature_type: Type of features
            limit: Number of records to retrieve

        Returns:
            List of feature records
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)

            query = """
                SELECT equipment_id, timestamp, feature_type, features
                FROM processed_features
                WHERE equipment_id = %s AND feature_type = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """

            cursor.execute(query, (equipment_id, feature_type, limit))
            results = cursor.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error querying features: {e}")
            return []

        finally:
            if conn:
                self.return_connection(conn)

    def close(self):
        """Close connection pool"""
        try:
            # Flush any remaining data
            self.flush_all()

            # Close pool
            if self.connection_pool:
                self.connection_pool.closeall()
                logger.info("Connection pool closed")

        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")


class MockTimescaleDBWriter:
    """Mock writer for testing without database"""

    def __init__(self, config: Dict):
        self.config = config
        self.sensor_readings: List[Dict] = []
        self.processed_features: List[Dict] = []
        self.predictions: List[Dict] = []
        self.alerts: List[Dict] = []
        logger.info("Mock TimescaleDB writer initialized")

    def write_sensor_reading(
        self,
        equipment_id: str,
        timestamp: datetime,
        sensor_data: Dict,
        metadata: Optional[Dict] = None,
    ):
        self.sensor_readings.append(
            {
                "equipment_id": equipment_id,
                "timestamp": timestamp,
                "sensor_data": sensor_data,
                "metadata": metadata,
            }
        )

    def write_processed_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_type: str = "time_domain",
    ):
        self.processed_features.append(
            {
                "equipment_id": equipment_id,
                "timestamp": timestamp,
                "feature_type": feature_type,
                "features": features,
            }
        )

    def write_prediction(
        self,
        equipment_id: str,
        timestamp: datetime,
        model_id: str,
        prediction_type: str,
        predicted_value: float,
        confidence: float,
        rul_estimate: Optional[int] = None,
    ):
        self.predictions.append(
            {
                "equipment_id": equipment_id,
                "timestamp": timestamp,
                "model_id": model_id,
                "prediction_type": prediction_type,
                "predicted_value": predicted_value,
                "confidence": confidence,
                "rul_estimate": rul_estimate,
            }
        )

    def write_alert(
        self,
        equipment_id: str,
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict] = None,
    ):
        self.alerts.append(
            {
                "equipment_id": equipment_id,
                "alert_type": alert_type,
                "severity": severity,
                "message": message,
                "details": details,
            }
        )

    def flush_sensor_readings(self):
        logger.debug(f"Mock flush: {len(self.sensor_readings)} sensor readings")

    def flush_processed_features(self):
        logger.debug(f"Mock flush: {len(self.processed_features)} features")

    def flush_all(self):
        self.flush_sensor_readings()
        self.flush_processed_features()

    def query_latest_features(
        self, equipment_id: str, feature_type: str, limit: int = 10
    ) -> List[Dict]:
        filtered = [
            f
            for f in self.processed_features
            if f["equipment_id"] == equipment_id and f["feature_type"] == feature_type
        ]
        return filtered[-limit:]

    def close(self):
        logger.info(
            f"Mock writer closed. "
            f"Readings: {len(self.sensor_readings)}, "
            f"Features: {len(self.processed_features)}, "
            f"Predictions: {len(self.predictions)}, "
            f"Alerts: {len(self.alerts)}"
        )
