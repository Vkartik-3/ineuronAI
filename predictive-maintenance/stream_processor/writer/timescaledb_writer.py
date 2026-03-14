"""
TimescaleDB Writer
Handles batch writes to TimescaleDB hypertables.
Matches db-schema.md table definitions.
"""

import psycopg2
from psycopg2 import pool, extras
from typing import Dict, List, Optional
import json
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


class TimescaleDBWriter:
    """Writes sensor data and features to TimescaleDB"""

    def __init__(self, config: Dict):
        self.config = config
        db_config = config.get("timescaledb", {})

        self.host = db_config.get("host", "localhost")
        self.port = db_config.get("port", 5432)
        self.database = db_config.get("database", "predictive_maintenance")
        self.user = db_config.get("user", "postgres")
        self.password = db_config.get("password", "postgres")

        pool_config = db_config.get("connection_pool", {})
        self.pool_min_conn = pool_config.get("min_size", 2)
        self.pool_max_conn = pool_config.get("max_size", 10)

        batch_config = db_config.get("batch_write", {})
        self.batch_size = batch_config.get("batch_size", 100)
        self.batch_timeout = batch_config.get("timeout_seconds", 5)

        self.connection_pool = None
        self._init_connection_pool()
        self._initialize_database()

        # Batch buffers
        self.sensor_readings_buffer: List[tuple] = []
        self.engineered_features_buffer: List[tuple] = []

        logger.info(
            f"TimescaleDB writer initialized. "
            f"Host: {self.host}:{self.port}, Database: {self.database}"
        )

    def _init_connection_pool(self):
        """Initialize connection pool with retry logic"""
        max_retries = 5
        for attempt in range(max_retries):
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
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        f"DB connection attempt {attempt + 1} failed, retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"Failed to create connection pool after {max_retries} attempts: {e}"
                    )
                    raise

    def _initialize_database(self):
        """
        Defensive fallback: ensure tables and hypertables exist.
        Primary creation is via init-db SQL scripts; this is a safety net.
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Ensure TimescaleDB extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            # sensor_readings — matches db-schema.md
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    time                TIMESTAMPTZ NOT NULL,
                    equipment_id        VARCHAR(64) NOT NULL,
                    cycle               INTEGER,
                    data_source         VARCHAR(32),
                    operational_settings JSONB,
                    sensor_readings     JSONB NOT NULL,
                    quality_flag        VARCHAR(16) DEFAULT 'ok',
                    metadata            JSONB,
                    ingested_at         TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cursor.execute(
                "SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);"
            )

            # engineered_features — matches db-schema.md
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engineered_features (
                    time                TIMESTAMPTZ NOT NULL,
                    equipment_id        VARCHAR(64) NOT NULL,
                    feature_set         VARCHAR(32) NOT NULL,
                    features            JSONB NOT NULL,
                    window_size         INTEGER,
                    computation_time_ms FLOAT,
                    source              VARCHAR(32),
                    created_at          TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cursor.execute(
                "SELECT create_hypertable('engineered_features', 'time', if_not_exists => TRUE);"
            )

            # predictions — matches db-schema.md
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    time                TIMESTAMPTZ NOT NULL,
                    equipment_id        VARCHAR(64) NOT NULL,
                    prediction_type     VARCHAR(32) NOT NULL,
                    rul_cycles          FLOAT,
                    rul_hours           FLOAT,
                    confidence          FLOAT,
                    confidence_lower    FLOAT,
                    confidence_upper    FLOAT,
                    health_status       VARCHAR(32),
                    health_probabilities JSONB,
                    anomaly_score       FLOAT,
                    model_name          VARCHAR(64),
                    model_version       VARCHAR(32),
                    inference_time_ms   FLOAT,
                    input_features      JSONB,
                    created_at          TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cursor.execute(
                "SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);"
            )

            # alerts — matches db-schema.md
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id            VARCHAR(128) PRIMARY KEY,
                    equipment_id        VARCHAR(64) NOT NULL,
                    rule_id             VARCHAR(64) NOT NULL,
                    severity            VARCHAR(16) NOT NULL,
                    message             TEXT NOT NULL,
                    status              VARCHAR(16) NOT NULL DEFAULT 'triggered',
                    data                JSONB,
                    notifications_sent  JSONB,
                    triggered_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    acknowledged_by     VARCHAR(128),
                    acknowledged_at     TIMESTAMPTZ,
                    resolved_at         TIMESTAMPTZ,
                    suppressed          BOOLEAN DEFAULT FALSE,
                    suppress_reason     VARCHAR(256)
                );
            """)

            conn.commit()
            logger.info("Database tables verified/created (defensive fallback)")

        except Exception as e:
            logger.warning(f"Database initialization fallback encountered issue: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.return_connection(conn)

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
        cycle: Optional[int] = None,
        data_source: str = "cmapss",
        operational_settings: Optional[Dict] = None,
        quality_flag: str = "ok",
    ):
        """
        Buffer a sensor reading for batch insert into sensor_readings table.
        Columns: (time, equipment_id, cycle, data_source,
                  operational_settings, sensor_readings, quality_flag, metadata)
        """
        reading = (
            timestamp,
            equipment_id,
            cycle,
            data_source,
            json.dumps(operational_settings) if operational_settings else None,
            json.dumps(sensor_data),
            quality_flag,
            json.dumps(metadata) if metadata else None,
        )

        self.sensor_readings_buffer.append(reading)

        if len(self.sensor_readings_buffer) >= self.batch_size:
            self.flush_sensor_readings()

    def write_engineered_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_set: str = "time_domain",
        window_size: Optional[int] = None,
        computation_time_ms: Optional[float] = None,
        source: str = "stream_processor",
    ):
        """
        Buffer an engineered-feature row for batch insert into engineered_features.
        Columns: (time, equipment_id, feature_set, features,
                  window_size, computation_time_ms, source)
        """
        record = (
            timestamp,
            equipment_id,
            feature_set,
            json.dumps(features),
            window_size,
            computation_time_ms,
            source,
        )

        self.engineered_features_buffer.append(record)

        if len(self.engineered_features_buffer) >= self.batch_size:
            self.flush_engineered_features()

    # Keep backwards-compatible alias
    def write_processed_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_type: str = "time_domain",
    ):
        """Backwards-compatible wrapper — delegates to write_engineered_features."""
        self.write_engineered_features(
            equipment_id=equipment_id,
            timestamp=timestamp,
            features=features,
            feature_set=feature_type,
        )

    # ------------------------------------------------------------------
    # Flush helpers
    # ------------------------------------------------------------------

    def flush_sensor_readings(self):
        """Flush sensor readings buffer to sensor_readings hypertable."""
        if not self.sensor_readings_buffer:
            return

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO sensor_readings
                (time, equipment_id, cycle, data_source,
                 operational_settings, sensor_readings, quality_flag, metadata)
                VALUES %s
            """

            extras.execute_values(
                cursor,
                insert_query,
                self.sensor_readings_buffer,
                page_size=self.batch_size,
            )

            conn.commit()
            logger.debug(f"Flushed {len(self.sensor_readings_buffer)} sensor readings")
            self.sensor_readings_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing sensor readings: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.return_connection(conn)

    def flush_engineered_features(self):
        """Flush engineered features buffer to engineered_features hypertable."""
        if not self.engineered_features_buffer:
            return

        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO engineered_features
                (time, equipment_id, feature_set, features,
                 window_size, computation_time_ms, source)
                VALUES %s
            """

            extras.execute_values(
                cursor,
                insert_query,
                self.engineered_features_buffer,
                page_size=self.batch_size,
            )

            conn.commit()
            logger.debug(
                f"Flushed {len(self.engineered_features_buffer)} engineered features"
            )
            self.engineered_features_buffer.clear()

        except Exception as e:
            logger.error(f"Error flushing engineered features: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self.return_connection(conn)

    # Backwards-compatible alias
    def flush_processed_features(self):
        """Alias for flush_engineered_features."""
        self.flush_engineered_features()

    def flush_all(self):
        """Flush all buffers."""
        self.flush_sensor_readings()
        self.flush_engineered_features()

    # ------------------------------------------------------------------
    # Direct writes (predictions, alerts)
    # ------------------------------------------------------------------

    def write_prediction(
        self,
        equipment_id: str,
        timestamp: datetime,
        prediction_type: str,
        confidence: float,
        rul_cycles: Optional[float] = None,
        rul_hours: Optional[float] = None,
        confidence_lower: Optional[float] = None,
        confidence_upper: Optional[float] = None,
        health_status: Optional[str] = None,
        health_probabilities: Optional[Dict] = None,
        anomaly_score: Optional[float] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        inference_time_ms: Optional[float] = None,
        input_features: Optional[Dict] = None,
    ):
        """
        Write a single prediction row to the predictions hypertable.
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO predictions
                (time, equipment_id, prediction_type,
                 rul_cycles, rul_hours, confidence,
                 confidence_lower, confidence_upper,
                 health_status, health_probabilities,
                 anomaly_score, model_name, model_version,
                 inference_time_ms, input_features)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            cursor.execute(
                insert_query,
                (
                    timestamp,
                    equipment_id,
                    prediction_type,
                    rul_cycles,
                    rul_hours,
                    confidence,
                    confidence_lower,
                    confidence_upper,
                    health_status,
                    json.dumps(health_probabilities) if health_probabilities else None,
                    anomaly_score,
                    model_name,
                    model_version,
                    inference_time_ms,
                    json.dumps(input_features) if input_features else None,
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
        alert_id: str,
        equipment_id: str,
        rule_id: str,
        severity: str,
        message: str,
        status: str = "triggered",
        data: Optional[Dict] = None,
        notifications_sent: Optional[Dict] = None,
    ):
        """
        Write a single alert row to the alerts table.
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            insert_query = """
                INSERT INTO alerts
                (alert_id, equipment_id, rule_id, severity, message,
                 status, data, notifications_sent, triggered_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (alert_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    data = EXCLUDED.data,
                    notifications_sent = EXCLUDED.notifications_sent
            """

            cursor.execute(
                insert_query,
                (
                    alert_id,
                    equipment_id,
                    rule_id,
                    severity,
                    message,
                    status,
                    json.dumps(data) if data else None,
                    json.dumps(notifications_sent) if notifications_sent else None,
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

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_latest_features(
        self, equipment_id: str, feature_set: str, limit: int = 10
    ) -> List[Dict]:
        """
        Query latest engineered features for a given equipment & feature_set.
        """
        conn = None
        try:
            conn = self.get_connection()
            cursor = conn.cursor(cursor_factory=extras.RealDictCursor)

            query = """
                SELECT time, equipment_id, feature_set, features,
                       window_size, computation_time_ms, source
                FROM engineered_features
                WHERE equipment_id = %s AND feature_set = %s
                ORDER BY time DESC
                LIMIT %s
            """

            cursor.execute(query, (equipment_id, feature_set, limit))
            results = cursor.fetchall()

            return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Error querying features: {e}")
            return []
        finally:
            if conn:
                self.return_connection(conn)

    def close(self):
        """Close connection pool."""
        try:
            self.flush_all()
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
        self.engineered_features: List[Dict] = []
        self.predictions: List[Dict] = []
        self.alerts: List[Dict] = []
        logger.info("Mock TimescaleDB writer initialized")

    def write_sensor_reading(
        self,
        equipment_id: str,
        timestamp: datetime,
        sensor_data: Dict,
        metadata: Optional[Dict] = None,
        cycle: Optional[int] = None,
        data_source: str = "cmapss",
        operational_settings: Optional[Dict] = None,
        quality_flag: str = "ok",
    ):
        self.sensor_readings.append(
            {
                "time": timestamp,
                "equipment_id": equipment_id,
                "cycle": cycle,
                "data_source": data_source,
                "operational_settings": operational_settings,
                "sensor_readings": sensor_data,
                "quality_flag": quality_flag,
                "metadata": metadata,
            }
        )

    def write_engineered_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_set: str = "time_domain",
        window_size: Optional[int] = None,
        computation_time_ms: Optional[float] = None,
        source: str = "stream_processor",
    ):
        self.engineered_features.append(
            {
                "time": timestamp,
                "equipment_id": equipment_id,
                "feature_set": feature_set,
                "features": features,
                "window_size": window_size,
                "computation_time_ms": computation_time_ms,
                "source": source,
            }
        )

    def write_processed_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_type: str = "time_domain",
    ):
        """Backwards-compatible alias."""
        self.write_engineered_features(
            equipment_id=equipment_id,
            timestamp=timestamp,
            features=features,
            feature_set=feature_type,
        )

    def write_prediction(
        self,
        equipment_id: str,
        timestamp: datetime,
        prediction_type: str,
        confidence: float,
        rul_cycles: Optional[float] = None,
        rul_hours: Optional[float] = None,
        confidence_lower: Optional[float] = None,
        confidence_upper: Optional[float] = None,
        health_status: Optional[str] = None,
        health_probabilities: Optional[Dict] = None,
        anomaly_score: Optional[float] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        inference_time_ms: Optional[float] = None,
        input_features: Optional[Dict] = None,
    ):
        self.predictions.append(
            {
                "time": timestamp,
                "equipment_id": equipment_id,
                "prediction_type": prediction_type,
                "rul_cycles": rul_cycles,
                "confidence": confidence,
                "health_status": health_status,
                "model_name": model_name,
                "model_version": model_version,
            }
        )

    def write_alert(
        self,
        alert_id: str,
        equipment_id: str,
        rule_id: str,
        severity: str,
        message: str,
        status: str = "triggered",
        data: Optional[Dict] = None,
        notifications_sent: Optional[Dict] = None,
    ):
        self.alerts.append(
            {
                "alert_id": alert_id,
                "equipment_id": equipment_id,
                "rule_id": rule_id,
                "severity": severity,
                "message": message,
                "status": status,
                "data": data,
            }
        )

    def flush_sensor_readings(self):
        logger.debug(f"Mock flush: {len(self.sensor_readings)} sensor readings")

    def flush_engineered_features(self):
        logger.debug(f"Mock flush: {len(self.engineered_features)} features")

    def flush_processed_features(self):
        self.flush_engineered_features()

    def flush_all(self):
        self.flush_sensor_readings()
        self.flush_engineered_features()

    def query_latest_features(
        self, equipment_id: str, feature_set: str, limit: int = 10
    ) -> List[Dict]:
        filtered = [
            f
            for f in self.engineered_features
            if f["equipment_id"] == equipment_id and f["feature_set"] == feature_set
        ]
        return filtered[-limit:]

    def close(self):
        logger.info(
            f"Mock writer closed. "
            f"Readings: {len(self.sensor_readings)}, "
            f"Features: {len(self.engineered_features)}, "
            f"Predictions: {len(self.predictions)}, "
            f"Alerts: {len(self.alerts)}"
        )
