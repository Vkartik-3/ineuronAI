"""
Kafka Prediction Pipeline for the Inference Service.

Consumes features from the ``processed_features`` topic, runs inference,
publishes predictions to ``failure_predictions``, and writes them to
TimescaleDB.

Designed to run **alongside** the FastAPI server in a background thread so
the REST API remains available for on-demand predictions.
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import psycopg2
from psycopg2 import pool as pg_pool

logger = logging.getLogger(__name__)


class KafkaPredictionPipeline:
    """End-to-end Kafka consume → predict → produce → persist pipeline."""

    def __init__(self, config: Dict, model_manager: Any, inference_engine: Any):
        self._config = config
        self._model_manager = model_manager
        self._inference_engine = inference_engine

        kafka_cfg = config.get("kafka", {})
        consumer_cfg = kafka_cfg.get("consumer", {})
        producer_cfg = kafka_cfg.get("producer", {})

        self._bootstrap_servers = kafka_cfg.get("bootstrap_servers", ["localhost:9092"])

        # Consumer settings
        self._consumer_topics = consumer_cfg.get("topics", ["processed_features"])
        self._consumer_group = consumer_cfg.get("group_id", "inference-consumer-group")

        # Producer settings
        self._prediction_topic = producer_cfg.get("topic", "failure_predictions")

        # DB settings (for writing predictions)
        db_cfg = config.get("timescaledb", {})
        self._db_host = db_cfg.get("host", "timescaledb")
        self._db_port = db_cfg.get("port", 5432)
        self._db_name = db_cfg.get("database", "predictive_maintenance")
        self._db_user = db_cfg.get("user", "pmuser")
        self._db_password = db_cfg.get("password", "pmpassword")

        self._consumer: Optional[KafkaConsumer] = None
        self._producer: Optional[KafkaProducer] = None
        self._db_pool: Optional[pg_pool.ThreadedConnectionPool] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._stats = {"consumed": 0, "predicted": 0, "published": 0, "errors": 0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the pipeline in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="kafka-prediction-pipeline"
        )
        self._thread.start()
        logger.info("Kafka prediction pipeline started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._close_resources()
        logger.info("Kafka prediction pipeline stopped — %s", self._stats)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self):
        self._init_consumer()
        self._init_producer()
        self._init_db_pool()

        while self._running:
            try:
                if self._consumer is None:
                    self._init_consumer()
                    if self._consumer is None:
                        time.sleep(5)
                        continue

                records = self._consumer.poll(timeout_ms=1000)
                for tp, messages in records.items():
                    for msg in messages:
                        self._handle_message(msg)
            except Exception as exc:
                logger.error("Pipeline loop error: %s", exc)
                self._stats["errors"] += 1
                time.sleep(2)

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    def _handle_message(self, msg):
        try:
            payload = (
                json.loads(msg.value.decode("utf-8"))
                if isinstance(msg.value, bytes)
                else msg.value
            )
            self._stats["consumed"] += 1

            equipment_id = payload.get("equipment_id")
            features = payload.get("features", {})
            timestamp_str = payload.get("timestamp", datetime.utcnow().isoformat())

            if not equipment_id or not features:
                return

            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if isinstance(timestamp_str, str)
                else timestamp_str
            )

            # --- Run RUL prediction (LSTM) ---
            prediction = self._predict_rul(equipment_id, features, timestamp)

            if prediction:
                self._stats["predicted"] += 1

                # Publish to Kafka
                self._publish_prediction(prediction)

                # Write to TimescaleDB
                self._persist_prediction(prediction)

        except Exception as exc:
            logger.error("Error handling message: %s", exc)
            self._stats["errors"] += 1

    def _predict_rul(
        self, equipment_id: str, features: Dict, timestamp: datetime
    ) -> Optional[Dict]:
        model = self._model_manager.get_model("lstm")
        if model is None:
            return None

        try:
            start = time.time()

            # Build a single-step sequence from features
            processed = self._inference_engine.preprocess_features(
                features, equipment_id
            )

            # For pyfunc models loaded from MLflow, use .predict()
            if hasattr(model, "predict"):
                import numpy as np

                result = model.predict(processed)
                rul = float(np.clip(result.flat[0], 0, 200))
            else:
                rul = 100.0  # safe default

            inference_ms = (time.time() - start) * 1000
            health_status = self._inference_engine.get_health_status_from_rul(rul)

            model_meta = self._model_manager.get_model_metadata("lstm") or {}

            return {
                "equipment_id": equipment_id,
                "timestamp": timestamp.isoformat()
                if isinstance(timestamp, datetime)
                else str(timestamp),
                "prediction_type": "rul",
                "rul_cycles": rul,
                "rul_hours": rul * 1.5,  # approximate conversion
                "confidence": 0.85,
                "health_status": health_status,
                "model_name": model_meta.get("name", "lstm_rul_predictor"),
                "model_version": model_meta.get("version", "unknown"),
                "inference_time_ms": round(inference_ms, 2),
                "input_features": features,
            }
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", equipment_id, exc)
            return None

    # ------------------------------------------------------------------
    # Kafka producer
    # ------------------------------------------------------------------

    def _publish_prediction(self, prediction: Dict):
        if self._producer is None:
            return
        try:
            self._producer.send(
                self._prediction_topic,
                key=prediction["equipment_id"].encode("utf-8"),
                value=json.dumps(prediction, default=str).encode("utf-8"),
            )
            self._stats["published"] += 1
        except KafkaError as exc:
            logger.error("Kafka publish error: %s", exc)

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _persist_prediction(self, prediction: Dict):
        if self._db_pool is None:
            return
        conn = None
        try:
            conn = self._db_pool.getconn()
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions
                    (time, equipment_id, prediction_type, rul_cycles, rul_hours,
                     confidence, health_status, model_name, model_version,
                     inference_time_ms, input_features)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    prediction["timestamp"],
                    prediction["equipment_id"],
                    prediction["prediction_type"],
                    prediction["rul_cycles"],
                    prediction.get("rul_hours"),
                    prediction["confidence"],
                    prediction["health_status"],
                    prediction["model_name"],
                    prediction["model_version"],
                    prediction["inference_time_ms"],
                    json.dumps(prediction.get("input_features")),
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.error("DB persist error: %s", exc)
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._db_pool.putconn(conn)

    # ------------------------------------------------------------------
    # Resource helpers
    # ------------------------------------------------------------------

    def _init_consumer(self):
        try:
            self._consumer = KafkaConsumer(
                *self._consumer_topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._consumer_group,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: v,  # raw bytes, decoded in handler
            )
            logger.info("Kafka consumer connected to %s", self._consumer_topics)
        except Exception as exc:
            logger.warning("Failed to create Kafka consumer: %s", exc)
            self._consumer = None

    def _init_producer(self):
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                compression_type="gzip",
                acks=1,
                retries=3,
            )
            logger.info("Kafka producer connected")
        except Exception as exc:
            logger.warning("Failed to create Kafka producer: %s", exc)
            self._producer = None

    def _init_db_pool(self):
        try:
            self._db_pool = pg_pool.ThreadedConnectionPool(
                1,
                5,
                host=self._db_host,
                port=self._db_port,
                database=self._db_name,
                user=self._db_user,
                password=self._db_password,
            )
            logger.info("Prediction DB pool connected")
        except Exception as exc:
            logger.warning("Failed to create DB pool: %s", exc)
            self._db_pool = None

    def _close_resources(self):
        if self._consumer:
            try:
                self._consumer.close()
            except Exception:
                pass
        if self._producer:
            try:
                self._producer.flush()
                self._producer.close()
            except Exception:
                pass
        if self._db_pool:
            try:
                self._db_pool.closeall()
            except Exception:
                pass

    @property
    def stats(self) -> Dict:
        return dict(self._stats)
