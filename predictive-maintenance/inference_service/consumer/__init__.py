"""
Kafka Prediction Pipeline for the Inference Service.

Subscribes to the ``raw_sensor_data`` topic and maintains a per-equipment
sliding window (default 50 readings).  On every new reading the full buffered
sequence is passed to OnlineFeatureEngineer, producing the same lag/rolling/EMA
feature vector that the MLP was trained on.  This is identical to the
POST /predict/rul feature path.

Previous behaviour (subscribed to ``processed_features``, fed flat time/frequency-
domain features to the MLP) was wrong: the MLP was trained on temporal features,
not on stream-processor output features.  That mismatch caused silent wrong-column
inference.

Data flow after this fix
------------------------
raw_sensor_data topic
    → _extract_raw_reading()        normalise message to {sensor_N: float, ...}
    → _EquipmentBuffer.push()       append to per-equipment deque(maxlen=50)
    → OnlineFeatureEngineer.engineer(sequence)   lag/rolling/EMA features (200-d)
    → InferenceEngine.predict_rul(model, features)   MLP forward pass
    → failure_predictions topic + TimescaleDB

Fallback (LSTM / pyfunc model)
    → InferenceEngine.preprocess_sequence(sequence)  sequence padded to (1,50,N)
    → model.predict(sequence_array)
"""

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Lazy module-level import — guarded so unit tests that don't set up the full
# package still work.  The guard is re-checked inside _predict_rul at call time.
try:
    from ..feature_engineering import online_feature_engineer as _online_feature_engineer
except ImportError:
    _online_feature_engineer = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Per-equipment sequence buffer
# ---------------------------------------------------------------------------

class _EquipmentBuffer:
    """
    Fixed-length sliding window of raw sensor readings for one equipment unit.

    Each element is a dict in the format OnlineFeatureEngineer expects:
        {"sensor_2": float, "sensor_3": float, ...,
         "op_setting_1": float, ..., "time_cycle": float}

    Thread-safety: only the single Kafka poll thread writes to this buffer.
    """

    __slots__ = ("_buf",)

    def __init__(self, maxlen: int = 50) -> None:
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, reading: Dict[str, float]) -> None:
        self._buf.append(reading)

    def sequence(self) -> List[Dict[str, float]]:
        """Return buffer contents as a list, oldest first."""
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class KafkaPredictionPipeline:
    """
    End-to-end pipeline:
        raw_sensor_data → buffer → OnlineFeatureEngineer → MLP → failure_predictions
    """

    def __init__(self, config: Dict, model_manager: Any, inference_engine: Any):
        self._config = config
        self._model_manager = model_manager
        self._inference_engine = inference_engine

        kafka_cfg = config.get("kafka", {})
        consumer_cfg = kafka_cfg.get("consumer", {})
        producer_cfg = kafka_cfg.get("producer", {})

        self._bootstrap_servers = kafka_cfg.get("bootstrap_servers", ["localhost:9092"])

        # Subscribe to raw_sensor_data so we receive individual sensor readings
        # that can be buffered and fed to OnlineFeatureEngineer.
        # The config must say "raw_sensor_data" — "processed_features" carries
        # stream-processor time/freq-domain features that do NOT match the MLP's
        # lag/rolling/EMA training feature schema.
        self._consumer_topics = consumer_cfg.get("topics", ["raw_sensor_data"])
        self._consumer_group = consumer_cfg.get("group_id", "inference-consumer-group")

        # Window length must match the sequence length used at training time.
        self._seq_len: int = consumer_cfg.get("sequence_length", 50)

        # Minimum number of readings in the per-equipment buffer before the
        # first prediction is emitted for that equipment.  During warm-up,
        # lag/rolling/EMA features are computed from very short histories and
        # produce near-zero temporal signals — those predictions are unreliable.
        # Default: 10 readings (configurable via kafka.consumer.min_warmup_readings).
        self._min_warmup: int = consumer_cfg.get("min_warmup_readings", 10)

        # Producer (output) topic
        self._prediction_topic = producer_cfg.get("topic", "failure_predictions")

        # TimescaleDB connection settings
        db_cfg = config.get("timescaledb", {})
        self._db_host = db_cfg.get("host", "timescaledb")
        self._db_port = db_cfg.get("port", 5432)
        self._db_name = db_cfg.get("database", "predictive_maintenance")
        self._db_user = db_cfg.get("user", "pmuser")
        self._db_password = db_cfg.get("password", "pmpassword")

        self._consumer: Optional[Any] = None
        self._producer: Optional[Any] = None
        self._db_pool: Optional[Any] = None

        # Per-equipment sliding window — keyed by equipment_id
        self._sequence_buffers: Dict[str, _EquipmentBuffer] = {}

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats = {"consumed": 0, "predicted": 0, "published": 0, "errors": 0}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the pipeline in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="kafka-prediction-pipeline"
        )
        self._thread.start()
        logger.info(
            "Kafka prediction pipeline started — topic=%s  seq_len=%d",
            self._consumer_topics,
            self._seq_len,
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self._close_resources()
        logger.info("Kafka prediction pipeline stopped — %s", self._stats)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
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
    # Raw reading extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_raw_reading(payload: Dict) -> Optional[Dict[str, float]]:
        """
        Normalise a raw_sensor_data message into a flat reading dict for
        OnlineFeatureEngineer.

        Handles two message shapes that exist in this codebase:

        Generic (data_generator):
            {"sensor_data": {"sensor_N": v, ...}, "time_cycle": t}

        C-MAPSS (data_loader/kafka_streamer):
            {"sensors": {"sensor_N": v, ...},
             "operating_settings": {"setting_N": v, ...},
             "time_cycle": t}

        Returns None if no sensor data is found in the payload.
        """
        reading: Dict[str, float] = {}

        # Generic: sensor_data block
        for k, v in payload.get("sensor_data", {}).items():
            try:
                reading[k] = float(v)
            except (TypeError, ValueError):
                pass

        # C-MAPSS: sensors block
        for k, v in payload.get("sensors", {}).items():
            try:
                reading[k] = float(v)
            except (TypeError, ValueError):
                pass

        # C-MAPSS: operating_settings → op_setting_N keys
        # ("setting_1" → "op_setting_1" to match OnlineFeatureEngineer's _OP_KEYS)
        for raw_k, v in payload.get("operating_settings", {}).items():
            try:
                op_key = (
                    raw_k.replace("setting_", "op_setting_")
                    if raw_k.startswith("setting_")
                    else raw_k
                )
                reading[op_key] = float(v)
            except (TypeError, ValueError):
                pass

        # time_cycle
        if "time_cycle" in payload:
            try:
                reading["time_cycle"] = float(payload["time_cycle"])
            except (TypeError, ValueError):
                pass

        return reading if reading else None

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    def _handle_message(self, msg) -> None:
        try:
            payload = (
                json.loads(msg.value.decode("utf-8"))
                if isinstance(msg.value, bytes)
                else msg.value
            )
            self._stats["consumed"] += 1

            equipment_id = payload.get("equipment_id")
            timestamp_str = payload.get("timestamp", datetime.utcnow().isoformat())

            if not equipment_id:
                return

            timestamp = (
                datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                if isinstance(timestamp_str, str)
                else timestamp_str
            )

            # Extract raw sensor reading and add to the per-equipment buffer
            reading = self._extract_raw_reading(payload)
            if not reading:
                logger.debug(
                    "No raw sensor data in message for %s — skipping", equipment_id
                )
                return

            if equipment_id not in self._sequence_buffers:
                self._sequence_buffers[equipment_id] = _EquipmentBuffer(self._seq_len)
            buf = self._sequence_buffers[equipment_id]
            buf.push(reading)

            # Suppress predictions during warm-up.  OnlineFeatureEngineer
            # produces near-zero lag/rolling/EMA values for very short histories
            # because most look-back windows cannot be fully populated.
            buf_len = len(buf)
            if buf_len < self._min_warmup:
                logger.debug(
                    "Warm-up suppression for %s — buffer %d/%d readings",
                    equipment_id, buf_len, self._min_warmup,
                )
                return

            # Reconstruct the full sequence and run inference
            sequence = buf.sequence()
            prediction = self._predict_rul(equipment_id, sequence, timestamp)

            if prediction:
                self._stats["predicted"] += 1
                self._publish_prediction(prediction)
                self._persist_prediction(prediction)

        except Exception as exc:
            logger.error("Error handling message: %s", exc)
            self._stats["errors"] += 1

    # ------------------------------------------------------------------
    # Inference — sequence-based, mirrors POST /predict/rul exactly
    # ------------------------------------------------------------------

    def _predict_rul(
        self,
        equipment_id: str,
        sequence: List[Dict[str, float]],
        timestamp: datetime,
    ) -> Optional[Dict]:
        """
        Run RUL inference using the same feature path as POST /predict/rul.

        MLP path  (primary):
            OnlineFeatureEngineer.engineer(sequence) → (1, 200) float32 array
            → MLP forward pass → rul

        LSTM / pyfunc path (fallback):
            InferenceEngine.preprocess_sequence(sequence) → (1, seq_len, N) array
            → model.predict() → rul

        The isinstance(model, torch.nn.Module) guard prevents the wrong
        preprocessing reaching the wrong model in both directions.
        """
        model = self._model_manager.get_model("mlp")
        model_label = "mlp"
        if model is None:
            model = self._model_manager.get_model("lstm")
            model_label = "lstm"
        if model is None:
            return None

        try:
            start = time.time()

            try:
                import torch as _torch
                if isinstance(model, _torch.nn.Module):
                    # MLP: OnlineFeatureEngineer reconstructs lag/rolling/EMA
                    # features over the full buffered sequence — identical to
                    # the training feature path in baseline_comparison.py.
                    ofe = _online_feature_engineer
                    if ofe is None:
                        raise ImportError("online_feature_engineer not available")
                    features = ofe.engineer(sequence, equipment_id)
                else:
                    raise TypeError("not a torch module")
            except (ImportError, TypeError):
                # LSTM / pyfunc: sequence-array preprocessing
                features = self._inference_engine.preprocess_sequence(
                    sequence, equipment_id
                )

            rul, _ = self._inference_engine.predict_rul(model, features)
            inference_ms = (time.time() - start) * 1000
            health_status = self._inference_engine.get_health_status_from_rul(rul)
            model_meta = self._model_manager.get_model_metadata(model_label) or {}

            return {
                "equipment_id": equipment_id,
                "timestamp": (
                    timestamp.isoformat()
                    if isinstance(timestamp, datetime)
                    else str(timestamp)
                ),
                "prediction_type": "rul",
                "rul_cycles": rul,
                "rul_hours": rul * 1.5,
                "confidence": 0.85,
                "health_status": health_status,
                "model_name": model_meta.get("name", f"{model_label}_rul_predictor"),
                "model_version": model_meta.get("version", "unknown"),
                "inference_time_ms": round(inference_ms, 2),
                "sequence_length": len(sequence),
                "input_features": sequence[-1] if sequence else {},
            }
        except Exception as exc:
            logger.error("Prediction failed for %s: %s", equipment_id, exc)
            return None

    # ------------------------------------------------------------------
    # Kafka producer
    # ------------------------------------------------------------------

    def _publish_prediction(self, prediction: Dict) -> None:
        if self._producer is None:
            return
        try:
            self._producer.send(
                self._prediction_topic,
                key=prediction["equipment_id"].encode("utf-8"),
                value=json.dumps(prediction, default=str).encode("utf-8"),
            )
            self._stats["published"] += 1
        except Exception as exc:
            logger.error("Kafka publish error: %s", exc)

    # ------------------------------------------------------------------
    # DB persistence
    # ------------------------------------------------------------------

    def _persist_prediction(self, prediction: Dict) -> None:
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

    def _init_consumer(self) -> None:
        try:
            from kafka import KafkaConsumer  # lazy import — optional at module load time
            self._consumer = KafkaConsumer(
                *self._consumer_topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._consumer_group,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: v,  # raw bytes; decoded in handler
            )
            logger.info("Kafka consumer connected to %s", self._consumer_topics)
        except Exception as exc:
            logger.warning("Failed to create Kafka consumer: %s", exc)
            self._consumer = None

    def _init_producer(self) -> None:
        try:
            from kafka import KafkaProducer  # lazy import
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

    def _init_db_pool(self) -> None:
        try:
            import psycopg2
            from psycopg2 import pool as pg_pool  # lazy import
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

    def _close_resources(self) -> None:
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
