"""
Kafka Feature Publisher
Publishes computed features to a Kafka topic so downstream services
(inference, alerting) can consume them in real time.
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class KafkaFeaturePublisher:
    """Publishes engineered features to the ``processed_features`` Kafka topic."""

    def __init__(self, config: Dict):
        kafka_cfg = config.get("kafka", {})
        producer_cfg = kafka_cfg.get("producer", {})

        self.topic = producer_cfg.get("topic", "processed_features")
        self.bootstrap_servers = kafka_cfg.get("bootstrap_servers", ["localhost:9092"])

        self._producer: Optional[KafkaProducer] = None
        self._connect(producer_cfg)

        # Counters
        self._published = 0
        self._errors = 0

        logger.info(
            "KafkaFeaturePublisher initialised — topic=%s, servers=%s",
            self.topic,
            self.bootstrap_servers,
        )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self, producer_cfg: Dict):
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                compression_type=producer_cfg.get("compression_type", "gzip"),
                acks=producer_cfg.get("acks", 1),
                retries=producer_cfg.get("retries", 3),
                linger_ms=producer_cfg.get("linger_ms", 10),
                batch_size=producer_cfg.get("batch_size_bytes", 16384),
            )
        except KafkaError as exc:
            logger.error("Failed to create KafkaProducer: %s", exc)
            self._producer = None

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_set: str = "combined",
        metadata: Optional[Dict] = None,
    ):
        """Publish a feature envelope to Kafka.

        The message key is ``equipment_id`` so that all data for a given
        piece of equipment lands on the same partition.
        """
        if self._producer is None:
            logger.warning("KafkaProducer not available — skipping publish")
            return

        message = {
            "equipment_id": equipment_id,
            "timestamp": timestamp.isoformat()
            if isinstance(timestamp, datetime)
            else str(timestamp),
            "feature_set": feature_set,
            "features": features,
            "metadata": metadata or {},
        }

        try:
            future = self._producer.send(self.topic, key=equipment_id, value=message)
            future.add_callback(self._on_success)
            future.add_errback(self._on_error)
        except KafkaError as exc:
            self._errors += 1
            logger.error("Failed to publish features for %s: %s", equipment_id, exc)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_success(self, record_metadata):
        self._published += 1
        logger.debug(
            "Published to %s partition=%s offset=%s",
            record_metadata.topic,
            record_metadata.partition,
            record_metadata.offset,
        )

    def _on_error(self, exc):
        self._errors += 1
        logger.error("Kafka publish error: %s", exc)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self):
        if self._producer:
            self._producer.flush()

    def close(self):
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info(
                "KafkaFeaturePublisher closed — published=%d errors=%d",
                self._published,
                self._errors,
            )

    @property
    def stats(self) -> Dict:
        return {"published": self._published, "errors": self._errors}


class MockKafkaFeaturePublisher:
    """In-memory publisher for testing without a real Kafka broker."""

    def __init__(self, config: Dict):
        self.messages: list = []
        logger.info("MockKafkaFeaturePublisher initialised")

    def publish_features(
        self,
        equipment_id: str,
        timestamp: datetime,
        features: Dict,
        feature_set: str = "combined",
        metadata: Optional[Dict] = None,
    ):
        self.messages.append(
            {
                "equipment_id": equipment_id,
                "timestamp": timestamp,
                "feature_set": feature_set,
                "features": features,
                "metadata": metadata,
            }
        )

    def flush(self):
        pass

    def close(self):
        logger.info(
            "MockKafkaFeaturePublisher closed — %d messages buffered",
            len(self.messages),
        )

    @property
    def stats(self) -> Dict:
        return {"published": len(self.messages), "errors": 0}
