"""
Kafka Alert Consumer

Consumes prediction messages from the ``failure_predictions`` topic and
routes them through the AlertManager for evaluation and notification.

This module serves as the **main entry-point** for the alert-engine
container.
"""

import json
import logging
import signal
import sys
import time
from typing import Dict, Optional

from kafka import KafkaConsumer
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class KafkaAlertConsumer:
    """Consumes predictions from Kafka and feeds them to the AlertManager."""

    def __init__(self, alert_manager, config: Dict):
        self._alert_manager = alert_manager
        self._config = config

        kafka_cfg = config.get("kafka", {})
        consumer_cfg = kafka_cfg.get("consumer", {})

        self._bootstrap_servers = kafka_cfg.get("bootstrap_servers", ["kafka:9092"])
        self._topics = consumer_cfg.get("topics", ["failure_predictions"])
        self._group_id = consumer_cfg.get("group_id", "alert-engine-group")

        self._consumer: Optional[KafkaConsumer] = None
        self._running = False

        # Stats
        self._consumed = 0
        self._alerts_triggered = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Connect to Kafka and start the consume loop (blocking)."""
        self._running = True
        self._connect()

        logger.info(
            "KafkaAlertConsumer started — topics=%s, group=%s",
            self._topics,
            self._group_id,
        )

        while self._running:
            try:
                if self._consumer is None:
                    self._connect()
                    if self._consumer is None:
                        time.sleep(5)
                        continue

                records = self._consumer.poll(timeout_ms=1000)
                for tp, messages in records.items():
                    for msg in messages:
                        self._handle(msg)

            except Exception as exc:
                logger.error("Consumer loop error: %s", exc)
                self._errors += 1
                time.sleep(2)

    def stop(self):
        logger.info("Stopping KafkaAlertConsumer...")
        self._running = False
        if self._consumer:
            try:
                self._consumer.close()
            except Exception:
                pass
        logger.info(
            "KafkaAlertConsumer stopped — consumed=%d alerts=%d errors=%d",
            self._consumed,
            self._alerts_triggered,
            self._errors,
        )

    # ------------------------------------------------------------------
    # Message handler
    # ------------------------------------------------------------------

    def _handle(self, msg):
        try:
            payload = (
                json.loads(msg.value) if isinstance(msg.value, bytes) else msg.value
            )
            self._consumed += 1

            # Map Kafka prediction payload to the format AlertManager expects
            prediction_data = self._normalise_prediction(payload)

            alerts = self._alert_manager.process_prediction(prediction_data)
            self._alerts_triggered += len(alerts)

        except Exception as exc:
            logger.error("Error handling message: %s", exc)
            self._errors += 1

    @staticmethod
    def _normalise_prediction(payload: Dict) -> Dict:
        """Translate prediction payload keys to what AlertRuleEngine rules expect."""
        return {
            "equipment_id": payload.get("equipment_id", ""),
            "rul": payload.get("rul_cycles", payload.get("rul")),
            "anomaly_score": payload.get("anomaly_score", 0.0),
            "health_status": payload.get("health_status", ""),
            "confidence": payload.get("confidence", 0.0),
            "timestamp": payload.get("timestamp"),
            # Passthrough any extra sensor values the rules may reference
            **{
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "equipment_id",
                    "rul_cycles",
                    "rul",
                    "anomaly_score",
                    "health_status",
                    "confidence",
                    "timestamp",
                }
            },
        }

    # ------------------------------------------------------------------
    # Kafka connection
    # ------------------------------------------------------------------

    def _connect(self):
        try:
            self._consumer = KafkaConsumer(
                *self._topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: v,
            )
            logger.info("Kafka consumer connected to %s", self._topics)
        except KafkaError as exc:
            logger.warning("Failed to connect to Kafka: %s — retrying...", exc)
            self._consumer = None
