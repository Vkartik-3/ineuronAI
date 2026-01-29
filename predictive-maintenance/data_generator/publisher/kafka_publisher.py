"""
Kafka Publisher
Publishes sensor data to Kafka topics
"""

import json
import logging
from typing import Dict, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import time

logger = logging.getLogger(__name__)


class SensorDataPublisher:
    """Publishes sensor data to Kafka"""

    def __init__(self, kafka_config: Dict):
        """
        Initialize Kafka publisher

        Args:
            kafka_config: Kafka configuration dictionary
        """
        self.config = kafka_config
        kafka_settings = kafka_config.get("kafka", {})
        producer_config = kafka_settings.get("producer", {})

        self.bootstrap_servers = kafka_settings.get(
            "bootstrap_servers", ["localhost:9092"]
        )
        self.raw_topic = kafka_settings.get("topics", {}).get(
            "raw_sensor_data", "raw_sensor_data"
        )
        self.metadata_topic = kafka_settings.get("topics", {}).get(
            "equipment_metadata", "equipment_metadata"
        )

        self.producer = None
        self.messages_sent = 0
        self.errors_count = 0
        self.last_error = None

        # Initialize producer
        self._init_producer(producer_config)

    def _init_producer(self, producer_config: Dict):
        """Initialize Kafka producer"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks=producer_config.get("acks", 1),
                retries=producer_config.get("retries", 3),
                max_in_flight_requests_per_connection=producer_config.get(
                    "max_in_flight_requests_per_connection", 5
                ),
                compression_type=producer_config.get("compression_type", "gzip"),
                batch_size=producer_config.get("batch_size", 16384),
                linger_ms=producer_config.get("linger_ms", 10),
                buffer_memory=producer_config.get("buffer_memory", 33554432),
            )
            logger.info(
                f"Kafka producer initialized successfully. Bootstrap servers: {self.bootstrap_servers}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Kafka producer: {e}")
            self.producer = None

    def publish_sensor_data(self, data: Dict, equipment_id: str) -> bool:
        """
        Publish sensor data to Kafka

        Args:
            data: Sensor data dictionary
            equipment_id: Equipment identifier (used as message key for partitioning)

        Returns:
            True if successful, False otherwise
        """
        if not self.producer:
            logger.error("Kafka producer not initialized")
            return False

        try:
            # Send message asynchronously
            future = self.producer.send(
                topic=self.raw_topic, key=equipment_id, value=data
            )

            # Optionally wait for confirmation (blocking)
            # record_metadata = future.get(timeout=10)
            # logger.debug(f"Message sent to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}")

            self.messages_sent += 1

            if self.messages_sent % 100 == 0:
                logger.info(f"Sent {self.messages_sent} messages to Kafka")

            return True

        except KafkaError as e:
            self.errors_count += 1
            self.last_error = str(e)
            logger.error(f"Failed to send message to Kafka: {e}")
            return False
        except Exception as e:
            self.errors_count += 1
            self.last_error = str(e)
            logger.error(f"Unexpected error sending message: {e}")
            return False

    def publish_metadata(self, metadata: Dict, equipment_id: str) -> bool:
        """
        Publish equipment metadata to Kafka

        Args:
            metadata: Equipment metadata dictionary
            equipment_id: Equipment identifier

        Returns:
            True if successful, False otherwise
        """
        if not self.producer:
            logger.error("Kafka producer not initialized")
            return False

        try:
            future = self.producer.send(
                topic=self.metadata_topic, key=equipment_id, value=metadata
            )

            logger.info(f"Published metadata for equipment {equipment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish metadata: {e}")
            return False

    def flush(self):
        """Flush any pending messages"""
        if self.producer:
            self.producer.flush()
            logger.debug("Flushed pending messages")

    def close(self):
        """Close the producer and clean up resources"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info(
                f"Kafka producer closed. Total messages sent: {self.messages_sent}, Errors: {self.errors_count}"
            )

    def get_stats(self) -> Dict:
        """Get publisher statistics"""
        return {
            "messages_sent": self.messages_sent,
            "errors_count": self.errors_count,
            "last_error": self.last_error,
            "is_connected": self.producer is not None,
        }


class MockPublisher:
    """Mock publisher for testing without Kafka"""

    def __init__(self, kafka_config: Dict):
        self.config = kafka_config
        self.messages_sent = 0
        self.messages = []
        logger.info("Mock publisher initialized (Kafka not required)")

    def publish_sensor_data(self, data: Dict, equipment_id: str) -> bool:
        """Mock publish - stores messages in memory"""
        self.messages.append(
            {"equipment_id": equipment_id, "data": data, "timestamp": time.time()}
        )
        self.messages_sent += 1

        if self.messages_sent % 100 == 0:
            logger.info(f"[MOCK] Sent {self.messages_sent} messages")

        return True

    def publish_metadata(self, metadata: Dict, equipment_id: str) -> bool:
        """Mock publish metadata"""
        logger.info(f"[MOCK] Published metadata for {equipment_id}")
        return True

    def flush(self):
        """Mock flush"""
        pass

    def close(self):
        """Mock close"""
        logger.info(f"[MOCK] Publisher closed. Total messages: {self.messages_sent}")

    def get_stats(self) -> Dict:
        """Get mock statistics"""
        return {
            "messages_sent": self.messages_sent,
            "errors_count": 0,
            "last_error": None,
            "is_connected": True,
        }

    def get_messages(self):
        """Get all stored messages (for testing)"""
        return self.messages
