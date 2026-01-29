"""
Kafka Consumer for Stream Processor
Consumes raw sensor data from Kafka topics
"""

import json
import logging
from typing import Dict, List, Optional, Callable
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import time

logger = logging.getLogger(__name__)


class SensorDataConsumer:
    """Consumes sensor data from Kafka topics"""

    def __init__(self, config: Dict):
        """
        Initialize Kafka consumer

        Args:
            config: Configuration dictionary
        """
        self.config = config
        kafka_config = config.get("kafka", {})
        consumer_config = kafka_config.get("consumer", {})

        self.bootstrap_servers = kafka_config.get(
            "bootstrap_servers", ["localhost:9092"]
        )
        self.group_id = consumer_config.get("group_id", "stream-processor-group")
        self.topics = consumer_config.get("topics", ["raw_sensor_data"])

        self.consumer: Optional[KafkaConsumer] = None
        self.messages_consumed = 0
        self.errors_count = 0
        self.last_commit_time = time.time()

        self._init_consumer(consumer_config)

    def _init_consumer(self, consumer_config: Dict):
        """Initialize Kafka consumer"""
        try:
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=consumer_config.get("auto_offset_reset", "earliest"),
                enable_auto_commit=consumer_config.get("enable_auto_commit", False),
                max_poll_records=consumer_config.get("max_poll_records", 500),
                max_poll_interval_ms=consumer_config.get(
                    "max_poll_interval_ms", 300000
                ),
                session_timeout_ms=consumer_config.get("session_timeout_ms", 30000),
                heartbeat_interval_ms=consumer_config.get(
                    "heartbeat_interval_ms", 10000
                ),
                fetch_min_bytes=consumer_config.get("fetch_min_bytes", 1024),
                fetch_max_wait_ms=consumer_config.get("fetch_max_wait_ms", 500),
                max_partition_fetch_bytes=consumer_config.get(
                    "max_partition_fetch_bytes", 1048576
                ),
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
            )
            logger.info(
                f"Kafka consumer initialized. Topics: {self.topics}, Group: {self.group_id}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Kafka consumer: {e}")
            self.consumer = None

    def consume(
        self,
        callback: Callable[[Dict], bool],
        batch_size: int = 100,
        commit_interval: int = 10,
    ) -> None:
        """
        Consume messages and process with callback

        Args:
            callback: Function to process each message. Returns True on success.
            batch_size: Number of messages to process before committing
            commit_interval: Time in seconds between commits
        """
        if not self.consumer:
            logger.error("Consumer not initialized")
            return

        logger.info("Starting message consumption...")

        try:
            message_buffer = []

            for message in self.consumer:
                try:
                    # Extract message data
                    data = message.value

                    # Process message with callback
                    success = callback(data)

                    if success:
                        self.messages_consumed += 1
                        message_buffer.append(message)
                    else:
                        self.errors_count += 1
                        logger.warning(
                            f"Failed to process message from {message.topic}"
                        )

                    # Commit offsets periodically
                    current_time = time.time()
                    should_commit = (
                        len(message_buffer) >= batch_size
                        or (current_time - self.last_commit_time) >= commit_interval
                    )

                    if should_commit and message_buffer:
                        self._commit_offsets()
                        message_buffer.clear()
                        self.last_commit_time = current_time

                    # Log progress
                    if self.messages_consumed % 1000 == 0:
                        logger.info(
                            f"Consumed {self.messages_consumed} messages, "
                            f"Errors: {self.errors_count}"
                        )

                except json.JSONDecodeError as e:
                    self.errors_count += 1
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    self.errors_count += 1
                    logger.error(f"Error processing message: {e}", exc_info=True)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping consumer...")
        except Exception as e:
            logger.error(f"Consumer error: {e}", exc_info=True)
        finally:
            self._commit_offsets()
            self.close()

    def consume_batch(
        self, max_messages: int = 500, timeout_ms: int = 5000
    ) -> List[Dict]:
        """
        Consume a batch of messages

        Args:
            max_messages: Maximum number of messages to consume
            timeout_ms: Timeout in milliseconds

        Returns:
            List of message data dictionaries
        """
        if not self.consumer:
            logger.error("Consumer not initialized")
            return []

        messages = []

        try:
            message_batch = self.consumer.poll(
                timeout_ms=timeout_ms, max_records=max_messages
            )

            for topic_partition, records in message_batch.items():
                for record in records:
                    try:
                        data = record.value
                        messages.append(data)
                        self.messages_consumed += 1
                    except Exception as e:
                        self.errors_count += 1
                        logger.error(f"Error processing record: {e}")

            if messages:
                logger.debug(f"Consumed batch of {len(messages)} messages")

        except Exception as e:
            logger.error(f"Error consuming batch: {e}")

        return messages

    def _commit_offsets(self):
        """Commit consumer offsets"""
        if self.consumer:
            try:
                self.consumer.commit()
                logger.debug("Committed offsets")
            except Exception as e:
                logger.error(f"Failed to commit offsets: {e}")

    def close(self):
        """Close consumer and clean up"""
        if self.consumer:
            try:
                self.consumer.close()
                logger.info(
                    f"Consumer closed. Total messages: {self.messages_consumed}, "
                    f"Errors: {self.errors_count}"
                )
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")

    def get_stats(self) -> Dict:
        """Get consumer statistics"""
        return {
            "messages_consumed": self.messages_consumed,
            "errors_count": self.errors_count,
            "topics": self.topics,
            "group_id": self.group_id,
            "is_connected": self.consumer is not None,
        }


class MockConsumer:
    """Mock consumer for testing without Kafka"""

    def __init__(self, config: Dict):
        self.config = config
        self.messages_consumed = 0
        logger.info("Mock consumer initialized (Kafka not required)")

    def consume(self, callback: Callable[[Dict], bool], **kwargs):
        """Mock consume - does nothing"""
        logger.info("[MOCK] Consumer started")

    def consume_batch(self, **kwargs) -> List[Dict]:
        """Mock consume batch - returns empty list"""
        return []

    def close(self):
        """Mock close"""
        logger.info("[MOCK] Consumer closed")

    def get_stats(self) -> Dict:
        """Mock stats"""
        return {
            "messages_consumed": self.messages_consumed,
            "errors_count": 0,
            "is_connected": True,
        }
