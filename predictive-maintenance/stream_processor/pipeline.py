"""
Stream Processing Pipeline
Orchestrates data flow from Kafka to TimescaleDB with feature engineering
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import time
import yaml

from consumer.kafka_consumer import SensorDataConsumer, MockConsumer
from features.time_domain_features import TimeDomainFeatures, AggregatedFeatures
from features.frequency_domain_features import FrequencyDomainFeatures
from writer.timescaledb_writer import TimescaleDBWriter, MockTimescaleDBWriter

logger = logging.getLogger(__name__)


class StreamProcessor:
    """Main stream processing pipeline"""

    def __init__(self, config_path: str, mock_mode: bool = False):
        """
        Initialize stream processor

        Args:
            config_path: Path to configuration file
            mock_mode: Whether to use mock components
        """
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.mock_mode = mock_mode

        # Initialize components
        self._init_consumer()
        self._init_feature_extractors()
        self._init_writer()

        # Processing settings
        proc_config = self.config.get("processing", {})
        self.num_workers = proc_config.get("num_workers", 4)
        self.input_buffer_size = proc_config.get("input_buffer_size", 1000)
        self.output_buffer_size = proc_config.get("output_buffer_size", 500)

        # Buffers
        self.input_queue = queue.Queue(maxsize=self.input_buffer_size)
        self.output_queue = queue.Queue(maxsize=self.output_buffer_size)

        # Statistics
        self.stats = {
            "messages_consumed": 0,
            "messages_processed": 0,
            "messages_written": 0,
            "errors": 0,
            "start_time": None,
        }

        # Running flag
        self.running = False

        logger.info(
            f"Stream processor initialized. "
            f"Mock mode: {mock_mode}, Workers: {self.num_workers}"
        )

    def _init_consumer(self):
        """Initialize Kafka consumer"""
        if self.mock_mode:
            self.consumer = MockConsumer(self.config)
            logger.info("Using mock Kafka consumer")
        else:
            self.consumer = SensorDataConsumer(self.config)
            logger.info("Using real Kafka consumer")

    def _init_feature_extractors(self):
        """Initialize feature extractors"""
        self.time_features = TimeDomainFeatures(self.config)
        self.freq_features = FrequencyDomainFeatures(self.config)
        logger.info("Feature extractors initialized")

    def _init_writer(self):
        """Initialize database writer"""
        if self.mock_mode:
            self.writer = MockTimescaleDBWriter(self.config)
            logger.info("Using mock TimescaleDB writer")
        else:
            self.writer = TimescaleDBWriter(self.config)
            logger.info("Using real TimescaleDB writer")

    def process_message(self, message: Dict):
        """
        Process a single message

        Args:
            message: Kafka message with sensor data
        """
        try:
            # Detect data source (synthetic vs C-MAPSS)
            data_source = message.get("data_source", "synthetic")

            # Extract message fields based on source
            if data_source == "cmapss":
                # C-MAPSS format
                equipment_id = message.get("equipment_id")
                timestamp_str = message.get("timestamp")
                unit_id = message.get("unit_id")  # Preserve C-MAPSS unit_id
                time_cycle = message.get("time_cycle")

                # Extract sensor data from C-MAPSS format
                sensor_data = message.get("sensors", {})

                # Add operating settings to metadata
                metadata = {
                    "unit_id": unit_id,
                    "time_cycle": time_cycle,
                    "dataset": message.get("dataset", "FD001"),
                    "data_source": "cmapss",
                    "operating_settings": message.get("operating_settings", {}),
                }

                # Add RUL if available (for evaluation)
                if "rul" in message:
                    metadata["rul"] = message["rul"]
            else:
                # Synthetic data format (original)
                equipment_id = message.get("equipment_id")
                timestamp_str = message.get("timestamp")
                sensor_data = message.get("sensor_data", {})
                metadata = message.get("metadata", {})

            if not equipment_id or not timestamp_str:
                logger.warning("Missing equipment_id or timestamp")
                self.stats["errors"] += 1
                return

            # Parse timestamp
            timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            # Write raw sensor data
            self.writer.write_sensor_reading(
                equipment_id=equipment_id,
                timestamp=timestamp,
                sensor_data=sensor_data,
                metadata=metadata,
            )

            # Extract time-domain features
            time_domain_features = self.time_features.extract_features(
                equipment_id=equipment_id, sensor_data=sensor_data
            )

            # Extract frequency-domain features
            freq_domain_features = self.freq_features.extract_features(
                equipment_id=equipment_id, sensor_data=sensor_data
            )

            # Extract cross-sensor features
            cross_sensor_features = AggregatedFeatures.compute_cross_sensor_features(
                sensor_data
            )

            # Combine all features
            all_features = {
                **time_domain_features,
                **freq_domain_features,
                **cross_sensor_features,
            }

            # Write processed features
            if all_features:
                self.writer.write_processed_features(
                    equipment_id=equipment_id,
                    timestamp=timestamp,
                    features=all_features,
                    feature_type="combined",
                )

            self.stats["messages_processed"] += 1

            # Log progress every 100 messages
            if self.stats["messages_processed"] % 100 == 0:
                logger.info(
                    f"Processed {self.stats['messages_processed']} messages. "
                    f"Errors: {self.stats['errors']}"
                )

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.stats["errors"] += 1

    def consumer_thread(self):
        """Consumer thread - reads from Kafka"""
        logger.info("Consumer thread started")

        def callback(message):
            """Callback for each consumed message"""
            self.stats["messages_consumed"] += 1
            self.input_queue.put(message)

        try:
            self.consumer.consume(callback=callback)
        except Exception as e:
            logger.error(f"Consumer thread error: {e}")

    def worker_thread(self, worker_id: int):
        """Worker thread - processes messages"""
        logger.info(f"Worker {worker_id} started")

        while self.running:
            try:
                # Get message from queue with timeout
                message = self.input_queue.get(timeout=1)

                # Process message
                self.process_message(message)

                # Mark as done
                self.input_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    def writer_thread(self):
        """Writer thread - flushes data periodically"""
        logger.info("Writer thread started")

        flush_interval = 5  # seconds

        while self.running:
            try:
                time.sleep(flush_interval)

                # Flush all buffers
                self.writer.flush_all()

                logger.debug("Flushed database buffers")

            except Exception as e:
                logger.error(f"Writer thread error: {e}")

    def start(self):
        """Start stream processing"""
        if self.running:
            logger.warning("Processor already running")
            return

        self.running = True
        self.stats["start_time"] = time.time()

        logger.info("Starting stream processor...")

        # Start threads
        with ThreadPoolExecutor(max_workers=self.num_workers + 2) as executor:
            # Consumer thread
            consumer_future = executor.submit(self.consumer_thread)

            # Worker threads
            worker_futures = [
                executor.submit(self.worker_thread, i) for i in range(self.num_workers)
            ]

            # Writer thread
            writer_future = executor.submit(self.writer_thread)

            logger.info(f"Started {self.num_workers} worker threads")

            try:
                # Wait for threads
                for future in as_completed(
                    [consumer_future, writer_future] + worker_futures
                ):
                    future.result()

            except KeyboardInterrupt:
                logger.info("Received interrupt signal")

            finally:
                self.stop()

    def stop(self):
        """Stop stream processing"""
        if not self.running:
            return

        logger.info("Stopping stream processor...")

        self.running = False

        # Wait for queue to empty
        logger.info("Waiting for queue to empty...")
        self.input_queue.join()

        # Final flush
        logger.info("Final flush...")
        self.writer.flush_all()

        # Close connections
        self.writer.close()

        # Print statistics
        self._print_statistics()

        logger.info("Stream processor stopped")

    def _print_statistics(self):
        """Print processing statistics"""
        elapsed_time = time.time() - self.stats["start_time"]

        logger.info("=" * 50)
        logger.info("Stream Processing Statistics")
        logger.info("=" * 50)
        logger.info(f"Messages consumed: {self.stats['messages_consumed']}")
        logger.info(f"Messages processed: {self.stats['messages_processed']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

        if elapsed_time > 0:
            throughput = self.stats["messages_processed"] / elapsed_time
            logger.info(f"Throughput: {throughput:.2f} messages/sec")

        logger.info("=" * 50)


class DataQualityChecker:
    """Validates data quality"""

    def __init__(self, config: Dict):
        """Initialize quality checker"""
        self.config = config
        dq_config = config.get("data_quality", {})

        # Validation settings
        validation_config = dq_config.get("validation", {})
        self.null_threshold = validation_config.get("max_null_percentage", 0.1)

        # Outlier detection
        outlier_config = dq_config.get("outlier_detection", {})
        self.outlier_enabled = outlier_config.get("enabled", True)
        self.outlier_method = outlier_config.get("method", "iqr")
        self.outlier_threshold = outlier_config.get("threshold", 3.0)

    def validate_message(self, message: Dict) -> tuple[bool, Optional[str]]:
        """
        Validate message quality

        Args:
            message: Message to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if "equipment_id" not in message:
            return False, "Missing equipment_id"

        if "timestamp" not in message:
            return False, "Missing timestamp"

        if "sensor_data" not in message:
            return False, "Missing sensor_data"

        sensor_data = message["sensor_data"]

        # Check for excessive nulls
        null_count = sum(1 for v in sensor_data.values() if v is None)
        null_percentage = null_count / len(sensor_data) if sensor_data else 0

        if null_percentage > self.null_threshold:
            return False, f"Excessive null values: {null_percentage:.2%}"

        # Check for outliers if enabled
        if self.outlier_enabled:
            outlier_result = self._detect_outliers(sensor_data)
            if outlier_result:
                return False, f"Outliers detected: {outlier_result}"

        return True, None

    def _detect_outliers(self, sensor_data: Dict) -> Optional[str]:
        """Detect outliers in sensor data"""
        # This is a simplified implementation
        # In production, use historical data for better outlier detection

        numeric_values = [
            v for v in sensor_data.values() if isinstance(v, (int, float))
        ]

        if len(numeric_values) < 3:
            return None

        import numpy as np

        if self.outlier_method == "iqr":
            q1 = np.percentile(numeric_values, 25)
            q3 = np.percentile(numeric_values, 75)
            iqr = q3 - q1

            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr

            outliers = [
                k
                for k, v in sensor_data.items()
                if isinstance(v, (int, float)) and (v < lower_bound or v > upper_bound)
            ]

            if outliers:
                return f"IQR outliers in {outliers}"

        return None
