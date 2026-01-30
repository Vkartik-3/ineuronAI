"""
Kafka Streamer for NASA C-MAPSS Dataset

Replays C-MAPSS turbofan engine data chronologically to Kafka,
simulating real-time sensor streaming.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pathlib import Path
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError
import yaml

from .cmapss_loader import CMAPSSLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CMAPSSKafkaStreamer:
    """
    Streams NASA C-MAPSS turbofan engine data to Kafka topic,
    simulating real-time sensor readings.

    Features:
    - Chronological replay of engine degradation trajectories
    - Configurable streaming rate (readings per second)
    - Multi-engine concurrent streaming
    - Adds timestamp metadata for each reading
    - Publishes to raw_sensor_data topic
    """

    def __init__(
        self,
        dataset_path: str = "../archive/CMaps",
        dataset_id: str = "FD001",
        kafka_config_path: str = "config/kafka_config.yaml",
        use_train_data: bool = True,
    ):
        """
        Initialize C-MAPSS Kafka streamer.

        Args:
            dataset_path: Path to C-MAPSS dataset
            dataset_id: Dataset ID (FD001, FD002, FD003, FD004)
            kafka_config_path: Path to Kafka configuration
            use_train_data: If True, stream training data; else test data
        """
        self.dataset_path = dataset_path
        self.dataset_id = dataset_id
        self.use_train_data = use_train_data

        # Load Kafka configuration
        config_path = Path(kafka_config_path)
        if config_path.exists():
            with open(config_path, "r") as f:
                self.kafka_config = yaml.safe_load(f)
        else:
            # Default configuration
            self.kafka_config = {
                "bootstrap_servers": ["localhost:9092"],
                "topic": "raw_sensor_data",
                "streaming_rate": 1.0,  # readings per second per engine
            }
            logger.warning(f"Config file not found, using defaults")

        # Initialize C-MAPSS loader
        self.loader = CMAPSSLoader(dataset_path, dataset_id)

        # Load data
        if use_train_data:
            self.data = self.loader.load_train_data()
            logger.info(f"Loaded training data: {len(self.data)} records")
        else:
            self.data = self.loader.load_test_data()
            logger.info(f"Loaded test data: {len(self.data)} records")

        # Initialize Kafka producer
        self.producer = None
        self._connect_kafka()

        logger.info(
            f"CMAPSSKafkaStreamer initialized for {dataset_id} "
            f"({'train' if use_train_data else 'test'} data)"
        )

    def _connect_kafka(self) -> None:
        """Initialize Kafka producer connection"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_config["bootstrap_servers"],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                compression_type="gzip",
                acks="all",
                retries=3,
            )
            logger.info(f"Connected to Kafka: {self.kafka_config['bootstrap_servers']}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise

    def stream_all_engines(
        self,
        rate_per_engine: float = 1.0,
        start_time: Optional[datetime] = None,
        loop: bool = False,
    ) -> None:
        """
        Stream all engines concurrently in chronological order.

        Args:
            rate_per_engine: Readings per second per engine
            start_time: Starting timestamp (default: now)
            loop: If True, restart from beginning after completion
        """
        if start_time is None:
            start_time = datetime.now()

        topic = self.kafka_config["topic"]

        logger.info(
            f"Starting stream: {self.data['unit_id'].nunique()} engines, "
            f"rate={rate_per_engine} msg/sec/engine"
        )

        try:
            while True:
                messages_sent = 0
                start_stream_time = time.time()

                # Group by time cycle for chronological streaming
                max_cycle = self.data["time_cycle"].max()

                for cycle in range(1, max_cycle + 1):
                    cycle_data = self.data[self.data["time_cycle"] == cycle]

                    if len(cycle_data) == 0:
                        continue

                    # Calculate timestamp for this cycle
                    timestamp = start_time + timedelta(seconds=cycle / rate_per_engine)

                    # Send all engines at this cycle
                    for _, row in cycle_data.iterrows():
                        message = self._create_message(row, timestamp)

                        try:
                            future = self.producer.send(topic, message)
                            future.get(timeout=10)  # Wait for confirmation
                            messages_sent += 1

                            if messages_sent % 100 == 0:
                                logger.info(
                                    f"Sent {messages_sent} messages (cycle {cycle}/{max_cycle})"
                                )

                        except KafkaError as e:
                            logger.error(f"Failed to send message: {e}")

                    # Rate limiting: sleep to maintain desired rate
                    expected_time = cycle / rate_per_engine
                    actual_time = time.time() - start_stream_time
                    sleep_time = expected_time - actual_time

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                logger.info(f"✅ Completed streaming: {messages_sent} messages sent")

                if not loop:
                    break

                logger.info("Restarting stream from beginning...")
                start_time = datetime.now()

        except KeyboardInterrupt:
            logger.info("Streaming interrupted by user")
        finally:
            self.close()

    def stream_single_engine(
        self, unit_id: int, rate: float = 1.0, start_time: Optional[datetime] = None
    ) -> None:
        """
        Stream data for a single engine.

        Args:
            unit_id: Engine unit ID to stream
            rate: Readings per second
            start_time: Starting timestamp (default: now)
        """
        if start_time is None:
            start_time = datetime.now()

        engine_data = self.data[self.data["unit_id"] == unit_id].sort_values(
            "time_cycle"
        )

        if len(engine_data) == 0:
            logger.error(f"Engine {unit_id} not found in dataset")
            return

        topic = self.kafka_config["topic"]

        logger.info(
            f"Streaming engine {unit_id}: {len(engine_data)} cycles, "
            f"rate={rate} msg/sec"
        )

        try:
            for idx, (_, row) in enumerate(engine_data.iterrows()):
                timestamp = start_time + timedelta(seconds=idx / rate)
                message = self._create_message(row, timestamp)

                try:
                    future = self.producer.send(topic, message)
                    future.get(timeout=10)

                    if (idx + 1) % 10 == 0:
                        logger.info(f"Sent cycle {idx + 1}/{len(engine_data)}")

                except KafkaError as e:
                    logger.error(f"Failed to send message: {e}")

                # Rate limiting
                time.sleep(1.0 / rate)

            logger.info(f"✅ Completed streaming engine {unit_id}")

        except KeyboardInterrupt:
            logger.info("Streaming interrupted")
        finally:
            self.close()

    def _create_message(self, row: pd.Series, timestamp: datetime) -> Dict:
        """
        Create Kafka message from data row.

        Args:
            row: DataFrame row with sensor data
            timestamp: Timestamp for this reading

        Returns:
            Dictionary formatted for Kafka
        """
        message = {
            "equipment_id": f"ENGINE_{int(row['unit_id']):04d}",
            "unit_id": int(row["unit_id"]),
            "equipment_type": "turbofan_engine",
            "timestamp": timestamp.isoformat(),
            "time_cycle": int(row["time_cycle"]),
            "dataset": self.dataset_id,
            "data_source": "cmapss",
            # Operating settings
            "operating_settings": {
                "op_setting_1": float(row["op_setting_1"]),
                "op_setting_2": float(row["op_setting_2"]),
                "op_setting_3": float(row["op_setting_3"]),
            },
            # Sensor readings (all 21 sensors)
            "sensors": {},
        }

        # Add all sensor readings
        for i in range(1, 22):
            sensor_col = f"sensor_{i}"
            message["sensors"][sensor_col] = float(row[sensor_col])

        # Add RUL if available (for evaluation)
        if "rul" in row.index:
            message["rul"] = float(row["rul"])

        return message

    def get_engine_list(self) -> List[int]:
        """
        Get list of all engine unit IDs in the dataset.

        Returns:
            List of engine unit IDs
        """
        return sorted(self.data["unit_id"].unique().tolist())

    def get_streaming_stats(self) -> Dict:
        """
        Get statistics about the streaming dataset.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "total_engines": self.data["unit_id"].nunique(),
            "total_cycles": len(self.data),
            "max_cycle": self.data["time_cycle"].max(),
            "avg_cycles_per_engine": len(self.data) / self.data["unit_id"].nunique(),
            "dataset": self.dataset_id,
            "data_type": "train" if self.use_train_data else "test",
        }

    def close(self) -> None:
        """Close Kafka producer connection"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")


def main():
    """Test Kafka streamer"""
    import argparse

    parser = argparse.ArgumentParser(description="Stream C-MAPSS data to Kafka")
    parser.add_argument("--dataset", default="FD001", help="Dataset ID (FD001-FD004)")
    parser.add_argument(
        "--rate", type=float, default=1.0, help="Readings per second per engine"
    )
    parser.add_argument("--engine", type=int, help="Stream single engine (by unit_id)")
    parser.add_argument(
        "--train", action="store_true", help="Use training data (default: test data)"
    )
    parser.add_argument(
        "--loop", action="store_true", help="Loop streaming continuously"
    )

    args = parser.parse_args()

    print("\n=== C-MAPSS Kafka Streamer ===\n")

    # Initialize streamer
    streamer = CMAPSSKafkaStreamer(
        dataset_path="../archive/CMaps",
        dataset_id=args.dataset,
        use_train_data=args.train,
    )

    # Print stats
    stats = streamer.get_streaming_stats()
    print("Streaming Configuration:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  rate: {args.rate} msg/sec/engine")
    print()

    # Stream data
    if args.engine:
        print(f"Streaming single engine: {args.engine}")
        streamer.stream_single_engine(args.engine, rate=args.rate)
    else:
        print(f"Streaming all engines...")
        streamer.stream_all_engines(rate_per_engine=args.rate, loop=args.loop)


if __name__ == "__main__":
    main()
