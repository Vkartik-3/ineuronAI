"""
Main entry point for Data Generator
Orchestrates multiple equipment simulations with failure injection
"""

import argparse
import logging
import time
import signal
import sys
import random
from typing import List
from datetime import datetime

from utils.config_loader import ConfigLoader
from simulator.equipment_simulator import Equipment
from publisher.kafka_publisher import SensorDataPublisher, MockPublisher

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataGeneratorOrchestrator:
    """Orchestrates multiple equipment simulations"""

    def __init__(self, config_loader: ConfigLoader, use_mock: bool = False):
        """
        Initialize orchestrator

        Args:
            config_loader: Configuration loader instance
            use_mock: Use mock publisher instead of real Kafka
        """
        self.config_loader = config_loader
        self.configs = config_loader.load_all_configs()

        # Initialize publisher
        if use_mock:
            self.publisher = MockPublisher(self.configs["kafka"])
            logger.info("Using MOCK publisher (Kafka not required)")
        else:
            self.publisher = SensorDataPublisher(self.configs["kafka"])
            logger.info("Using REAL Kafka publisher")

        self.equipment_list: List[Equipment] = []
        self.running = True
        self.start_time = datetime.utcnow()

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.running = False

    def create_equipment_fleet(self, equipment_configs: List[dict]) -> None:
        """
        Create a fleet of equipment simulators

        Args:
            equipment_configs: List of equipment configuration dictionaries
                Each dict should have: equipment_id, equipment_type, inject_failure, failure_mode
        """
        for config in equipment_configs:
            equipment_id = config["equipment_id"]
            equipment_type = config["equipment_type"]

            # Create equipment
            equipment = Equipment(
                equipment_id=equipment_id,
                equipment_type=equipment_type,
                equipment_config=self.configs["equipment"],
                degradation_config=self.configs["degradation"],
                publisher=self.publisher,
            )

            # Inject failure if specified
            if config.get("inject_failure", False):
                failure_mode = config.get("failure_mode")
                if failure_mode:
                    equipment.inject_failure(failure_mode)
                else:
                    # Random failure mode
                    available_failures = self.config_loader.get_failure_modes(
                        self.configs["degradation"], equipment_type
                    )
                    if available_failures:
                        random_failure = random.choice(available_failures)
                        equipment.inject_failure(random_failure)

            self.equipment_list.append(equipment)
            logger.info(f"Added equipment {equipment_id} to fleet")

        logger.info(f"Created fleet of {len(self.equipment_list)} equipment")

    def run(self, sampling_interval: float = 1.0, max_cycles: int = None):
        """
        Run the data generation loop

        Args:
            sampling_interval: Time between samples in seconds
            max_cycles: Maximum cycles to run (None = infinite)
        """
        logger.info(
            f"Starting data generation. Sampling interval: {sampling_interval}s"
        )

        cycle = 0

        try:
            while self.running:
                cycle_start = time.time()

                # Generate data for all equipment
                active_equipment = [
                    eq for eq in self.equipment_list if not eq.is_failed
                ]

                if not active_equipment:
                    logger.warning("All equipment has failed. Stopping simulation.")
                    break

                for equipment in active_equipment:
                    equipment.generate_and_publish()

                cycle += 1

                # Log progress
                if cycle % 100 == 0:
                    self._log_progress(cycle)

                # Check max cycles
                if max_cycles and cycle >= max_cycles:
                    logger.info(f"Reached maximum cycles ({max_cycles}). Stopping.")
                    break

                # Sleep to maintain sampling rate
                elapsed = time.time() - cycle_start
                sleep_time = max(0, sampling_interval - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)

        finally:
            self._shutdown()

    def _log_progress(self, cycle: int):
        """Log progress statistics"""
        total_messages = sum(eq.total_messages_sent for eq in self.equipment_list)
        active_count = sum(1 for eq in self.equipment_list if not eq.is_failed)
        failed_count = sum(1 for eq in self.equipment_list if eq.is_failed)
        degraded_count = sum(
            1
            for eq in self.equipment_list
            if eq.has_failure_injected and not eq.is_failed
        )

        runtime = (datetime.utcnow() - self.start_time).total_seconds()
        messages_per_sec = total_messages / runtime if runtime > 0 else 0

        logger.info(
            f"Cycle {cycle} | Active: {active_count} | Degraded: {degraded_count} | "
            f"Failed: {failed_count} | Total messages: {total_messages} | "
            f"Rate: {messages_per_sec:.1f} msg/s"
        )

        # Publisher stats
        pub_stats = self.publisher.get_stats()
        logger.info(f"Publisher stats: {pub_stats}")

    def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down data generator...")

        # Print final statistics
        logger.info("=" * 80)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 80)

        for equipment in self.equipment_list:
            status = equipment.get_status()
            logger.info(f"Equipment {status['equipment_id']}: {status}")

        # Close publisher
        self.publisher.close()

        logger.info("Shutdown complete")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance Data Generator"
    )

    parser.add_argument(
        "--num-equipment",
        type=int,
        default=5,
        help="Number of equipment to simulate (default: 5)",
    )

    parser.add_argument(
        "--equipment-type",
        type=str,
        default="turbofan_engine",
        choices=["turbofan_engine", "pump", "compressor"],
        help="Type of equipment to simulate (default: turbofan_engine)",
    )

    parser.add_argument(
        "--failure-probability",
        type=float,
        default=0.3,
        help="Probability of injecting failure into equipment (default: 0.3)",
    )

    parser.add_argument(
        "--sampling-interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Maximum cycles to run (default: infinite)",
    )

    parser.add_argument(
        "--mock", action="store_true", help="Use mock publisher (no Kafka required)"
    )

    parser.add_argument(
        "--config-dir", type=str, default=None, help="Path to configuration directory"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()

    logger.info("=" * 80)
    logger.info("PREDICTIVE MAINTENANCE - DATA GENERATOR")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Number of equipment: {args.num_equipment}")
    logger.info(f"  Equipment type: {args.equipment_type}")
    logger.info(f"  Failure probability: {args.failure_probability}")
    logger.info(f"  Sampling interval: {args.sampling_interval}s")
    logger.info(f"  Max cycles: {args.max_cycles if args.max_cycles else 'Infinite'}")
    logger.info(f"  Mock mode: {args.mock}")
    logger.info("=" * 80)

    # Load configurations
    config_loader = ConfigLoader(args.config_dir)

    # Create orchestrator
    orchestrator = DataGeneratorOrchestrator(config_loader, use_mock=args.mock)

    # Create equipment fleet
    equipment_configs = []
    for i in range(args.num_equipment):
        equipment_id = f"{args.equipment_type.upper()}_{i + 1:03d}"

        # Decide if this equipment should have a failure
        inject_failure = random.random() < args.failure_probability

        config = {
            "equipment_id": equipment_id,
            "equipment_type": args.equipment_type,
            "inject_failure": inject_failure,
            "failure_mode": None,  # Will be randomly selected
        }
        equipment_configs.append(config)

    orchestrator.create_equipment_fleet(equipment_configs)

    # Run simulation
    orchestrator.run(
        sampling_interval=args.sampling_interval, max_cycles=args.max_cycles
    )

    logger.info("Data generator exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
