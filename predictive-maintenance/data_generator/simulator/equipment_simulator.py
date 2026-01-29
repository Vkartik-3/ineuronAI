"""
Equipment Simulator
Manages individual equipment lifecycle including normal operation and failure injection
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import time

from simulator.sensor_simulator import SensorDataGenerator
from simulator.degradation_engine import DegradationEngine, FailureMode
from publisher.kafka_publisher import SensorDataPublisher

logger = logging.getLogger(__name__)


class Equipment:
    """Represents a single piece of equipment being simulated"""

    def __init__(
        self,
        equipment_id: str,
        equipment_type: str,
        equipment_config: Dict,
        degradation_config: Dict,
        publisher: SensorDataPublisher,
    ):
        """
        Initialize equipment simulator

        Args:
            equipment_id: Unique identifier for this equipment
            equipment_type: Type of equipment (turbofan_engine, pump, compressor)
            equipment_config: Equipment configuration
            degradation_config: Degradation configuration
            publisher: Kafka publisher instance
        """
        self.equipment_id = equipment_id
        self.equipment_type = equipment_type
        self.publisher = publisher

        # Initialize components
        self.sensor_generator = SensorDataGenerator(equipment_type, equipment_config)
        self.degradation_engine = DegradationEngine(degradation_config)

        # State
        self.current_cycle = 0
        self.failure_mode: Optional[FailureMode] = None
        self.is_failed = False
        self.has_failure_injected = False

        # Metrics
        self.total_messages_sent = 0
        self.start_time = datetime.utcnow()

        logger.info(f"Initialized equipment {equipment_id} (type: {equipment_type})")

    def inject_failure(self, failure_mode_name: str) -> bool:
        """
        Inject a failure mode into the equipment

        Args:
            failure_mode_name: Name of the failure mode to inject

        Returns:
            True if successful, False otherwise
        """
        if self.has_failure_injected:
            logger.warning(
                f"Equipment {self.equipment_id} already has a failure injected"
            )
            return False

        failure_mode = self.degradation_engine.create_failure_mode(
            self.equipment_type, failure_mode_name, self.current_cycle
        )

        if failure_mode:
            self.failure_mode = failure_mode
            self.has_failure_injected = True
            logger.info(
                f"Injected failure mode '{failure_mode_name}' into {self.equipment_id}. "
                f"RUL: {failure_mode.rul_remaining} cycles"
            )
            return True
        else:
            logger.error(f"Failed to inject failure mode '{failure_mode_name}'")
            return False

    def generate_and_publish(self) -> bool:
        """
        Generate sensor data for current cycle and publish to Kafka

        Returns:
            True if successful, False otherwise
        """
        # Check if equipment has failed
        if self.is_failed:
            logger.warning(
                f"Equipment {self.equipment_id} has failed. No more data generated."
            )
            return False

        # Calculate degradation factors if failure is active
        degradation_factors = {}
        noise_multiplier = 1.0

        if self.failure_mode:
            degradation_factors = self.degradation_engine.calculate_degradation_factors(
                self.failure_mode, self.current_cycle
            )
            noise_multiplier = self.degradation_engine.get_noise_multiplier(
                self.failure_mode
            )

            # Update RUL
            self.failure_mode = self.degradation_engine.update_rul(self.failure_mode)

            # Check if failed
            if self.degradation_engine.is_failed(self.failure_mode):
                self.is_failed = True
                logger.warning(
                    f"Equipment {self.equipment_id} has FAILED at cycle {self.current_cycle}. "
                    f"Failure mode: {self.failure_mode.name}"
                )

        # Generate sensor data
        data = self.sensor_generator.generate(
            self.equipment_id, self.current_cycle, degradation_factors, noise_multiplier
        )

        # Add failure information to metadata (for ground truth)
        if self.failure_mode:
            data["metadata"]["failure_mode"] = self.failure_mode.name
            data["metadata"]["rul_remaining"] = self.failure_mode.rul_remaining
            data["metadata"]["degradation_stage"] = (
                self.failure_mode.get_degradation_stage()
            )
            data["metadata"]["is_degraded"] = True
        else:
            data["metadata"]["failure_mode"] = None
            data["metadata"]["rul_remaining"] = None
            data["metadata"]["degradation_stage"] = "healthy"
            data["metadata"]["is_degraded"] = False

        # Publish to Kafka
        success = self.publisher.publish_sensor_data(data, self.equipment_id)

        if success:
            self.total_messages_sent += 1
            self.current_cycle += 1

        return success

    def get_status(self) -> Dict:
        """Get current equipment status"""
        status = {
            "equipment_id": self.equipment_id,
            "equipment_type": self.equipment_type,
            "current_cycle": self.current_cycle,
            "is_failed": self.is_failed,
            "has_failure_injected": self.has_failure_injected,
            "total_messages_sent": self.total_messages_sent,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
        }

        if self.failure_mode:
            status["failure_mode"] = self.failure_mode.name
            status["rul_remaining"] = self.failure_mode.rul_remaining
            status["degradation_stage"] = self.failure_mode.get_degradation_stage()

        return status
