"""
Degradation Pattern Engine
Simulates various failure modes and degradation patterns for equipment
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DegradationPattern(Enum):
    """Types of degradation patterns"""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"
    OSCILLATING = "oscillating"
    COMBINED = "combined"
    NONE = "none"


@dataclass
class FailureMode:
    """Represents a specific failure mode"""

    name: str
    equipment_type: str
    primary_sensors: List[str]
    pattern_type: DegradationPattern
    rul_remaining: int  # Remaining Useful Life in cycles
    total_rul: int  # Total RUL when failure was injected
    severity_multiplier: float
    pattern_params: Dict
    current_cycle: int = 0

    def get_degradation_stage(self) -> str:
        """Determine current degradation stage"""
        rul_percentage = (
            self.rul_remaining / self.total_rul if self.total_rul > 0 else 0
        )

        if rul_percentage > 0.7:
            return "early"
        elif rul_percentage > 0.3:
            return "middle"
        elif rul_percentage > 0.1:
            return "late"
        else:
            return "critical"


class LinearDegradation:
    """Linear degradation pattern - steady wear over time"""

    @staticmethod
    def calculate(
        cycle: int, start_cycle: int, rate: float, rul_remaining: int, total_rul: int
    ) -> float:
        """
        Calculate linear degradation factor

        Args:
            cycle: Current cycle
            start_cycle: Cycle when degradation started
            rate: Degradation rate per cycle
            rul_remaining: Remaining useful life
            total_rul: Total RUL

        Returns:
            Degradation factor (0 to 1)
        """
        if cycle < start_cycle:
            return 0.0

        # Linear progression from 0 to 1
        cycles_degraded = cycle - start_cycle
        total_degradation_cycles = total_rul

        if total_degradation_cycles <= 0:
            return 0.0

        factor = (cycles_degraded / total_degradation_cycles) * rate * 100
        return np.clip(factor, 0.0, 1.0)


class ExponentialDegradation:
    """Exponential degradation pattern - accelerating failure"""

    @staticmethod
    def calculate(
        cycle: int,
        start_cycle: int,
        growth_rate: float,
        rul_remaining: int,
        total_rul: int,
    ) -> float:
        """
        Calculate exponential degradation factor

        Args:
            cycle: Current cycle
            start_cycle: Cycle when degradation started
            growth_rate: Exponential growth rate
            rul_remaining: Remaining useful life
            total_rul: Total RUL

        Returns:
            Degradation factor (0 to 1)
        """
        if cycle < start_cycle:
            return 0.0

        cycles_degraded = cycle - start_cycle

        # Exponential growth: factor = (1 - e^(-growth_rate * t))
        factor = 1 - np.exp(-growth_rate * cycles_degraded / 10)
        return np.clip(factor, 0.0, 1.0)


class StepDegradation:
    """Step degradation pattern - sudden performance drop"""

    @staticmethod
    def calculate(cycle: int, occurrence_cycle: int, magnitude: float) -> float:
        """
        Calculate step degradation factor

        Args:
            cycle: Current cycle
            occurrence_cycle: Cycle when step occurs
            magnitude: Magnitude of the step (0 to 1)

        Returns:
            Degradation factor (0 to 1)
        """
        if cycle < occurrence_cycle:
            return 0.0
        else:
            return magnitude


class OscillatingDegradation:
    """Oscillating degradation pattern - periodic issues"""

    @staticmethod
    def calculate(
        cycle: int, start_cycle: int, amplitude: float, frequency: float
    ) -> float:
        """
        Calculate oscillating degradation factor

        Args:
            cycle: Current cycle
            start_cycle: Cycle when oscillation starts
            amplitude: Oscillation amplitude
            frequency: Oscillation frequency

        Returns:
            Degradation factor (0 to 1)
        """
        if cycle < start_cycle:
            return 0.0

        cycles_active = cycle - start_cycle
        # Sine wave oscillation with increasing baseline
        baseline = 0.3 * (cycles_active / 100)  # Slowly increasing baseline
        oscillation = amplitude * np.sin(2 * np.pi * frequency * cycles_active)

        factor = baseline + oscillation
        return np.clip(factor, 0.0, 1.0)


class DegradationEngine:
    """Main engine for managing equipment degradation"""

    def __init__(self, degradation_config: Dict):
        """
        Initialize degradation engine

        Args:
            degradation_config: Configuration dictionary for degradation patterns
        """
        self.config = degradation_config
        self.patterns_config = degradation_config.get("degradation_patterns", {})
        self.failure_modes_config = degradation_config.get("failure_modes", {})
        self.progression_config = degradation_config.get("progression", {})

    def create_failure_mode(
        self, equipment_type: str, failure_mode_name: str, current_cycle: int = 0
    ) -> Optional[FailureMode]:
        """
        Create a failure mode instance for an equipment

        Args:
            equipment_type: Type of equipment
            failure_mode_name: Name of the failure mode
            current_cycle: Current operational cycle

        Returns:
            FailureMode instance or None if not found
        """
        if equipment_type not in self.failure_modes_config:
            logger.warning(
                f"Equipment type {equipment_type} not found in failure modes config"
            )
            return None

        failure_config = self.failure_modes_config[equipment_type].get(
            failure_mode_name
        )
        if not failure_config:
            logger.warning(
                f"Failure mode {failure_mode_name} not found for {equipment_type}"
            )
            return None

        # Get pattern type
        pattern_name = failure_config.get("degradation_pattern", "linear")
        pattern_type = DegradationPattern(pattern_name)

        # Get pattern parameters
        pattern_config = self.patterns_config.get(pattern_name, {})
        pattern_params = pattern_config.get("parameters", {})

        # Determine RUL
        rul_range = failure_config.get("typical_rul_range", [100, 300])
        total_rul = np.random.randint(rul_range[0], rul_range[1])
        rul_remaining = total_rul

        # Adjust for current cycle if equipment already running
        if current_cycle > 0:
            # Randomly decide if degradation has started
            if np.random.random() < 0.3:  # 30% chance degradation already started
                cycles_elapsed = np.random.randint(
                    0, min(current_cycle, total_rul // 2)
                )
                rul_remaining = max(total_rul - cycles_elapsed, 10)

        failure_mode = FailureMode(
            name=failure_mode_name,
            equipment_type=equipment_type,
            primary_sensors=failure_config.get("primary_sensors", []),
            pattern_type=pattern_type,
            rul_remaining=rul_remaining,
            total_rul=total_rul,
            severity_multiplier=failure_config.get("severity_multiplier", 1.0),
            pattern_params=pattern_params,
            current_cycle=current_cycle,
        )

        return failure_mode

    def calculate_degradation_factors(
        self, failure_mode: FailureMode, current_cycle: int
    ) -> Dict[str, float]:
        """
        Calculate degradation factors for all sensors affected by a failure mode

        Args:
            failure_mode: The active failure mode
            current_cycle: Current operational cycle

        Returns:
            Dictionary mapping sensor names to degradation factors (0 to 1)
        """
        degradation_factors = {}

        # Get base degradation factor based on pattern type
        base_factor = self._calculate_pattern_factor(failure_mode, current_cycle)

        # Apply severity multiplier
        base_factor *= failure_mode.severity_multiplier

        # Apply stage-specific adjustments
        stage = failure_mode.get_degradation_stage()
        stage_config = self.progression_config.get(f"{stage}_stage", {})
        stage_deviation = stage_config.get("sensor_deviation", 0.1)

        # Amplify factor based on stage
        amplified_factor = base_factor * (1 + stage_deviation)
        amplified_factor = np.clip(amplified_factor, 0.0, 1.0)

        # Assign factors to primary sensors
        for sensor in failure_mode.primary_sensors:
            # Add some randomness per sensor
            sensor_factor = amplified_factor * np.random.uniform(0.9, 1.1)
            degradation_factors[sensor] = np.clip(sensor_factor, 0.0, 1.0)

        return degradation_factors

    def _calculate_pattern_factor(
        self, failure_mode: FailureMode, current_cycle: int
    ) -> float:
        """Calculate base degradation factor based on pattern type"""

        params = failure_mode.pattern_params
        start_cycle = failure_mode.current_cycle
        rul_remaining = failure_mode.rul_remaining
        total_rul = failure_mode.total_rul

        if failure_mode.pattern_type == DegradationPattern.LINEAR:
            rate = np.random.uniform(
                params.get("rate_range", [0.001, 0.005])[0],
                params.get("rate_range", [0.001, 0.005])[1],
            )
            return LinearDegradation.calculate(
                current_cycle, start_cycle, rate, rul_remaining, total_rul
            )

        elif failure_mode.pattern_type == DegradationPattern.EXPONENTIAL:
            growth_rate = np.random.uniform(
                params.get("growth_rate_range", [0.002, 0.008])[0],
                params.get("growth_rate_range", [0.002, 0.008])[1],
            )
            return ExponentialDegradation.calculate(
                current_cycle, start_cycle, growth_rate, rul_remaining, total_rul
            )

        elif failure_mode.pattern_type == DegradationPattern.STEP:
            occurrence_cycle = start_cycle + (total_rul - rul_remaining)
            magnitude = np.random.uniform(
                params.get("magnitude_range", [0.05, 0.15])[0],
                params.get("magnitude_range", [0.05, 0.15])[1],
            )
            return StepDegradation.calculate(current_cycle, occurrence_cycle, magnitude)

        elif failure_mode.pattern_type == DegradationPattern.OSCILLATING:
            amplitude = np.random.uniform(
                params.get("amplitude_range", [0.02, 0.08])[0],
                params.get("amplitude_range", [0.02, 0.08])[1],
            )
            frequency = np.random.uniform(
                params.get("frequency_range", [0.1, 0.5])[0],
                params.get("frequency_range", [0.1, 0.5])[1],
            )
            return OscillatingDegradation.calculate(
                current_cycle, start_cycle, amplitude, frequency
            )

        elif failure_mode.pattern_type == DegradationPattern.COMBINED:
            # Combine multiple patterns
            factors = []
            components = self.patterns_config.get("combined", {}).get("components", [])
            for component in components:
                pattern_name = component.get("pattern")
                weight = component.get("weight", 0.5)
                # Recursively calculate each component
                temp_params = self.patterns_config.get(pattern_name, {}).get(
                    "parameters", {}
                )
                temp_mode = FailureMode(
                    name=failure_mode.name,
                    equipment_type=failure_mode.equipment_type,
                    primary_sensors=failure_mode.primary_sensors,
                    pattern_type=DegradationPattern(pattern_name),
                    rul_remaining=rul_remaining,
                    total_rul=total_rul,
                    severity_multiplier=1.0,
                    pattern_params=temp_params,
                    current_cycle=start_cycle,
                )
                factor = self._calculate_pattern_factor(temp_mode, current_cycle)
                factors.append(factor * weight)

            return sum(factors)

        else:
            return 0.0

    def get_noise_multiplier(self, failure_mode: FailureMode) -> float:
        """Get noise multiplier based on degradation stage"""
        stage = failure_mode.get_degradation_stage()
        stage_config = self.progression_config.get(f"{stage}_stage", {})
        return stage_config.get("noise_increase", 1.0)

    def update_rul(self, failure_mode: FailureMode) -> FailureMode:
        """Update RUL (decrement by 1 cycle)"""
        failure_mode.rul_remaining = max(0, failure_mode.rul_remaining - 1)
        return failure_mode

    def is_failed(self, failure_mode: FailureMode) -> bool:
        """Check if equipment has failed"""
        return failure_mode.rul_remaining <= 0
