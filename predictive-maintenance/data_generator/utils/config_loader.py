"""
Configuration Loader
Loads and validates YAML configuration files
"""

import yaml
import os
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads configuration files from the config directory"""

    def __init__(self, config_dir: str = None):
        """
        Initialize config loader

        Args:
            config_dir: Path to configuration directory
        """
        if config_dir is None:
            # Default to config/ subdirectory at data_generator level
            # This file is in utils/, so go up one level to data_generator/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(
                current_dir
            )  # Go up from utils/ to data_generator/
            config_dir = os.path.join(parent_dir, "config")

        self.config_dir = config_dir
        logger.info(f"Config loader initialized. Config directory: {config_dir}")

    def load_yaml(self, filename: str) -> Dict:
        """
        Load a YAML configuration file

        Args:
            filename: Name of the YAML file (with or without .yaml extension)

        Returns:
            Dictionary containing configuration
        """
        if not filename.endswith(".yaml"):
            filename += ".yaml"

        filepath = os.path.join(self.config_dir, filename)

        try:
            with open(filepath, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {filepath}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {filepath}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading {filepath}: {e}")
            return {}

    def load_all_configs(self) -> Dict:
        """
        Load all configuration files

        Returns:
            Dictionary containing all configurations
        """
        configs = {
            "equipment": self.load_yaml("equipment_config.yaml"),
            "degradation": self.load_yaml("degradation_config.yaml"),
            "kafka": self.load_yaml("kafka_config.yaml"),
        }

        logger.info("All configurations loaded")
        return configs

    def get_equipment_types(self, equipment_config: Dict) -> list:
        """Get list of available equipment types"""
        return list(equipment_config.get("equipment_types", {}).keys())

    def get_failure_modes(self, degradation_config: Dict, equipment_type: str) -> list:
        """Get list of available failure modes for an equipment type"""
        failure_modes = degradation_config.get("failure_modes", {})
        return list(failure_modes.get(equipment_type, {}).keys())
