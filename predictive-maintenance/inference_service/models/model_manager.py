"""
Model Manager - Load and cache models for inference
"""

import os
import pickle
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import tensorflow as tf
import numpy as np


logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model loading, caching, and lifecycle.
    Implements singleton pattern for shared model cache.
    """

    _instance = None
    _models: Dict[str, Any] = {}
    _model_metadata: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model manager"""
        self.initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize models from configuration

        Args:
            config: Configuration dictionary with model paths
        """
        if self.initialized:
            logger.info("Model manager already initialized")
            return

        logger.info("Initializing model manager...")

        # Load LSTM model
        lstm_config = config["models"]["lstm_rul"]
        if lstm_config.get("warm_start", False):
            self.load_lstm_model(
                lstm_config["path"], lstm_config["name"], lstm_config["version"]
            )

        # Load Random Forest model
        rf_config = config["models"]["random_forest_health"]
        if rf_config.get("warm_start", False):
            self.load_sklearn_model(
                rf_config["path"],
                rf_config["name"],
                rf_config["version"],
                model_key="random_forest",
            )

        self.initialized = True
        logger.info("Model manager initialized successfully")

    def load_lstm_model(
        self, model_path: str, model_name: str, version: str
    ) -> tf.keras.Model:
        """
        Load TensorFlow/Keras LSTM model

        Args:
            model_path: Path to saved model
            model_name: Model name
            version: Model version

        Returns:
            Loaded TensorFlow model
        """
        try:
            logger.info(f"Loading LSTM model from {model_path}")
            model = tf.keras.models.load_model(model_path)

            self._models["lstm"] = model
            self._model_metadata["lstm"] = {
                "name": model_name,
                "version": version,
                "type": "lstm",
                "loaded": True,
                "loaded_at": datetime.utcnow(),
                "path": model_path,
            }

            logger.info(f"LSTM model loaded successfully: {model_name} {version}")
            return model

        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            self._model_metadata["lstm"] = {
                "name": model_name,
                "version": version,
                "type": "lstm",
                "loaded": False,
                "error": str(e),
            }
            raise

    def load_sklearn_model(
        self,
        model_path: str,
        model_name: str,
        version: str,
        model_key: str = "random_forest",
    ) -> Any:
        """
        Load scikit-learn model from pickle

        Args:
            model_path: Path to saved model (pickle file)
            model_name: Model name
            version: Model version
            model_key: Key for storing model in cache

        Returns:
            Loaded scikit-learn model
        """
        try:
            logger.info(f"Loading sklearn model from {model_path}")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            self._models[model_key] = model
            self._model_metadata[model_key] = {
                "name": model_name,
                "version": version,
                "type": "random_forest",
                "loaded": True,
                "loaded_at": datetime.utcnow(),
                "path": model_path,
            }

            logger.info(f"Sklearn model loaded successfully: {model_name} {version}")
            return model

        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            self._model_metadata[model_key] = {
                "name": model_name,
                "version": version,
                "type": "random_forest",
                "loaded": False,
                "error": str(e),
            }
            raise

    def get_model(self, model_key: str) -> Optional[Any]:
        """
        Get cached model

        Args:
            model_key: Model identifier ('lstm', 'random_forest')

        Returns:
            Cached model or None if not loaded
        """
        return self._models.get(model_key)

    def get_model_metadata(self, model_key: str) -> Optional[Dict[str, Any]]:
        """
        Get model metadata

        Args:
            model_key: Model identifier

        Returns:
            Model metadata dictionary
        """
        return self._model_metadata.get(model_key)

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all models and their metadata

        Returns:
            Dictionary of model metadata
        """
        return self._model_metadata.copy()

    def reload_model(self, model_key: str) -> None:
        """
        Reload a specific model

        Args:
            model_key: Model identifier to reload
        """
        metadata = self._model_metadata.get(model_key)
        if not metadata:
            raise ValueError(f"Model {model_key} not found")

        logger.info(f"Reloading model: {model_key}")

        if metadata["type"] == "lstm":
            self.load_lstm_model(
                metadata["path"], metadata["name"], metadata["version"]
            )
        elif metadata["type"] == "random_forest":
            self.load_sklearn_model(
                metadata["path"],
                metadata["name"],
                metadata["version"],
                model_key=model_key,
            )
        else:
            raise ValueError(f"Unknown model type: {metadata['type']}")

    def unload_model(self, model_key: str) -> None:
        """
        Unload model from memory

        Args:
            model_key: Model identifier to unload
        """
        if model_key in self._models:
            logger.info(f"Unloading model: {model_key}")
            del self._models[model_key]

            if model_key in self._model_metadata:
                self._model_metadata[model_key]["loaded"] = False
                self._model_metadata[model_key]["unloaded_at"] = datetime.utcnow()

    def is_loaded(self, model_key: str) -> bool:
        """
        Check if model is loaded

        Args:
            model_key: Model identifier

        Returns:
            True if model is loaded and ready
        """
        return model_key in self._models and self._models[model_key] is not None

    def get_model_info(self) -> Dict[str, bool]:
        """
        Get model load status for all models

        Returns:
            Dictionary mapping model keys to load status
        """
        return {
            key: metadata.get("loaded", False)
            for key, metadata in self._model_metadata.items()
        }


# Global model manager instance
model_manager = ModelManager()
