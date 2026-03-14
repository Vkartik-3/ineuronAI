"""
Model Manager — Load and cache models for inference.

Supports two loading strategies:
  1. **MLflow Model Registry** (primary) — loads models registered as
     ``models:/<model_name>/<stage>`` (e.g. ``Production``).
  2. **Local file paths** (fallback) — used when MLflow is unreachable or
     the model has not yet been registered.

A background polling thread periodically checks MLflow for new Production
versions and hot-swaps models without downtime.
"""

import os
import pickle
import logging
import threading
import time
from typing import Dict, Optional, Any
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so unit tests can still run when these
# packages are absent.
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore


class ModelManager:
    """
    Manages model loading, caching, and lifecycle.
    Singleton — one instance across the process.
    """

    _instance = None
    _models: Dict[str, Any] = {}
    _model_metadata: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.initialized = False
        self._config: Dict[str, Any] = {}
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_running = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, config: Dict[str, Any]) -> None:
        if self.initialized:
            logger.info("Model manager already initialized")
            return

        self._config = config
        logger.info("Initializing model manager...")

        # MLflow connection (best-effort)
        mlflow_cfg = config.get("mlflow", {})
        self._mlflow_uri = mlflow_cfg.get(
            "tracking_uri",
            os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        )
        self._mlflow_poll_interval = mlflow_cfg.get("poll_interval_seconds", 60)
        self._mlflow_client: Optional[Any] = None

        if mlflow is not None:
            try:
                mlflow.set_tracking_uri(self._mlflow_uri)
                self._mlflow_client = MlflowClient(self._mlflow_uri)
                logger.info("MLflow client connected — %s", self._mlflow_uri)
            except Exception as exc:
                logger.warning("MLflow unavailable, will use local fallback: %s", exc)

        # Load models — try MLflow first, then local paths
        for model_key, model_cfg in config.get("models", {}).items():
            if not model_cfg.get("warm_start", False):
                continue
            self._load_model(model_key, model_cfg)

        self.initialized = True

        # Start background poller
        self._start_polling()

        logger.info("Model manager initialized successfully")

    # ------------------------------------------------------------------
    # Unified loader — MLflow → local fallback
    # ------------------------------------------------------------------

    def _load_model(self, model_key: str, model_cfg: Dict) -> None:
        """Try MLflow registry first; fall back to local path."""
        registry_name = model_cfg.get("registry_name")
        registry_stage = model_cfg.get("registry_stage", "Production")
        model_type = model_cfg.get("type", self._infer_type(model_key))

        # Attempt 1 — MLflow
        if self._mlflow_client and registry_name:
            try:
                self._load_from_mlflow(
                    model_key, registry_name, registry_stage, model_cfg
                )
                return
            except Exception as exc:
                logger.warning(
                    "MLflow load failed for %s (%s/%s): %s — trying local path",
                    model_key,
                    registry_name,
                    registry_stage,
                    exc,
                )

        # Attempt 2 — local path
        local_path = model_cfg.get("path")
        if local_path:
            try:
                if model_type == "lstm":
                    self.load_lstm_model(
                        local_path,
                        model_cfg.get("name", model_key),
                        model_cfg.get("version", "local"),
                    )
                else:
                    self.load_sklearn_model(
                        local_path,
                        model_cfg.get("name", model_key),
                        model_cfg.get("version", "local"),
                        model_key=model_key,
                    )
                return
            except Exception as exc:
                logger.error("Local load also failed for %s: %s", model_key, exc)

        # Neither worked — register as "not loaded"
        self._model_metadata[model_key] = {
            "name": model_cfg.get("name", model_key),
            "version": model_cfg.get("version", "unknown"),
            "type": model_type,
            "loaded": False,
            "error": "No model source available",
        }

    def _load_from_mlflow(
        self,
        model_key: str,
        registry_name: str,
        stage: str,
        model_cfg: Dict,
    ) -> None:
        model_uri = f"models:/{registry_name}/{stage}"
        logger.info("Loading %s from MLflow: %s", model_key, model_uri)

        model = mlflow.pyfunc.load_model(model_uri)

        # Unwrap native model when possible
        native = getattr(model, "_model_impl", model)
        if hasattr(native, "python_model"):
            native = native.python_model

        # Resolve version
        versions = self._mlflow_client.get_latest_versions(
            registry_name, stages=[stage]
        )
        version_str = versions[0].version if versions else "unknown"

        self._models[model_key] = model
        self._model_metadata[model_key] = {
            "name": model_cfg.get("name", model_key),
            "version": version_str,
            "type": self._infer_type(model_key),
            "loaded": True,
            "loaded_at": datetime.utcnow(),
            "source": "mlflow",
            "registry_name": registry_name,
            "registry_stage": stage,
            "mlflow_uri": model_uri,
        }
        logger.info("Loaded %s from MLflow (v%s)", model_key, version_str)

    # ------------------------------------------------------------------
    # Local file loaders (kept for fallback & backwards compat)
    # ------------------------------------------------------------------

    def load_lstm_model(self, model_path: str, model_name: str, version: str):
        try:
            logger.info("Loading LSTM model from %s", model_path)
            if tf is None:
                raise ImportError("tensorflow not installed")
            model = tf.keras.models.load_model(model_path)
            self._models["lstm"] = model
            self._model_metadata["lstm"] = {
                "name": model_name,
                "version": version,
                "type": "lstm",
                "loaded": True,
                "loaded_at": datetime.utcnow(),
                "path": model_path,
                "source": "local",
            }
            logger.info("LSTM model loaded: %s %s", model_name, version)
            return model
        except Exception as exc:
            logger.error("Failed to load LSTM model: %s", exc)
            self._model_metadata["lstm"] = {
                "name": model_name,
                "version": version,
                "type": "lstm",
                "loaded": False,
                "error": str(exc),
            }
            raise

    def load_sklearn_model(
        self,
        model_path: str,
        model_name: str,
        version: str,
        model_key: str = "random_forest",
    ):
        try:
            logger.info("Loading sklearn model from %s", model_path)
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
                "source": "local",
            }
            logger.info("Sklearn model loaded: %s %s", model_name, version)
            return model
        except Exception as exc:
            logger.error("Failed to load sklearn model: %s", exc)
            self._model_metadata[model_key] = {
                "name": model_name,
                "version": version,
                "type": "random_forest",
                "loaded": False,
                "error": str(exc),
            }
            raise

    # ------------------------------------------------------------------
    # Background model polling
    # ------------------------------------------------------------------

    def _start_polling(self):
        if not self._mlflow_client:
            return
        self._poll_running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(
            "MLflow polling thread started (interval=%ds)", self._mlflow_poll_interval
        )

    def _poll_loop(self):
        while self._poll_running:
            time.sleep(self._mlflow_poll_interval)
            try:
                self._check_for_new_versions()
            except Exception as exc:
                logger.warning("Polling error: %s", exc)

    def _check_for_new_versions(self):
        for model_key, meta in list(self._model_metadata.items()):
            registry_name = meta.get("registry_name")
            stage = meta.get("registry_stage", "Production")
            if not registry_name:
                continue
            try:
                versions = self._mlflow_client.get_latest_versions(
                    registry_name, stages=[stage]
                )
                if not versions:
                    continue
                latest_version = versions[0].version
                if latest_version != meta.get("version"):
                    logger.info(
                        "New %s version detected for %s: v%s → v%s — reloading",
                        stage,
                        model_key,
                        meta.get("version"),
                        latest_version,
                    )
                    model_cfg = self._config.get("models", {}).get(model_key, {})
                    self._load_from_mlflow(model_key, registry_name, stage, model_cfg)
            except Exception as exc:
                logger.warning("Version check failed for %s: %s", model_key, exc)

    def stop_polling(self):
        self._poll_running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Public API (unchanged interface)
    # ------------------------------------------------------------------

    def get_model(self, model_key: str) -> Optional[Any]:
        return self._models.get(model_key)

    def get_model_metadata(self, model_key: str) -> Optional[Dict[str, Any]]:
        return self._model_metadata.get(model_key)

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        return self._model_metadata.copy()

    def reload_model(self, model_key: str) -> None:
        """Reload a model — tries MLflow first, then local."""
        meta = self._model_metadata.get(model_key)
        if not meta:
            raise ValueError(f"Model {model_key} not found")
        logger.info("Reloading model: %s", model_key)
        model_cfg = self._config.get("models", {}).get(model_key, {})
        if model_cfg:
            self._load_model(model_key, model_cfg)
        elif meta.get("source") == "local":
            if meta["type"] == "lstm":
                self.load_lstm_model(meta["path"], meta["name"], meta["version"])
            else:
                self.load_sklearn_model(
                    meta["path"], meta["name"], meta["version"], model_key=model_key
                )

    def unload_model(self, model_key: str) -> None:
        if model_key in self._models:
            logger.info("Unloading model: %s", model_key)
            del self._models[model_key]
            if model_key in self._model_metadata:
                self._model_metadata[model_key]["loaded"] = False
                self._model_metadata[model_key]["unloaded_at"] = datetime.utcnow()

    def is_loaded(self, model_key: str) -> bool:
        return model_key in self._models and self._models[model_key] is not None

    def get_model_info(self) -> Dict[str, bool]:
        return {k: v.get("loaded", False) for k, v in self._model_metadata.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_type(model_key: str) -> str:
        if "lstm" in model_key.lower():
            return "lstm"
        return "random_forest"


# Global singleton
model_manager = ModelManager()
