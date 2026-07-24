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
    import torch
    import torch.nn as nn
    TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_DEVICE = None  # type: ignore

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore


def _device_label_for(storage_key: str) -> str:
    """
    Best-effort runtime device label, surfaced in model metadata and the
    /health and /models endpoints — this is the evidence trail for the
    resume's "GPU acceleration" claim: it shows, at runtime, whether each
    loaded model is actually executing on CUDA or falling back to CPU.

    'mlp'  -> the PyTorch device the model is placed on (cuda:N / cpu),
              taken from the same TORCH_DEVICE used to .to() the weights
    'lstm' -> whether TensorFlow can see a GPU on this host
    other  -> 'cpu' (sklearn / random-forest have no GPU execution path)
    """
    if storage_key == "mlp":
        return str(TORCH_DEVICE) if TORCH_DEVICE is not None else "cpu"
    if storage_key == "lstm":
        if tf is None:
            return "unknown"
        try:
            return "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
        except Exception:
            return "cpu"
    return "cpu"


if torch is not None and TORCH_DEVICE is not None:
    if TORCH_DEVICE.type == "cuda":
        try:
            _gpu_name = torch.cuda.get_device_name(TORCH_DEVICE)
        except Exception:
            _gpu_name = "unknown CUDA device"
        logger.info(
            "ModelManager: CUDA available — PyTorch MLP will run on GPU (%s, device=%s)",
            _gpu_name, TORCH_DEVICE,
        )
    else:
        logger.info("ModelManager: CUDA unavailable, running on CPU fallback. (device=%s)", TORCH_DEVICE)


class ModelManager:
    """
    Manages model loading, caching, and lifecycle.

    **Singleton scope — per process, not system-wide.**

    Under Uvicorn with ``workers=4`` (default in inference_config.yaml) the OS
    forks four separate Python processes after the first import. Each process
    constructs its own ``ModelManager`` singleton independently, so:

    * ``_models``         — 4 separate in-memory copies of the loaded weights.
    * ``_model_metadata`` — 4 independent metadata dicts.
    * background polling  — 4 polling threads, one per worker.
    * Redis (PredictionCache) — the ONLY shared state across workers; all
      workers read/write the same Redis instance so cached predictions are
      visible across workers without duplication of inference work.

    Consequence: a ``/models/reload`` call hits only the worker that receives
    the request. To reload across all workers restart the service or call the
    endpoint 4 times (not recommended; use MLflow auto-polling instead).

    **Thread-safety within a single process:**

    ``_lock`` (RLock) protects the final atomic swap of ``_models[key]`` and
    ``_model_metadata[key]``. The expensive parts of model loading (file I/O,
    deserialisation) happen outside the lock so the request path is not blocked
    during a reload. ``get_model()`` and ``get_model_metadata()`` also acquire
    the lock so readers always see a consistent model+metadata pair.
    """

    _instance = None
    _models: Dict[str, Any] = {}
    _model_metadata: Dict[str, Dict[str, Any]] = {}
    _lock: threading.RLock = threading.RLock()

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
        # 12-factor: an explicit MLFLOW_TRACKING_URI env var overrides the
        # baked-in config default (which points at the docker-compose "mlflow"
        # hostname). This lets a local/CI run point at a file store and avoids
        # long DNS-retry stalls when the compose network isn't present.
        self._mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI") or mlflow_cfg.get(
            "tracking_uri", "http://mlflow:5000"
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
                if model_type == "mlp":
                    self.load_mlp_model(
                        local_path,
                        model_cfg.get("name", model_key),
                        model_cfg.get("version", "local"),
                    )
                elif model_type == "lstm":
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
                        model_key=self._infer_type(model_key),
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

        # Heavy I/O happens outside the lock so readers are not blocked.
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

        # Normalize to short lookup key so get_model("mlp") / get_model("lstm") /
        # get_model("random_forest") always work regardless of config key name.
        storage_key = self._infer_type(model_key)

        new_meta = {
            "name": model_cfg.get("name", model_key),
            "version": version_str,
            "type": storage_key,
            "loaded": True,
            "loaded_at": datetime.utcnow(),
            "source": "mlflow",
            "registry_name": registry_name,
            "registry_stage": stage,
            "mlflow_uri": model_uri,
            "device": _device_label_for(storage_key),
        }

        # Known gap: mlflow.pytorch.log_model() (used by TrainingPipeline.train_mlp
        # and RetrainingPipeline._train_model to register "predictive_maintenance_model")
        # registers only the raw nn.Module — the StandardScaler attached to the
        # MLPRULPredictor is not embedded in that artifact, so it cannot be
        # recovered from mlflow.pyfunc.load_model() here. The scaler IS embedded
        # in the sibling "checkpoint" .pt artifact (see RetrainingPipeline._train_model),
        # which load_mlp_model()'s local-checkpoint path attaches as model._scaler.
        scaler = getattr(native, "_scaler", None) or getattr(native, "scaler", None)
        if storage_key == "mlp":
            model._scaler = scaler  # type: ignore[attr-defined]
            # Explicit GPU placement — InferenceEngine.predict_rul_torch reads
            # model._device to decide where to run the forward pass, so an
            # MLflow-loaded MLP gets the same CUDA/CPU placement as a
            # locally-loaded checkpoint (load_mlp_model).
            model._device = TORCH_DEVICE  # type: ignore[attr-defined]
            if scaler is None:
                logger.warning(
                    "MLflow-loaded model '%s' (%s/%s, v%s) carries no embedded "
                    "scaler — mlflow.pyfunc.load_model() cannot recover the "
                    "StandardScaler from a mlflow.pytorch.log_model() artifact. "
                    "InferenceEngine will run on raw, unscaled features unless "
                    "the local-checkpoint fallback (load_mlp_model, which reads "
                    "the embedded 'scaler' key) is used instead.",
                    model_key, registry_name, stage, version_str,
                )

        # Atomic swap — model and metadata updated together under the lock so
        # concurrent readers never see a new model with stale metadata.
        with self._lock:
            self._models[storage_key] = model
            self._model_metadata[storage_key] = new_meta

        logger.info("Loaded %s → stored as '%s' from MLflow (v%s)", model_key, storage_key, version_str)

    # ------------------------------------------------------------------
    # Local file loaders (kept for fallback & backwards compat)
    # ------------------------------------------------------------------

    def load_mlp_model(self, model_path: str, model_name: str, version: str):
        """
        Load a PyTorch MLP checkpoint (.pt file) onto the available device
        (CUDA GPU if present, otherwise CPU).
        """
        try:
            logger.info("Loading PyTorch MLP from %s (device=%s)", model_path, TORCH_DEVICE)
            if TORCH_DEVICE is not None and TORCH_DEVICE.type == "cuda":
                logger.info(
                    "CUDA available — MLP '%s' v%s will run on GPU (%s)",
                    model_name, version, torch.cuda.get_device_name(TORCH_DEVICE),
                )
            else:
                logger.info("CUDA unavailable, running on CPU fallback.")
            if torch is None:
                raise ImportError("torch not installed")

            checkpoint = torch.load(model_path, map_location=TORCH_DEVICE)

            # Rebuild network from stored config
            hidden_sizes = checkpoint.get("hidden_sizes", [256, 128, 64, 32])
            dropout = checkpoint.get("dropout", 0.2)
            input_dim = checkpoint["input_dim"]

            # Inline minimal MLPRULNet to avoid circular import.
            # Must mirror ml_pipeline.train.models.mlp_model.MLPRULNet/MLPBlock
            # exactly (Linear -> BatchNorm1d -> ReLU -> Dropout, each wrapped
            # in a "block" submodule, all wrapped in a top-level "network"
            # Sequential) — otherwise state_dict keys ("network.N.block.M.*")
            # won't match what MLPRULPredictor.save_model() persisted and
            # load_state_dict() raises before the scaler can be attached.
            class _InlineMLPBlock(nn.Module):
                def __init__(self, in_features, out_features, p_dropout):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Linear(in_features, out_features),
                        nn.BatchNorm1d(out_features),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=p_dropout),
                    )

                def forward(self, x):
                    return self.block(x)

            class _InlineMLPRULNet(nn.Module):
                def __init__(self, in_dim, sizes, p_dropout):
                    super().__init__()
                    blocks = []
                    prev = in_dim
                    for h in sizes:
                        blocks.append(_InlineMLPBlock(prev, h, p_dropout))
                        prev = h
                    blocks.append(nn.Linear(prev, 1))
                    self.network = nn.Sequential(*blocks)

                def forward(self, x):
                    return self.network(x).squeeze(-1)

            model = _InlineMLPRULNet(input_dim, hidden_sizes, dropout).to(TORCH_DEVICE)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Attach metadata
            model._input_dim = input_dim        # type: ignore[attr-defined]
            model._hidden_sizes = hidden_sizes  # type: ignore[attr-defined]
            model._device = TORCH_DEVICE        # type: ignore[attr-defined]
            model._framework = "pytorch"        # type: ignore[attr-defined]
            # Attach scaler if saved in checkpoint (used by InferenceEngine.predict_rul_torch).
            # Older checkpoints saved before scaler persistence was added will
            # have no "scaler" key — load must not crash, but the gap should
            # be visible in logs so wrong-scale inference can be diagnosed.
            scaler = checkpoint.get("scaler")
            model._scaler = scaler  # type: ignore[attr-defined]
            if scaler is None:
                logger.warning(
                    "Checkpoint %s has no embedded scaler (old format or "
                    "scaler persistence was skipped) — InferenceEngine will "
                    "run on raw, unscaled features for model '%s' v%s.",
                    model_path, model_name, version,
                )

            new_meta = {
                "name": model_name,
                "version": version,
                "type": "mlp",
                "framework": "PyTorch",
                "device": str(TORCH_DEVICE),
                "input_dim": input_dim,
                "loaded": True,
                "loaded_at": datetime.utcnow(),
                "path": model_path,
                "source": "local",
            }
            with self._lock:
                self._models["mlp"] = model
                self._model_metadata["mlp"] = new_meta
            logger.info(
                "PyTorch MLP loaded: %s %s | device=%s", model_name, version, TORCH_DEVICE
            )
            return model
        except Exception as exc:
            logger.error("Failed to load PyTorch MLP model: %s", exc)
            self._model_metadata["mlp"] = {
                "name": model_name,
                "version": version,
                "type": "mlp",
                "loaded": False,
                "error": str(exc),
            }
            raise

    def load_canonical_mlp(
        self,
        model_path: str,
        model_name: str,
        version: str,
        *,
        run_id: Optional[str] = None,
        expected_sha256: Optional[str] = None,
    ):
        """
        Load a CANONICAL checkpoint (.pt) with full contract validation.

        Unlike ``load_mlp_model`` (kept for legacy artifacts), this path refuses
        to serve unless the artifact carries a fitted scaler, a matching feature
        schema hash, matching feature names, and the correct input dimension.
        This is the production loader: a registered model version's canonical
        checkpoint artifact is downloaded and resolved through here.

        Model and metadata are swapped together under the lock.
        """
        from shared.checkpoint import load_canonical_checkpoint

        try:
            ckpt = load_canonical_checkpoint(
                model_path,
                strict=True,
                device=str(TORCH_DEVICE) if TORCH_DEVICE is not None else None,
                expected_sha256=expected_sha256,
            )
        except Exception as exc:
            logger.error("Refusing unsafe canonical checkpoint %s: %s", model_path, exc)
            with self._lock:
                self._model_metadata["mlp"] = {
                    "name": model_name,
                    "version": version,
                    "type": "mlp",
                    "loaded": False,
                    "error": str(exc),
                    "unsafe_checkpoint_rejected": True,
                }
            raise

        model = ckpt.model
        model._input_dim = ckpt.input_dim          # type: ignore[attr-defined]
        model._hidden_sizes = ckpt.hidden_sizes    # type: ignore[attr-defined]
        model._device = TORCH_DEVICE               # type: ignore[attr-defined]
        model._framework = "pytorch"               # type: ignore[attr-defined]
        model._scaler = ckpt.scaler                # type: ignore[attr-defined]
        model._feature_schema_hash = ckpt.feature_schema_hash  # type: ignore[attr-defined]

        new_meta = {
            "name": model_name,
            "version": version,
            "type": "mlp",
            "framework": "PyTorch",
            "device": str(TORCH_DEVICE),
            "input_dim": ckpt.input_dim,
            "loaded": True,
            "loaded_at": datetime.utcnow(),
            "path": model_path,
            "source": "canonical_checkpoint",
            "feature_schema_version": ckpt.feature_schema_version,
            "feature_schema_hash": ckpt.feature_schema_hash,
            "checkpoint_sha256": ckpt.checkpoint_sha256,
            "checkpoint_format_version": ckpt.checkpoint_format_version,
            "training_run_id": run_id or ckpt.training_run_id,
            "training_seed": ckpt.training_seed,
            "scaler_fitted": True,
            "evaluation_protocol": ckpt.evaluation_protocol,
            "code_commit": ckpt.code_commit,
        }
        with self._lock:
            self._models["mlp"] = model
            self._model_metadata["mlp"] = new_meta
        logger.info(
            "Canonical MLP loaded: %s v%s | schema=%s | sha=%s | device=%s",
            model_name, version, ckpt.feature_schema_hash[:12],
            ckpt.checkpoint_sha256[:12], TORCH_DEVICE,
        )
        return model

    def load_lstm_model(self, model_path: str, model_name: str, version: str):
        try:
            logger.info("Loading LSTM model from %s", model_path)
            if tf is None:
                raise ImportError("tensorflow not installed")
            model = tf.keras.models.load_model(model_path)
            new_meta = {
                "name": model_name,
                "version": version,
                "type": "lstm",
                "loaded": True,
                "loaded_at": datetime.utcnow(),
                "path": model_path,
                "source": "local",
                "device": _device_label_for("lstm"),
            }
            with self._lock:
                self._models["lstm"] = model
                self._model_metadata["lstm"] = new_meta
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
            new_meta = {
                "name": model_name,
                "version": version,
                "type": "random_forest",
                "loaded": True,
                "loaded_at": datetime.utcnow(),
                "path": model_path,
                "source": "local",
                "device": _device_label_for("random_forest"),
            }
            with self._lock:
                self._models[model_key] = model
                self._model_metadata[model_key] = new_meta
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
        with self._lock:
            return self._models.get(model_key)

    def get_model_metadata(self, model_key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
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
        key = model_key.lower()
        if "mlp" in key:
            return "mlp"
        if "lstm" in key:
            return "lstm"
        return "random_forest"


# Global singleton
model_manager = ModelManager()
