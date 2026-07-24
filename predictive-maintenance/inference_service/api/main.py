"""
FastAPI application and endpoints
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

try:
    from shared.logging_config import configure_logging

    configure_logging(service_name="inference-api")
except ImportError:
    pass  # fallback to default logging when shared module unavailable

from fastapi import FastAPI, HTTPException, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import yaml

from .schemas import (
    RULPredictionRequest,
    RULPredictionResponse,
    HealthPredictionRequest,
    HealthPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    DependencyStatus,
    ModelInfo,
    ErrorResponse,
)
from .error_handler import register_error_handlers, APIError
from .auth import verify_api_key
from .metrics import (
    router as metrics_router,
    INFERENCE_REQUESTS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    PREDICTION_RUL_HOURS,
    MODELS_LOADED,
    KAFKA_PIPELINE_RUNNING,
    SERVICE_UPTIME_SECONDS,
    CACHE_HITS_TOTAL,
    CACHE_MISSES_TOTAL,
    CURRENT_ACCURACY,
    RETRAINING_TRIGGERED_TOTAL,
    FEEDBACK_RECORDS_TOTAL,
    MODEL_VERSION_INFO,
)
from ..models.model_manager import model_manager
from ..models.inference_engine import InferenceEngine
from ..consumer import KafkaPredictionPipeline
from ..cache.prediction_cache import PredictionCache
from ..feature_engineering import online_feature_engineer, ENGINEERED_FEATURE_DIM


logger = logging.getLogger(__name__)

# Service startup time
SERVICE_START_TIME = time.time()
CONFIG = {}
kafka_pipeline: KafkaPredictionPipeline = None  # type: ignore
prediction_cache: PredictionCache = None  # type: ignore
# Shared PerformanceMonitor — initialized once in lifespan, persists for the
# process lifetime.  Single-worker deployment means all /feedback calls mutate
# the same instance and check_accuracy_threshold() always sees all records.
_performance_monitor = None

# Rate limiter (keyed by client IP)
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("Starting inference service...")

    # Load configuration
    global CONFIG
    try:
        with open("inference_service/config/inference_config.yaml", "r") as f:
            CONFIG = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        CONFIG = {
            "service": {"name": "inference-service", "version": "1.0.0"},
            "models": {
                "lstm_rul": {"path": "models/lstm", "warm_start": False},
                "random_forest_health": {"path": "models/rf.pkl", "warm_start": False},
            },
            "api": {"cors_origins": ["*"]},
        }

    # Initialize model manager
    try:
        model_manager.initialize(CONFIG)
        logger.info("Models loaded successfully")

        # GPU-acceleration evidence trail (resume claim: "GPU acceleration").
        # State, in plain words, exactly where the MLP is executing.
        _mlp_meta = model_manager.get_model_metadata("mlp")
        _mlp_device = (_mlp_meta or {}).get("device")
        if _mlp_device and _mlp_device.startswith("cuda"):
            logger.info(
                "GPU acceleration ACTIVE — PyTorch MLP running on CUDA device '%s'",
                _mlp_device,
            )
        else:
            logger.info("CUDA unavailable, running on CPU fallback.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

    # Initialize Redis prediction cache
    global prediction_cache
    try:
        prediction_cache = PredictionCache.from_config(CONFIG)
        if prediction_cache.is_available:
            logger.info("Redis prediction cache connected")
        else:
            logger.warning("Redis unavailable — predictions will not be cached")
    except Exception as e:
        logger.warning(f"Prediction cache init failed: {e}")

    # Initialize PerformanceMonitor singleton — must persist across requests
    global _performance_monitor
    try:
        import sys as _sys
        from pathlib import Path as _Path
        _ml_dir = _Path(__file__).resolve().parent.parent.parent / "ml_pipeline"
        _sys.path.insert(0, str(_ml_dir))
        from retrain.performance_monitor import PerformanceMonitor as _PM
        _pm_cfg = str(_ml_dir / "retrain" / "config" / "retrain_config.yaml")
        _performance_monitor = _PM(config_path=_pm_cfg)
        logger.info("PerformanceMonitor singleton initialized")
    except Exception as exc:
        logger.warning(
            "PerformanceMonitor unavailable — POST /feedback will not persist records: %s", exc
        )

    # Start Kafka prediction pipeline (background thread)
    global kafka_pipeline
    try:
        kafka_pipeline = KafkaPredictionPipeline(
            config=CONFIG,
            model_manager=model_manager,
            inference_engine=inference_engine,
        )
        kafka_pipeline.start()
        logger.info("Kafka prediction pipeline started")
    except Exception as e:
        logger.warning(f"Kafka pipeline not started: {e}")

    # ------------------------------------------------------------------ #
    # Wire up "production reliability" Prometheus gauges that previously
    # existed (imported) but were never updated — they always reported 0.
    # Registered as live callbacks so /metrics always reflects current
    # state regardless of which worker handles the scrape.
    # ------------------------------------------------------------------ #
    SERVICE_UPTIME_SECONDS.set_function(lambda: time.time() - SERVICE_START_TIME)
    MODELS_LOADED.set_function(
        lambda: float(sum(1 for loaded in model_manager.get_model_info().values() if loaded))
    )
    KAFKA_PIPELINE_RUNNING.set_function(
        lambda: 1.0 if (kafka_pipeline and getattr(kafka_pipeline, "_running", False)) else 0.0
    )

    for _model_key, _meta in model_manager.list_models().items():
        _version = _meta.get("version", "unknown")
        MODEL_VERSION_INFO.labels(model=_model_key, version=_version).set(1)

    yield

    # Shutdown
    logger.info("Shutting down inference service...")
    if kafka_pipeline:
        kafka_pipeline.stop()
    model_manager.stop_polling()


# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance Inference API",
    description="Real-time inference API for RUL prediction and health classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    dependencies=[Depends(verify_api_key)],
)

# Register standardised error handlers & request-ID middleware
register_error_handlers(app)

# Register rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Prometheus metrics router (no auth required)
app.include_router(metrics_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("api", {}).get(
        "cors_origins", ["http://localhost:3000", "http://localhost:8080"]
    ),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response: StarletteResponse = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Cache-Control"] = "no-store"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Initialize inference engine
inference_engine = InferenceEngine(
    sequence_length=CONFIG.get("preprocessing", {}).get("sequence_length", 50),
    n_features=ENGINEERED_FEATURE_DIM,
    normalization=CONFIG.get("preprocessing", {}).get("normalization", "standard"),
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Predictive Maintenance Inference API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint

    Returns service status, model availability, and dependency health.
    """
    import os

    uptime = time.time() - SERVICE_START_TIME
    deps: Dict[str, DependencyStatus] = {}
    overall = "healthy"

    # --- TimescaleDB check ---
    try:
        import psycopg2

        _t0 = time.time()
        conn = psycopg2.connect(
            host=os.environ.get(
                "DB_HOST", CONFIG.get("timescaledb", {}).get("host", "timescaledb")
            ),
            port=int(
                os.environ.get(
                    "DB_PORT", CONFIG.get("timescaledb", {}).get("port", 5432)
                )
            ),
            dbname=os.environ.get(
                "DB_NAME",
                CONFIG.get("timescaledb", {}).get("database", "predictive_maintenance"),
            ),
            user=os.environ.get(
                "DB_USER", CONFIG.get("timescaledb", {}).get("user", "pmuser")
            ),
            password=os.environ.get(
                "DB_PASSWORD",
                CONFIG.get("timescaledb", {}).get("password", "pmpassword"),
            ),
            connect_timeout=3,
        )
        conn.close()
        deps["timescaledb"] = DependencyStatus(
            name="timescaledb",
            status="healthy",
            latency_ms=round((time.time() - _t0) * 1000, 2),
        )
    except Exception as exc:
        deps["timescaledb"] = DependencyStatus(
            name="timescaledb",
            status="unhealthy",
            details=str(exc)[:200],
        )
        overall = "degraded"

    # --- Kafka check ---
    try:
        from kafka import KafkaConsumer as _KC

        _t0 = time.time()
        bootstrap = os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS",
            CONFIG.get("kafka", {}).get("bootstrap_servers", "kafka:29092"),
        )
        _kc = _KC(bootstrap_servers=bootstrap, request_timeout_ms=3000)
        _kc.topics()
        _kc.close()
        deps["kafka"] = DependencyStatus(
            name="kafka",
            status="healthy",
            latency_ms=round((time.time() - _t0) * 1000, 2),
        )
    except Exception as exc:
        deps["kafka"] = DependencyStatus(
            name="kafka",
            status="unhealthy",
            details=str(exc)[:200],
        )
        overall = "degraded"

    # --- Redis check ---
    try:
        import redis as _redis

        _t0 = time.time()
        r = _redis.Redis(
            host=os.environ.get(
                "REDIS_HOST", CONFIG.get("redis", {}).get("host", "redis")
            ),
            port=int(
                os.environ.get("REDIS_PORT", CONFIG.get("redis", {}).get("port", 6379))
            ),
            socket_timeout=3,
        )
        r.ping()
        r.close()
        deps["redis"] = DependencyStatus(
            name="redis",
            status="healthy",
            latency_ms=round((time.time() - _t0) * 1000, 2),
        )
    except Exception as exc:
        deps["redis"] = DependencyStatus(
            name="redis",
            status="unhealthy",
            details=str(exc)[:200],
        )
        overall = "degraded"

    # --- Model status ---
    model_info = model_manager.get_model_info()
    any_model_loaded = any(model_info.values()) if model_info else False
    if not any_model_loaded:
        overall = "degraded"

    # --- GPU acceleration evidence ---
    # Resume claim: "deploying a FastAPI service with GPU acceleration ...".
    # Report torch.cuda.is_available() and the actual per-model runtime
    # device so this claim has a live, queryable evidence trail rather than
    # only static code support.
    try:
        import torch as _torch
        cuda_available = bool(_torch.cuda.is_available())
    except ImportError:
        cuda_available = False

    model_devices: Dict[str, str] = {
        key: meta.get("device", "unknown")
        for key, meta in model_manager.list_models().items()
        if meta.get("device") is not None
    }

    # --- Kafka pipeline ---
    pipeline_status = (
        "running"
        if kafka_pipeline and getattr(kafka_pipeline, "_running", False)
        else "stopped"
    )
    deps["kafka_pipeline"] = DependencyStatus(
        name="kafka_pipeline",
        status="healthy" if pipeline_status == "running" else "unknown",
        details=pipeline_status,
    )

    return HealthCheckResponse(
        status=overall,
        version=CONFIG.get("service", {}).get("version", "1.0.0"),
        models_loaded=model_info,
        uptime=uptime,
        timestamp=datetime.utcnow(),
        dependencies=deps,
        model_devices=model_devices,
        cuda_available=cuda_available,
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all available models and their metadata
    """
    models_metadata = model_manager.list_models()

    return [
        ModelInfo(
            name=metadata["name"],
            version=metadata["version"],
            type=metadata["type"],
            loaded=metadata.get("loaded", False),
            last_updated=metadata.get("loaded_at"),
            performance_metrics=metadata.get("performance_metrics"),
            device=metadata.get("device"),
        )
        for metadata in models_metadata.values()
    ]


@app.post("/models/reload")
@limiter.limit("5/hour")
async def reload_models(request: Request, model_key: str = None):
    """
    Reload models from MLflow registry or local paths.

    If ``model_key`` is provided, only that model is reloaded.
    Otherwise all configured models are reloaded.
    """
    try:
        if model_key:
            model_manager.reload_model(model_key)
            return {"status": "ok", "reloaded": [model_key]}
        else:
            reloaded = []
            for key in list(model_manager.list_models().keys()):
                try:
                    model_manager.reload_model(key)
                    reloaded.append(key)
                except Exception as exc:
                    logger.warning("Failed to reload %s: %s", key, exc)
            return {"status": "ok", "reloaded": reloaded}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Model reload error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")


@app.post("/predict/rul", response_model=RULPredictionResponse)
@limiter.limit("100/minute")
async def predict_rul(payload: RULPredictionRequest, request: Request):
    """
    Predict Remaining Useful Life (RUL)

    Uses LSTM model to predict equipment RUL from sensor sequence.
    """
    try:
        # ------------------------------------------------------------------ #
        # Redis cache lookup — serve cached result immediately if available
        # This is the primary latency optimisation for the <300 ms SLA.
        # ------------------------------------------------------------------ #
        cache_key: str = ""
        if prediction_cache and prediction_cache.is_available:
            # Include model name+version so the key changes after a hot-reload.
            _meta_for_cache = model_manager.get_model_metadata("mlp") or \
                              model_manager.get_model_metadata("lstm") or {}
            cache_key = prediction_cache.make_key(
                payload.data.equipment_id,
                payload.data.sequence,
                model_name=_meta_for_cache.get("name", ""),
                model_version=_meta_for_cache.get("version", ""),
            )
            cached = prediction_cache.get(cache_key)
            if cached is not None:
                logger.debug("Cache HIT for %s", payload.data.equipment_id)
                INFERENCE_REQUESTS_TOTAL.labels(model="mlp", status="cache_hit").inc()
                CACHE_HITS_TOTAL.inc()
                return RULPredictionResponse(**cached)
            CACHE_MISSES_TOTAL.inc()

        # ------------------------------------------------------------------ #
        # Cache miss — run full inference
        # ------------------------------------------------------------------ #

        # Try PyTorch MLP first (primary model), fall back to LSTM
        model = model_manager.get_model("mlp")
        model_label = "mlp"
        if model is None:
            model = model_manager.get_model("lstm")
            model_label = "lstm"
        if model is None:
            raise APIError(
                status_code=503,
                error="model_unavailable",
                message="No RUL model loaded (mlp or lstm)",
            )

        # Preprocess — temporal features for MLP, sequence for LSTM
        try:
            import torch as _torch
            if isinstance(model, _torch.nn.Module):
                # PyTorch MLP: compute lag/rolling/EMA features over the FULL
                # sequence — mirrors _engineer_features() from training exactly.
                # Using only the last reading (old behaviour) discarded all
                # temporal signal and caused a training-serving feature mismatch.
                sequence = online_feature_engineer.engineer(
                    payload.data.sequence, payload.data.equipment_id
                )
            else:
                sequence = inference_engine.preprocess_sequence(
                    payload.data.sequence, payload.data.equipment_id
                )
        except ImportError:
            sequence = inference_engine.preprocess_sequence(
                payload.data.sequence, payload.data.equipment_id
            )

        # Predict (instrumented) — GPU-accelerated for PyTorch path
        _rul_start = time.time()
        rul, confidence_interval = inference_engine.predict_rul(
            model, sequence, return_confidence=payload.return_confidence
        )
        INFERENCE_LATENCY_SECONDS.labels(model=model_label).observe(time.time() - _rul_start)
        INFERENCE_REQUESTS_TOTAL.labels(model=model_label, status="success").inc()

        # Determine health status
        health_status = inference_engine.get_health_status_from_rul(rul)

        # Update RUL gauge
        PREDICTION_RUL_HOURS.labels(equipment_id=payload.data.equipment_id).set(
            rul * 0.5
        )

        # Get model version
        model_metadata = model_manager.get_model_metadata(model_label)
        model_version = (model_metadata or {}).get("version", "unknown")

        response = RULPredictionResponse(
            equipment_id=payload.data.equipment_id,
            rul_cycles=rul,
            rul_hours=rul * 0.5,  # approximate: 1 cycle ≈ 0.5 hours
            anomaly_score=None,
            health_status=health_status,
            confidence=confidence_interval.get("confidence")
            if confidence_interval
            else None,
            confidence_interval=confidence_interval,
            timestamp=datetime.utcnow(),
            model_version=model_version,
            recommendations=inference_engine.get_recommendations(rul, health_status),
        )

        # ------------------------------------------------------------------ #
        # Store result in Redis so subsequent identical requests are instant
        # ------------------------------------------------------------------ #
        if prediction_cache and prediction_cache.is_available and cache_key:
            prediction_cache.set(cache_key, response.dict())

        return response

    except APIError:
        INFERENCE_REQUESTS_TOTAL.labels(model="mlp", status="error").inc()
        raise
    except Exception as e:
        INFERENCE_REQUESTS_TOTAL.labels(model="mlp", status="error").inc()
        logger.error(f"RUL prediction error: {e}")
        raise APIError(
            status_code=500,
            error="prediction_error",
            message=f"RUL prediction failed: {str(e)}",
        )


@app.post("/predict/health", response_model=HealthPredictionResponse)
@limiter.limit("100/minute")
async def predict_health(payload: HealthPredictionRequest, request: Request):
    """
    Predict Health Status

    Uses Random Forest to classify equipment health from current sensor readings.
    """
    try:
        # Get model
        model = model_manager.get_model("random_forest")
        if model is None:
            raise APIError(
                status_code=503,
                error="model_unavailable",
                message="Random Forest model not loaded",
            )

        # Preprocess
        features = inference_engine.preprocess_features(
            payload.data.features, payload.data.equipment_id
        )

        # Predict (instrumented)
        _health_start = time.time()
        predicted_class, confidence, probabilities = inference_engine.predict_health(
            model, features, return_probabilities=payload.return_probabilities
        )
        INFERENCE_LATENCY_SECONDS.labels(model="random_forest").observe(
            time.time() - _health_start
        )
        INFERENCE_REQUESTS_TOTAL.labels(model="random_forest", status="success").inc()

        # Get model version
        model_metadata = model_manager.get_model_metadata("random_forest")
        model_version = model_metadata.get("version", "unknown")

        # Health status code mapping
        status_codes = {
            "healthy": 0,
            "warning": 1,
            "critical": 2,
            "imminent_failure": 3,
        }

        return HealthPredictionResponse(
            equipment_id=payload.data.equipment_id,
            health_status=predicted_class,
            health_status_code=status_codes.get(predicted_class),
            probabilities=probabilities,
            anomaly_score=None,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            model_version=model_version,
        )

    except APIError:
        INFERENCE_REQUESTS_TOTAL.labels(model="random_forest", status="error").inc()
        raise
    except Exception as e:
        INFERENCE_REQUESTS_TOTAL.labels(model="random_forest", status="error").inc()
        logger.error(f"Health prediction error: {e}")
        raise APIError(
            status_code=500,
            error="prediction_error",
            message=f"Health prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
@limiter.limit("100/minute")
async def predict_batch(payload: BatchPredictionRequest, request: Request):
    """
    Batch RUL Predictions

    Process multiple equipment sequences in a single request.
    """
    try:
        # Prefer MLP; fall back to LSTM
        model = model_manager.get_model("mlp")
        model_label = "mlp"
        if model is None:
            model = model_manager.get_model("lstm")
            model_label = "lstm"
        if model is None:
            raise APIError(
                status_code=503,
                error="model_unavailable",
                message="No RUL model loaded (mlp or lstm)",
            )

        start_time = time.time()
        predictions = []

        # Get model version
        model_metadata = model_manager.get_model_metadata(model_label)
        model_version = (model_metadata or {}).get("version", "unknown")

        # Process each sequence
        for seq_data in payload.sequences:
            try:
                # Preprocess — temporal features for MLP, sequence for LSTM
                try:
                    import torch as _torch
                    if isinstance(model, _torch.nn.Module):
                        # Full temporal feature engineering — same as /predict/rul
                        sequence = online_feature_engineer.engineer(
                            seq_data.sequence, seq_data.equipment_id
                        )
                    else:
                        raise TypeError
                except (ImportError, TypeError):
                    sequence = inference_engine.preprocess_sequence(
                        seq_data.sequence, seq_data.equipment_id
                    )

                # Predict
                rul, _ = inference_engine.predict_rul(
                    model, sequence, return_confidence=False
                )

                # Determine health status
                health_status = inference_engine.get_health_status_from_rul(rul)

                predictions.append(
                    RULPredictionResponse(
                        equipment_id=seq_data.equipment_id,
                        rul_cycles=rul,
                        rul_hours=rul * 0.5,
                        health_status=health_status,
                        timestamp=datetime.utcnow(),
                        model_version=model_version,
                    )
                )

            except Exception as e:
                logger.error(f"Failed to process {seq_data.equipment_id}: {e}")
                continue

        processing_time_ms = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            results=predictions,
            batch_size=len(predictions),
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow(),
        )

    except APIError:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise APIError(
            status_code=500,
            error="prediction_error",
            message=f"Batch prediction failed: {str(e)}",
        )


@app.post("/feedback", status_code=status.HTTP_200_OK)
@limiter.limit("200/minute")
async def record_feedback(
    request: Request,
    equipment_id: str,
    true_rul: float,
    predicted_rul: float,
    timestamp: Optional[datetime] = None,
):
    # Input bounds validation
    import math as _math
    if not equipment_id or not equipment_id.strip():
        raise APIError(status_code=400, error="validation_error",
                       message="equipment_id must not be empty")
    if _math.isnan(true_rul) or _math.isinf(true_rul):
        raise APIError(status_code=400, error="validation_error",
                       message="true_rul must be a finite number")
    if _math.isnan(predicted_rul) or _math.isinf(predicted_rul):
        raise APIError(status_code=400, error="validation_error",
                       message="predicted_rul must be a finite number")
    if true_rul < 0:
        raise APIError(status_code=400, error="validation_error",
                       message=f"true_rul must be >= 0, got {true_rul}")
    if predicted_rul < 0:
        raise APIError(status_code=400, error="validation_error",
                       message=f"predicted_rul must be >= 0, got {predicted_rul}")
    _MAX_RUL = 1000.0
    if true_rul > _MAX_RUL:
        raise APIError(status_code=400, error="validation_error",
                       message=f"true_rul exceeds plausible upper bound ({_MAX_RUL})")
    if predicted_rul > _MAX_RUL:
        raise APIError(status_code=400, error="validation_error",
                       message=f"predicted_rul exceeds plausible upper bound ({_MAX_RUL})")
    """
    Record ground-truth RUL for a previous prediction.

    Records are stored in the process-level PerformanceMonitor singleton so
    they accumulate across requests and are visible to check_accuracy_threshold()
    when the retraining pipeline runs.  The previous implementation created a
    new PerformanceMonitor per request, discarding every record on return.

    Args:
        equipment_id:  Identifier of the engine / equipment.
        true_rul:      Observed ground-truth RUL (cycles) at end-of-life or
                       maintenance inspection.
        predicted_rul: The RUL the model predicted at that time.
        timestamp:     Optional ISO-8601 timestamp for the observation
                       (defaults to now if omitted).
    """
    if _performance_monitor is None:
        logger.warning("PerformanceMonitor not initialized — feedback not stored")
        return {"status": "ignored", "reason": "performance_monitor_not_initialized"}
    try:
        _performance_monitor.record_prediction(
            equipment_id=equipment_id,
            true_rul=true_rul,
            predicted_rul=predicted_rul,
            timestamp=timestamp,
        )
        FEEDBACK_RECORDS_TOTAL.inc()

        # Refresh the rolling-accuracy gauge — the same number compared
        # against the 80% threshold that triggers retraining (resume claim:
        # "triggered retraining when accuracy dropped below 80%").
        try:
            _report = _performance_monitor.check_accuracy_threshold()
            CURRENT_ACCURACY.labels(model="mlp").set(_report.accuracy)
        except Exception:
            logger.debug("Could not refresh model_rolling_accuracy gauge", exc_info=True)

        logger.info(
            "Feedback recorded — equipment=%s true_rul=%.1f predicted_rul=%.1f "
            "(total records: %d)",
            equipment_id,
            true_rul,
            predicted_rul,
            _performance_monitor.n_records,
        )
        return {
            "status": "recorded",
            "equipment_id": equipment_id,
            "true_rul": true_rul,
            "predicted_rul": predicted_rul,
            "error_cycles": abs(true_rul - predicted_rul),
        }
    except Exception as exc:
        logger.error("Feedback recording failed: %s", exc)
        raise APIError(
            status_code=500,
            error="feedback_error",
            message=f"Failed to record feedback: {exc}",
        )


@app.post("/train")
@limiter.limit("2/hour")
async def trigger_training(request: Request, model_type: str = "all"):
    """
    Trigger model retraining.

    This is an admin-only endpoint (requires ADMIN_API_KEY).
    Runs a retraining check asynchronously.

    Args:
        model_type: Which model to retrain — "lstm", "rf", or "all".
    """
    import threading

    valid_types = {"lstm", "rf", "all"}
    if model_type not in valid_types:
        raise APIError(
            status_code=400,
            error="invalid_model_type",
            message=f"model_type must be one of {valid_types}",
        )

    def _run_retrain():
        try:
            logger.info("Manual retraining triggered for model_type=%s", model_type)
            RETRAINING_TRIGGERED_TOTAL.labels(reason="manual_api").inc()
            try:
                import sys
                from pathlib import Path

                sys.path.insert(
                    0,
                    str(Path(__file__).resolve().parent.parent.parent / "ml_pipeline"),
                )
                from retrain.retrain_pipeline import RetrainingPipeline

                # Pass the process-level monitor so the pipeline can read the
                # feedback records that accumulated via POST /feedback.
                pipeline = RetrainingPipeline(
                    performance_monitor=_performance_monitor
                )
                pipeline.trigger_retraining(
                    reason=f"Manual trigger via API (model_type={model_type})"
                )
                logger.info("Retraining completed successfully")
            except ImportError:
                logger.warning("ml_pipeline not available — retraining skipped")
        except Exception:
            logger.exception("Retraining failed")

    thread = threading.Thread(target=_run_retrain, daemon=True)
    thread.start()

    return {
        "status": "accepted",
        "message": f"Retraining triggered for model_type={model_type}. Running in background.",
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=CONFIG.get("service", {}).get("host", "0.0.0.0"),
        port=CONFIG.get("service", {}).get("port", 8000),
        workers=CONFIG.get("service", {}).get("workers", 4),
    )
