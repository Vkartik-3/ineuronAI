"""
FastAPI application and endpoints
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any
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
)
from ..models.model_manager import model_manager
from ..models.inference_engine import InferenceEngine
from ..consumer import KafkaPredictionPipeline


logger = logging.getLogger(__name__)

# Service startup time
SERVICE_START_TIME = time.time()
CONFIG = {}
kafka_pipeline: KafkaPredictionPipeline = None  # type: ignore

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
    except Exception as e:
        logger.error(f"Failed to load models: {e}")

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
    n_features=150,
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
        )
        for metadata in models_metadata.values()
    ]


@app.post("/models/reload")
@limiter.limit("5/hour")
async def reload_models(req: Request, model_key: str = None):
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
async def predict_rul(request: RULPredictionRequest, req: Request):
    """
    Predict Remaining Useful Life (RUL)

    Uses LSTM model to predict equipment RUL from sensor sequence.
    """
    try:
        # Get model
        model = model_manager.get_model("lstm")
        if model is None:
            raise APIError(
                status_code=503,
                error="model_unavailable",
                message="LSTM model not loaded",
            )

        # Preprocess
        sequence = inference_engine.preprocess_sequence(
            request.data.sequence, request.data.equipment_id
        )

        # Predict (instrumented)
        _rul_start = time.time()
        rul, confidence_interval = inference_engine.predict_rul(
            model, sequence, return_confidence=request.return_confidence
        )
        INFERENCE_LATENCY_SECONDS.labels(model="lstm").observe(time.time() - _rul_start)
        INFERENCE_REQUESTS_TOTAL.labels(model="lstm", status="success").inc()

        # Determine health status
        health_status = inference_engine.get_health_status_from_rul(rul)

        # Update RUL gauge
        PREDICTION_RUL_HOURS.labels(equipment_id=request.data.equipment_id).set(
            rul * 0.5
        )

        # Get model version
        model_metadata = model_manager.get_model_metadata("lstm")
        model_version = model_metadata.get("version", "unknown")

        return RULPredictionResponse(
            equipment_id=request.data.equipment_id,
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

    except APIError:
        INFERENCE_REQUESTS_TOTAL.labels(model="lstm", status="error").inc()
        raise
    except Exception as e:
        INFERENCE_REQUESTS_TOTAL.labels(model="lstm", status="error").inc()
        logger.error(f"RUL prediction error: {e}")
        raise APIError(
            status_code=500,
            error="prediction_error",
            message=f"RUL prediction failed: {str(e)}",
        )


@app.post("/predict/health", response_model=HealthPredictionResponse)
@limiter.limit("100/minute")
async def predict_health(request: HealthPredictionRequest, req: Request):
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
            request.data.features, request.data.equipment_id
        )

        # Predict (instrumented)
        _health_start = time.time()
        predicted_class, confidence, probabilities = inference_engine.predict_health(
            model, features, return_probabilities=request.return_probabilities
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
            equipment_id=request.data.equipment_id,
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
async def predict_batch(request: BatchPredictionRequest, req: Request):
    """
    Batch RUL Predictions

    Process multiple equipment sequences in a single request.
    """
    try:
        # Get model
        model = model_manager.get_model("lstm")
        if model is None:
            raise APIError(
                status_code=503,
                error="model_unavailable",
                message="LSTM model not loaded",
            )

        start_time = time.time()
        predictions = []

        # Get model version
        model_metadata = model_manager.get_model_metadata("lstm")
        model_version = model_metadata.get("version", "unknown")

        # Process each sequence
        for seq_data in request.sequences:
            try:
                # Preprocess
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


@app.post("/train")
@limiter.limit("2/hour")
async def trigger_training(req: Request, model_type: str = "all"):
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
            # Import retraining pipeline if available
            try:
                import sys
                from pathlib import Path

                sys.path.insert(
                    0,
                    str(Path(__file__).resolve().parent.parent.parent / "ml_pipeline"),
                )
                from retrain.retrain_pipeline import RetrainingPipeline

                pipeline = RetrainingPipeline()
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
