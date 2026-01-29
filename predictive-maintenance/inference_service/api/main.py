"""
FastAPI application and endpoints
"""

import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yaml

from .schemas import (
    RULPredictionRequest,
    RULPredictionResponse,
    HealthPredictionRequest,
    HealthPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
    ModelInfo,
    ErrorResponse,
)
from ..models.model_manager import model_manager
from ..models.inference_engine import InferenceEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Service startup time
SERVICE_START_TIME = time.time()
CONFIG = {}


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

    yield

    # Shutdown
    logger.info("Shutting down inference service...")


# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance Inference API",
    description="Real-time inference API for RUL prediction and health classification",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    Returns service status and model availability
    """
    uptime = time.time() - SERVICE_START_TIME

    return HealthCheckResponse(
        status="healthy",
        version=CONFIG.get("service", {}).get("version", "1.0.0"),
        models_loaded=model_manager.get_model_info(),
        uptime=uptime,
        timestamp=datetime.utcnow(),
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


@app.post("/predict/rul", response_model=RULPredictionResponse)
async def predict_rul(request: RULPredictionRequest):
    """
    Predict Remaining Useful Life (RUL)

    Uses LSTM model to predict equipment RUL from sensor sequence.
    """
    try:
        # Get model
        model = model_manager.get_model("lstm")
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LSTM model not loaded",
            )

        # Preprocess
        sequence = inference_engine.preprocess_sequence(
            request.data.sequence, request.data.equipment_id
        )

        # Predict
        rul, confidence_interval = inference_engine.predict_rul(
            model, sequence, return_confidence=request.return_confidence
        )

        # Determine health status
        health_status = inference_engine.get_health_status_from_rul(rul)

        # Get model version
        model_metadata = model_manager.get_model_metadata("lstm")
        model_version = model_metadata.get("version", "unknown")

        return RULPredictionResponse(
            equipment_id=request.data.equipment_id,
            predicted_rul=rul,
            confidence_interval=confidence_interval,
            health_status=health_status,
            timestamp=datetime.utcnow(),
            model_version=model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RUL prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/health", response_model=HealthPredictionResponse)
async def predict_health(request: HealthPredictionRequest):
    """
    Predict Health Status

    Uses Random Forest to classify equipment health from current sensor readings.
    """
    try:
        # Get model
        model = model_manager.get_model("random_forest")
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Random Forest model not loaded",
            )

        # Preprocess
        features = inference_engine.preprocess_features(
            request.data.features, request.data.equipment_id
        )

        # Predict
        predicted_class, confidence, probabilities = inference_engine.predict_health(
            model, features, return_probabilities=request.return_probabilities
        )

        # Get model version
        model_metadata = model_manager.get_model_metadata("random_forest")
        model_version = model_metadata.get("version", "unknown")

        return HealthPredictionResponse(
            equipment_id=request.data.equipment_id,
            predicted_class=predicted_class,
            probabilities=probabilities,
            confidence=confidence,
            timestamp=datetime.utcnow(),
            model_version=model_version,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch RUL Predictions

    Process multiple equipment sequences in a single request.
    """
    try:
        # Get model
        model = model_manager.get_model("lstm")
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LSTM model not loaded",
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
                        predicted_rul=rul,
                        confidence_interval=None,
                        health_status=health_status,
                        timestamp=datetime.utcnow(),
                        model_version=model_version,
                    )
                )

            except Exception as e:
                logger.error(f"Failed to process {seq_data.equipment_id}: {e}")
                # Continue with other sequences
                continue

        processing_time = time.time() - start_time

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=CONFIG.get("service", {}).get("host", "0.0.0.0"),
        port=CONFIG.get("service", {}).get("port", 8000),
        workers=CONFIG.get("service", {}).get("workers", 4),
    )
