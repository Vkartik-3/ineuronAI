"""
Pydantic models for API request/response validation
"""

import math
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class SensorData(BaseModel):
    """Single sensor reading (used by /predict/health for Random Forest classification)."""

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    features: Dict[str, float] = Field(
        ...,
        description=(
            "Sensor feature dict for the Random Forest health classifier. "
            "Keys and dimensionality depend on the feature set the RF was trained on."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "EQ001",
                "timestamp": "2024-01-15T10:30:00Z",
                "features": {
                    "temperature": 75.5,
                    "vibration": 0.42,
                    "pressure": 102.3,
                    "rpm": 1800,
                    "power_consumption": 45.2,
                },
            }
        }


class SequenceData(BaseModel):
    """Sequence of raw C-MAPSS sensor readings for MLP RUL prediction."""

    equipment_id: str = Field(..., description="Equipment identifier")
    sequence: List[Dict[str, float]] = Field(
        ...,
        description=(
            "Sequence of raw sensor reading dicts (oldest first). "
            "Each dict should contain sensor_N and op_setting_N keys matching "
            "the NASA C-MAPSS format.  Recommended length ≥ 20 for stable "
            "temporal features; minimum 1 reading accepted."
        ),
        min_length=1,
        max_length=200,
    )

    @field_validator("sequence")
    @classmethod
    def _reject_nan_inf(cls, seq: List[Dict[str, float]]) -> List[Dict[str, float]]:
        for i, reading in enumerate(seq):
            for k, v in reading.items():
                if not isinstance(v, (int, float)):
                    raise ValueError(
                        f"sequence[{i}]['{k}'] must be numeric, got {type(v).__name__}"
                    )
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(
                        f"sequence[{i}]['{k}'] contains NaN or Inf — "
                        "clean or impute sensor values before sending"
                    )
        return seq

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "EQ001",
                "sequence": [
                    {"sensor_2": 642.5, "sensor_3": 1589.7, "op_setting_1": 0.0,
                     "op_setting_2": 0.0, "op_setting_3": 100.0, "time_cycle": 1.0},
                    {"sensor_2": 643.1, "sensor_3": 1590.2, "op_setting_1": 0.0,
                     "op_setting_2": 0.0, "op_setting_3": 100.0, "time_cycle": 2.0},
                ],
            }
        }


class RULPredictionRequest(BaseModel):
    """Request for RUL prediction"""

    data: SequenceData = Field(..., description="Input sequence data")
    return_confidence: bool = Field(
        False, description="Return prediction confidence interval"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "equipment_id": "EQ001",
                    "sequence": [{"temperature": 75.5, "vibration": 0.42}],
                },
                "return_confidence": True,
            }
        }


class HealthPredictionRequest(BaseModel):
    """Request for health classification"""

    data: SensorData = Field(..., description="Current sensor readings")
    return_probabilities: bool = Field(False, description="Return class probabilities")

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "equipment_id": "EQ001",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "features": {"temperature": 75.5, "vibration": 0.42},
                },
                "return_probabilities": True,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions"""

    sequences: List[SequenceData] = Field(
        ..., description="Multiple sequences", max_length=100
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sequences": [
                    {"equipment_id": "EQ001", "sequence": [{"temperature": 75.5}]},
                    {"equipment_id": "EQ002", "sequence": [{"temperature": 80.2}]},
                ]
            }
        }


class RULPredictionResponse(BaseModel):
    """Response for RUL prediction — matches api-contract.md §1.3"""

    equipment_id: str = Field(..., description="Equipment identifier")
    rul_cycles: float = Field(
        ..., description="Predicted remaining useful life (cycles)"
    )
    rul_hours: Optional[float] = Field(None, description="Predicted RUL in hours")
    anomaly_score: Optional[float] = Field(None, description="Anomaly score [0-1]")
    health_status: str = Field(..., description="Health status derived from RUL")
    confidence: Optional[float] = Field(None, description="Prediction confidence [0-1]")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="95% confidence interval {lower, upper}"
    )
    model_version: str = Field(..., description="Model version used")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    recommendations: Optional[List[str]] = Field(
        None, description="Actionable recommendations"
    )

    # Keep backwards-compatible alias
    @property
    def predicted_rul(self) -> float:
        return self.rul_cycles

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "ENGINE_0001",
                "rul_cycles": 145,
                "rul_hours": 72.5,
                "anomaly_score": 0.35,
                "health_status": "warning",
                "confidence": 0.87,
                "confidence_interval": {"lower": 58.2, "upper": 86.8},
                "model_version": "v1.2.0",
                "timestamp": "2026-03-11T10:00:00Z",
                "recommendations": [
                    "Schedule maintenance within 72 hours",
                    "Monitor temperature_3 trend — elevated",
                ],
            }
        }


class HealthPredictionResponse(BaseModel):
    """Response for health classification — matches api-contract.md §1.4"""

    equipment_id: str = Field(..., description="Equipment identifier")
    health_status: str = Field(..., description="Predicted health status")
    health_status_code: Optional[int] = Field(
        None,
        description="Numeric code (0=healthy,1=warning,2=critical,3=imminent_failure)",
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Class probabilities"
    )
    anomaly_score: Optional[float] = Field(None, description="Anomaly score [0-1]")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "ENGINE_0001",
                "health_status": "warning",
                "health_status_code": 1,
                "probabilities": {
                    "healthy": 0.15,
                    "warning": 0.62,
                    "critical": 0.18,
                    "imminent_failure": 0.05,
                },
                "anomaly_score": 0.45,
                "confidence": 0.62,
                "timestamp": "2026-03-11T10:00:00Z",
                "model_version": "v1.1.0",
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions — matches api-contract.md §1.5"""

    results: List[RULPredictionResponse] = Field(
        ..., description="Per-equipment predictions"
    )
    batch_size: int = Field(..., description="Number of results returned")
    processing_time_ms: float = Field(
        ..., description="Total processing time (milliseconds)"
    )
    timestamp: datetime = Field(..., description="Batch timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "equipment_id": "ENGINE_0001",
                        "rul_cycles": 145,
                        "rul_hours": 72.5,
                        "anomaly_score": 0.35,
                        "health_status": "warning",
                        "confidence": 0.87,
                        "model_version": "v1.2.0",
                        "timestamp": "2026-03-11T10:00:00Z",
                    }
                ],
                "batch_size": 10,
                "processing_time_ms": 145,
                "timestamp": "2026-03-11T10:00:00Z",
            }
        }


class DependencyStatus(BaseModel):
    """Status of a single upstream dependency"""

    name: str = Field(..., description="Dependency name")
    status: str = Field(..., description="healthy | unhealthy | unknown")
    latency_ms: Optional[float] = Field(None, description="Check latency in ms")
    details: Optional[str] = Field(None, description="Additional info")


class HealthCheckResponse(BaseModel):
    """API health check response"""

    status: str = Field(
        ..., description="Service status (healthy | degraded | unhealthy)"
    )
    version: str = Field(..., description="Service version")
    models_loaded: Dict[str, bool] = Field(..., description="Model load status")
    uptime: float = Field(..., description="Service uptime (seconds)")
    timestamp: datetime = Field(..., description="Current timestamp")
    dependencies: Optional[Dict[str, DependencyStatus]] = Field(
        None, description="Upstream dependency health"
    )
    model_devices: Optional[Dict[str, str]] = Field(
        None,
        description=(
            "Runtime execution device per loaded model (e.g. 'cuda:0' or 'cpu'). "
            "Evidence for whether GPU acceleration is actually active — "
            "see torch.cuda.is_available()."
        ),
    )
    cuda_available: Optional[bool] = Field(
        None, description="torch.cuda.is_available() on this host"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": {"lstm_rul": True, "random_forest_health": True},
                "uptime": 3600.5,
                "timestamp": "2024-01-15T10:30:05Z",
                "model_devices": {"mlp": "cuda:0", "lstm": "cpu"},
                "cuda_available": True,
                "dependencies": {
                    "timescaledb": {
                        "name": "timescaledb",
                        "status": "healthy",
                        "latency_ms": 2.3,
                    },
                    "kafka": {"name": "kafka", "status": "healthy", "latency_ms": 5.1},
                    "redis": {"name": "redis", "status": "healthy", "latency_ms": 0.8},
                },
            }
        }


class ModelInfo(BaseModel):
    """Model metadata"""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type (lstm, random_forest)")
    loaded: bool = Field(..., description="Model load status")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    performance_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Latest performance metrics"
    )
    device: Optional[str] = Field(
        None,
        description=(
            "Runtime execution device this model is placed on, e.g. 'cuda:0' "
            "(GPU acceleration active) or 'cpu' (CUDA unavailable, CPU fallback)."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "PyTorch MLP RUL Predictor",
                "version": "v1.0.0",
                "type": "mlp",
                "loaded": True,
                "last_updated": "2024-01-10T14:20:00Z",
                "performance_metrics": {"rmse": 8.5, "mae": 6.2, "r2": 0.92},
                "device": "cuda:0",
            }
        }


class ErrorResponse(BaseModel):
    """Standardised error response — matches api-contract.md §7.3"""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable message")
    details: Optional[Dict[str, Any]] = Field(None, description="Extra context")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Sequence length must be >= 50 time steps",
                "details": {"received_length": 10, "required_length": 50},
                "timestamp": "2026-03-11T10:00:00Z",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }
