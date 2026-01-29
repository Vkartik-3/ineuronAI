"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class SensorData(BaseModel):
    """Single sensor reading"""

    equipment_id: str = Field(..., description="Equipment identifier")
    timestamp: datetime = Field(..., description="Measurement timestamp")
    features: Dict[str, float] = Field(
        ..., description="Sensor features (150 dimensions)"
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
    """Sequence of sensor readings for LSTM"""

    equipment_id: str = Field(..., description="Equipment identifier")
    sequence: List[Dict[str, float]] = Field(
        ...,
        description="Sequence of feature vectors (length=50)",
        min_length=1,
        max_length=100,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "EQ001",
                "sequence": [
                    {"temperature": 75.5, "vibration": 0.42, "pressure": 102.3},
                    {"temperature": 76.0, "vibration": 0.43, "pressure": 102.5},
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
    """Response for RUL prediction"""

    equipment_id: str = Field(..., description="Equipment identifier")
    predicted_rul: float = Field(
        ..., description="Predicted remaining useful life (cycles)"
    )
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="95% confidence interval"
    )
    health_status: str = Field(..., description="Health status based on RUL")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "EQ001",
                "predicted_rul": 45.3,
                "confidence_interval": {"lower": 38.1, "upper": 52.5},
                "health_status": "warning",
                "timestamp": "2024-01-15T10:30:05Z",
                "model_version": "v1.0.0",
            }
        }


class HealthPredictionResponse(BaseModel):
    """Response for health classification"""

    equipment_id: str = Field(..., description="Equipment identifier")
    predicted_class: str = Field(..., description="Predicted health class")
    probabilities: Optional[Dict[str, float]] = Field(
        None, description="Class probabilities"
    )
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")

    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "EQ001",
                "predicted_class": "warning",
                "probabilities": {
                    "healthy": 0.15,
                    "warning": 0.65,
                    "critical": 0.18,
                    "imminent_failure": 0.02,
                },
                "confidence": 0.65,
                "timestamp": "2024-01-15T10:30:05Z",
                "model_version": "v1.0.0",
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions"""

    predictions: List[RULPredictionResponse] = Field(
        ..., description="Batch predictions"
    )
    total_processed: int = Field(..., description="Number of predictions processed")
    processing_time: float = Field(..., description="Total processing time (seconds)")

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "equipment_id": "EQ001",
                        "predicted_rul": 45.3,
                        "health_status": "warning",
                        "timestamp": "2024-01-15T10:30:05Z",
                        "model_version": "v1.0.0",
                    }
                ],
                "total_processed": 10,
                "processing_time": 0.25,
            }
        }


class HealthCheckResponse(BaseModel):
    """API health check response"""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    models_loaded: Dict[str, bool] = Field(..., description="Model load status")
    uptime: float = Field(..., description="Service uptime (seconds)")
    timestamp: datetime = Field(..., description="Current timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": {"lstm_rul": True, "random_forest_health": True},
                "uptime": 3600.5,
                "timestamp": "2024-01-15T10:30:05Z",
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

    class Config:
        json_schema_extra = {
            "example": {
                "name": "LSTM RUL Predictor",
                "version": "v1.0.0",
                "type": "lstm",
                "loaded": True,
                "last_updated": "2024-01-10T14:20:00Z",
                "performance_metrics": {"rmse": 8.5, "mae": 6.2, "r2": 0.92},
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "detail": "Feature 'temperature' is out of range",
                "timestamp": "2024-01-15T10:30:05Z",
            }
        }
