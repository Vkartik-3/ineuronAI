# Inference Service

Real-time REST API for predictive maintenance inference.

## Features

- **RUL Prediction**: Predict remaining useful life using LSTM
- **Health Classification**: Classify equipment health using Random Forest
- **Batch Processing**: Process multiple predictions efficiently
- **Model Management**: Hot-reload models without downtime
- **OpenAPI Docs**: Interactive API documentation at `/docs`

## Installation

```bash
cd inference_service
pip install -r requirements.txt
```

## Configuration

Edit `config/inference_config.yaml`:

```yaml
service:
  host: "0.0.0.0"
  port: 8000
  workers: 4

models:
  lstm_rul:
    path: "models/lstm_rul_model"
    warm_start: true
  random_forest_health:
    path: "models/rf_health_classifier.pkl"
    warm_start: true
```

## Quick Start

### Start Server

```bash
# Development mode
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "lstm": true,
    "random_forest": true
  },
  "uptime": 3600.5,
  "timestamp": "2024-01-15T10:30:05Z"
}
```

## API Endpoints

### 1. Predict RUL

**Endpoint**: `POST /predict/rul`

**Request**:

```json
{
  "data": {
    "equipment_id": "EQ001",
    "sequence": [
      { "temperature": 75.5, "vibration": 0.42, "pressure": 102.3 },
      { "temperature": 76.0, "vibration": 0.43, "pressure": 102.5 }
    ]
  },
  "return_confidence": true
}
```

**Response**:

```json
{
  "equipment_id": "EQ001",
  "predicted_rul": 45.3,
  "confidence_interval": {
    "lower": 38.1,
    "upper": 52.5,
    "std": 3.6
  },
  "health_status": "warning",
  "timestamp": "2024-01-15T10:30:05Z",
  "model_version": "v1.0.0"
}
```

**cURL Example**:

```bash
curl -X POST "http://localhost:8000/predict/rul" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "equipment_id": "EQ001",
      "sequence": [{"temperature": 75.5, "vibration": 0.42}]
    },
    "return_confidence": true
  }'
```

### 2. Predict Health Status

**Endpoint**: `POST /predict/health`

**Request**:

```json
{
  "data": {
    "equipment_id": "EQ001",
    "timestamp": "2024-01-15T10:30:00Z",
    "features": {
      "temperature": 75.5,
      "vibration": 0.42,
      "pressure": 102.3,
      "rpm": 1800
    }
  },
  "return_probabilities": true
}
```

**Response**:

```json
{
  "equipment_id": "EQ001",
  "predicted_class": "warning",
  "probabilities": {
    "healthy": 0.15,
    "warning": 0.65,
    "critical": 0.18,
    "imminent_failure": 0.02
  },
  "confidence": 0.65,
  "timestamp": "2024-01-15T10:30:05Z",
  "model_version": "v1.0.0"
}
```

### 3. Batch Predictions

**Endpoint**: `POST /predict/batch`

**Request**:

```json
{
  "sequences": [
    {
      "equipment_id": "EQ001",
      "sequence": [{ "temperature": 75.5 }]
    },
    {
      "equipment_id": "EQ002",
      "sequence": [{ "temperature": 80.2 }]
    }
  ]
}
```

**Response**:

```json
{
  "predictions": [
    {
      "equipment_id": "EQ001",
      "predicted_rul": 45.3,
      "health_status": "warning",
      "timestamp": "2024-01-15T10:30:05Z",
      "model_version": "v1.0.0"
    }
  ],
  "total_processed": 10,
  "processing_time": 0.25
}
```

### 4. List Models

**Endpoint**: `GET /models`

**Response**:

```json
[
  {
    "name": "LSTM RUL Predictor",
    "version": "v1.0.0",
    "type": "lstm",
    "loaded": true,
    "last_updated": "2024-01-10T14:20:00Z",
    "performance_metrics": {
      "rmse": 8.5,
      "mae": 6.2,
      "r2": 0.92
    }
  }
]
```

## Python Client Example

```python
import requests
import json

# Configuration
API_URL = "http://localhost:8000"

# Predict RUL
def predict_rul(equipment_id, sequence_data):
    response = requests.post(
        f"{API_URL}/predict/rul",
        json={
            "data": {
                "equipment_id": equipment_id,
                "sequence": sequence_data
            },
            "return_confidence": True
        }
    )
    return response.json()

# Example usage
sequence = [
    {"temperature": 75.5, "vibration": 0.42, "pressure": 102.3},
    {"temperature": 76.0, "vibration": 0.43, "pressure": 102.5}
]

result = predict_rul("EQ001", sequence)
print(f"Predicted RUL: {result['predicted_rul']:.1f} cycles")
print(f"Health Status: {result['health_status']}")
print(f"Confidence: {result['confidence_interval']}")
```

## Docker Deployment

### Build Image

```bash
docker build -t pm-inference:latest .
```

### Run Container

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e WORKERS=4 \
  pm-inference:latest
```

## Performance

- **Latency**: <50ms (single prediction)
- **Throughput**: ~200 requests/second (with 4 workers)
- **Batch Processing**: Up to 100 sequences per request

## API Documentation

Interactive documentation available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

All errors return JSON with:

```json
{
  "error": "ErrorType",
  "message": "Human-readable message",
  "detail": "Technical details",
  "timestamp": "2024-01-15T10:30:05Z"
}
```

**Common Error Codes**:

- `400`: Bad Request (invalid input)
- `404`: Not Found (endpoint doesn't exist)
- `422`: Validation Error (schema mismatch)
- `500`: Internal Server Error
- `503`: Service Unavailable (model not loaded)

## Monitoring

### Prometheus Metrics

Available at: `http://localhost:9090/metrics`

**Key Metrics**:

- `prediction_latency_seconds`: Prediction latency histogram
- `predictions_total`: Total predictions counter
- `model_loaded`: Model load status gauge
- `errors_total`: Error counter by type

### Health Checks

```bash
# Quick health check
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/models
```

## Testing

```bash
# Unit tests
pytest tests/

# Load testing
locust -f tests/load_test.py --host http://localhost:8000
```

## Troubleshooting

**Models not loading**:

- Check model paths in `config/inference_config.yaml`
- Verify models exist in specified directories
- Check logs: `tail -f logs/inference.log`

**High latency**:

- Increase worker count in config
- Enable model caching
- Use batch predictions
- Consider GPU inference

**Memory issues**:

- Reduce worker count
- Unload unused models: `POST /models/{model_id}/unload`
- Enable model quantization

## Next Steps

Module 7 (Inference API) complete! ✅

**Phase 3 continues**:

- **Module 8**: Alerting system for critical warnings
- **Module 9**: Dashboard and visualization

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Deployment](https://www.uvicorn.org/deployment/)
- [Pydantic Models](https://docs.pydantic.dev/)
