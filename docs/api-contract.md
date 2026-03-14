# API Contracts — Predictive Maintenance System

> **Derived from**: PRD + System Design (not existing code)
> **Version**: 1.0.0
> **Last Updated**: 2026-03-11

---

## 1. Inference Service API (FastAPI — port 8000)

Base URL: `http://<host>:8000`

### 1.1 Health Check

| Field              | Value                                                                     |
| ------------------ | ------------------------------------------------------------------------- |
| **Endpoint**       | `GET /health`                                                             |
| **Description**    | Returns service health, loaded model status, and dependency connectivity. |
| **Authentication** | None (internal)                                                           |

**Response 200:**

```json
{
  "status": "healthy",
  "timestamp": "2026-03-11T10:00:00Z",
  "models_loaded": {
    "lstm_rul": { "version": "v1.2.0", "loaded_at": "2026-03-11T08:00:00Z" },
    "rf_health": { "version": "v1.1.0", "loaded_at": "2026-03-11T08:00:00Z" }
  },
  "dependencies": {
    "mlflow": "connected",
    "kafka": "connected",
    "timescaledb": "connected",
    "redis": "connected"
  }
}
```

**Response 503:**

```json
{
  "status": "unhealthy",
  "error": "LSTM model not loaded",
  "timestamp": "2026-03-11T10:00:00Z"
}
```

---

### 1.2 List Models

| Field              | Value                                                                |
| ------------------ | -------------------------------------------------------------------- |
| **Endpoint**       | `GET /models`                                                        |
| **Description**    | Lists all loaded models with metadata (version, metrics, load time). |
| **Authentication** | None (internal)                                                      |

**Response 200:**

```json
{
  "models": [
    {
      "model_id": "lstm_rul",
      "model_type": "LSTM",
      "version": "v1.2.0",
      "task": "rul_prediction",
      "metrics": { "mae": 12.5, "rmse": 18.3, "r2": 0.87 },
      "loaded_at": "2026-03-11T08:00:00Z",
      "source": "mlflow"
    },
    {
      "model_id": "rf_health",
      "model_type": "RandomForest",
      "version": "v1.1.0",
      "task": "health_classification",
      "metrics": { "accuracy": 0.94, "f1": 0.91 },
      "loaded_at": "2026-03-11T08:00:00Z",
      "source": "mlflow"
    }
  ]
}
```

---

### 1.3 Predict RUL (Remaining Useful Life)

| Field              | Value                                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **Endpoint**       | `POST /predict/rul`                                                                                                            |
| **Description**    | Predicts remaining useful life for a single equipment based on a time-series sequence of sensor readings. Uses the LSTM model. |
| **Authentication** | API Key via `X-API-Key` header (production)                                                                                    |

**Request Body:**

```json
{
  "equipment_id": "ENGINE_0001",
  "equipment_type": "turbofan_engine",
  "sequence": [
    {
      "timestamp": "2026-03-11T09:59:00Z",
      "cycle": 150,
      "sensor_readings": {
        "temperature_2": 643.5,
        "temperature_3": 1590.0,
        "pressure_4": 9045.0,
        "corrected_fan_speed": 47.2,
        "fuel_flow": 8.44,
        "vibration": 23.5
      },
      "operational_settings": {
        "op_setting_1": -0.0007,
        "op_setting_2": -0.0004,
        "op_setting_3": 100.0
      }
    }
  ]
}
```

**Response 200:**

```json
{
  "equipment_id": "ENGINE_0001",
  "rul_hours": 72.5,
  "rul_cycles": 145,
  "anomaly_score": 0.35,
  "health_status": "warning",
  "confidence": 0.87,
  "confidence_interval": { "lower": 58.2, "upper": 86.8 },
  "model_version": "v1.2.0",
  "timestamp": "2026-03-11T10:00:00Z",
  "recommendations": [
    "Schedule maintenance within 72 hours",
    "Monitor temperature_3 trend — elevated"
  ]
}
```

**Response 400:**

```json
{
  "error": "validation_error",
  "message": "Sequence length must be >= 50 time steps",
  "details": { "received_length": 10, "required_length": 50 }
}
```

**Response 422:**

```json
{
  "error": "unprocessable_entity",
  "message": "Missing required sensor: temperature_3"
}
```

**Response 503:**

```json
{
  "error": "model_unavailable",
  "message": "LSTM model not loaded"
}
```

---

### 1.4 Predict Health Status

| Field              | Value                                                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Endpoint**       | `POST /predict/health`                                                                                                                |
| **Description**    | Classifies equipment health status (healthy / warning / critical / imminent_failure) using Random Forest model on extracted features. |
| **Authentication** | API Key via `X-API-Key` header (production)                                                                                           |

**Request Body:**

```json
{
  "equipment_id": "ENGINE_0001",
  "features": {
    "temperature_rolling_60_mean": 643.5,
    "temperature_rolling_60_std": 2.1,
    "vibration_rms": 23.5,
    "vibration_fft_dominant_freq": 0.15,
    "vibration_fft_spectral_energy": 1250.0,
    "pressure_rolling_60_mean": 9045.0,
    "fuel_flow_rolling_60_mean": 8.44,
    "corrected_fan_speed_rolling_60_mean": 47.2
  }
}
```

**Response 200:**

```json
{
  "equipment_id": "ENGINE_0001",
  "health_status": "warning",
  "health_status_code": 1,
  "probabilities": {
    "healthy": 0.15,
    "warning": 0.62,
    "critical": 0.18,
    "imminent_failure": 0.05
  },
  "anomaly_score": 0.45,
  "model_version": "v1.1.0",
  "timestamp": "2026-03-11T10:00:00Z"
}
```

**Response 400 / 422 / 503:** Same structure as 1.3.

---

### 1.5 Batch Predict RUL

| Field              | Value                                                                     |
| ------------------ | ------------------------------------------------------------------------- |
| **Endpoint**       | `POST /predict/batch`                                                     |
| **Description**    | Batch prediction for multiple equipment. Returns RUL and health for each. |
| **Authentication** | API Key via `X-API-Key` header (production)                               |

**Request Body:**

```json
{
  "predictions": [
    {
      "equipment_id": "ENGINE_0001",
      "sequence": [ ... ]
    },
    {
      "equipment_id": "ENGINE_0002",
      "sequence": [ ... ]
    }
  ]
}
```

**Response 200:**

```json
{
  "results": [
    {
      "equipment_id": "ENGINE_0001",
      "rul_hours": 72.5,
      "rul_cycles": 145,
      "anomaly_score": 0.35,
      "health_status": "warning",
      "confidence": 0.87,
      "model_version": "v1.2.0"
    },
    {
      "equipment_id": "ENGINE_0002",
      "rul_hours": 250.0,
      "rul_cycles": 500,
      "anomaly_score": 0.08,
      "health_status": "healthy",
      "confidence": 0.95,
      "model_version": "v1.2.0"
    }
  ],
  "batch_size": 2,
  "processing_time_ms": 145,
  "timestamp": "2026-03-11T10:00:00Z"
}
```

---

### 1.6 Model Reload

| Field              | Value                                                   |
| ------------------ | ------------------------------------------------------- |
| **Endpoint**       | `POST /models/reload`                                   |
| **Description**    | Forces reload of a specific model from MLflow registry. |
| **Authentication** | Admin API Key                                           |

**Request Body:**

```json
{
  "model_id": "lstm_rul",
  "version": "v1.3.0"
}
```

**Response 200:**

```json
{
  "status": "reloaded",
  "model_id": "lstm_rul",
  "version": "v1.3.0",
  "loaded_at": "2026-03-11T10:01:00Z"
}
```

---

## 2. Alert Service API (port 8001)

Base URL: `http://<host>:8001`

### 2.1 Process Prediction (Internal)

| Field              | Value                                                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Endpoint**       | `POST /alerts/evaluate`                                                                                                                                 |
| **Description**    | Accepts prediction data and evaluates alert rules. Triggers notifications if conditions are met. Called by the inference service after each prediction. |
| **Authentication** | Internal service token                                                                                                                                  |

**Request Body:**

```json
{
  "equipment_id": "ENGINE_0001",
  "rul": 8,
  "anomaly_score": 0.92,
  "health_status": "imminent_failure",
  "temperature": 98.5,
  "vibration": 0.85,
  "timestamp": "2026-03-11T10:00:00Z"
}
```

**Response 200:**

```json
{
  "alerts_triggered": 3,
  "alerts": [
    {
      "alert_id": "rul_critical_ENGINE_0001_1741694400",
      "rule_id": "rul_critical",
      "severity": "critical",
      "message": "Equipment ENGINE_0001 has critical RUL: 8 cycles remaining",
      "notifications_sent": ["database", "slack", "email"]
    }
  ]
}
```

---

### 2.2 Get Active Alerts

| Field                | Value                                                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Endpoint**         | `GET /alerts`                                                                                                                                  |
| **Description**      | Returns list of active (triggered or acknowledged) alerts.                                                                                     |
| **Authentication**   | API Key                                                                                                                                        |
| **Query Parameters** | `equipment_id` (optional), `severity` (optional: info, warning, critical), `status` (optional: triggered, acknowledged), `limit` (default: 50) |

**Response 200:**

```json
{
  "total": 5,
  "alerts": [
    {
      "alert_id": "rul_critical_ENGINE_0001_1741694400",
      "rule_id": "rul_critical",
      "equipment_id": "ENGINE_0001",
      "severity": "critical",
      "message": "Equipment ENGINE_0001 has critical RUL: 8 cycles remaining",
      "timestamp": "2026-03-11T10:00:00Z",
      "status": "triggered",
      "data": { "rul": 8, "anomaly_score": 0.92 }
    }
  ]
}
```

---

### 2.3 Acknowledge Alert

| Field              | Value                                     |
| ------------------ | ----------------------------------------- |
| **Endpoint**       | `POST /alerts/{alert_id}/acknowledge`     |
| **Description**    | Marks an alert as acknowledged by a user. |
| **Authentication** | API Key                                   |

**Request Body:**

```json
{
  "user": "john.doe@company.com"
}
```

**Response 200:**

```json
{
  "alert_id": "rul_critical_ENGINE_0001_1741694400",
  "status": "acknowledged",
  "acknowledged_by": "john.doe@company.com",
  "acknowledged_at": "2026-03-11T10:05:00Z"
}
```

---

### 2.4 Resolve Alert

| Field              | Value                             |
| ------------------ | --------------------------------- |
| **Endpoint**       | `POST /alerts/{alert_id}/resolve` |
| **Description**    | Marks an alert as resolved.       |
| **Authentication** | API Key                           |

**Response 200:**

```json
{
  "alert_id": "rul_critical_ENGINE_0001_1741694400",
  "status": "resolved",
  "resolved_at": "2026-03-11T12:00:00Z"
}
```

---

### 2.5 Alert Statistics

| Field              | Value                                                                                   |
| ------------------ | --------------------------------------------------------------------------------------- |
| **Endpoint**       | `GET /alerts/statistics`                                                                |
| **Description**    | Returns aggregate statistics: counts by severity, mean acknowledgment time, alert rate. |
| **Authentication** | API Key                                                                                 |

**Response 200:**

```json
{
  "total_alerts": 245,
  "active_alerts": 5,
  "severity_counts": { "critical": 12, "warning": 83, "info": 150 },
  "status_counts": { "triggered": 3, "acknowledged": 2, "resolved": 240 },
  "average_acknowledgment_time_seconds": 180,
  "alerts_per_hour_24h": 4.2
}
```

---

## 3. Equipment Data API (port 8002)

> This API provides equipment metadata and sensor history to the dashboard.

### 3.1 List Equipment

| Field                | Value                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------- |
| **Endpoint**         | `GET /equipment`                                                                             |
| **Description**      | Lists all registered equipment with latest health status.                                    |
| **Authentication**   | API Key                                                                                      |
| **Query Parameters** | `type` (optional), `location` (optional), `health_status` (optional), `limit` (default: 100) |

**Response 200:**

```json
{
  "total": 100,
  "equipment": [
    {
      "equipment_id": "ENGINE_0001",
      "equipment_type": "turbofan_engine",
      "location": "Factory_Floor_1",
      "install_date": "2024-01-01",
      "latest_rul": 145,
      "latest_health_status": "warning",
      "latest_anomaly_score": 0.35,
      "last_maintenance": "2025-12-01",
      "last_updated": "2026-03-11T10:00:00Z"
    }
  ]
}
```

---

### 3.2 Get Equipment Detail

| Field              | Value                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------ |
| **Endpoint**       | `GET /equipment/{equipment_id}`                                                                        |
| **Description**    | Returns full equipment detail including latest sensor readings, feature history, and prediction trend. |
| **Authentication** | API Key                                                                                                |

**Response 200:**

```json
{
  "equipment_id": "ENGINE_0001",
  "equipment_type": "turbofan_engine",
  "location": "Factory_Floor_1",
  "install_date": "2024-01-01",
  "total_cycles": 1500,
  "latest_sensor_readings": { ... },
  "latest_prediction": {
    "rul_hours": 72.5,
    "health_status": "warning",
    "anomaly_score": 0.35,
    "timestamp": "2026-03-11T10:00:00Z"
  },
  "rul_trend_24h": [ ... ],
  "maintenance_history": [ ... ],
  "active_alerts": [ ... ]
}
```

---

### 3.3 Get Sensor History

| Field                | Value                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Endpoint**         | `GET /equipment/{equipment_id}/sensors`                                                                            |
| **Description**      | Returns historical sensor readings for a given time window.                                                        |
| **Authentication**   | API Key                                                                                                            |
| **Query Parameters** | `start_time`, `end_time`, `sensors` (comma-separated list), `resolution` (1s, 1m, 1h, 1d), `limit` (default: 1000) |

**Response 200:**

```json
{
  "equipment_id": "ENGINE_0001",
  "time_range": {
    "start": "2026-03-10T10:00:00Z",
    "end": "2026-03-11T10:00:00Z"
  },
  "resolution": "1m",
  "readings": [
    {
      "timestamp": "2026-03-10T10:00:00Z",
      "temperature_2": 643.2,
      "vibration": 23.4,
      "pressure_4": 9050.1
    }
  ]
}
```

---

### 3.4 Get Prediction History

| Field                | Value                                                                    |
| -------------------- | ------------------------------------------------------------------------ |
| **Endpoint**         | `GET /equipment/{equipment_id}/predictions`                              |
| **Description**      | Returns historical predictions (RUL, anomaly scores) for trend analysis. |
| **Authentication**   | API Key                                                                  |
| **Query Parameters** | `start_time`, `end_time`, `limit` (default: 500)                         |

**Response 200:**

```json
{
  "equipment_id": "ENGINE_0001",
  "predictions": [
    {
      "timestamp": "2026-03-11T10:00:00Z",
      "rul_hours": 72.5,
      "anomaly_score": 0.35,
      "health_status": "warning",
      "model_version": "v1.2.0"
    }
  ]
}
```

---

## 4. ML Pipeline API (port 8003)

> Manages model training, evaluation, and deployment. Could be internal only.

### 4.1 Trigger Training

| Field              | Value                                  |
| ------------------ | -------------------------------------- |
| **Endpoint**       | `POST /train`                          |
| **Description**    | Triggers a full training pipeline run. |
| **Authentication** | Admin API Key                          |

**Request Body:**

```json
{
  "model_types": ["lstm", "random_forest"],
  "equipment_ids": ["ENGINE_0001", "ENGINE_0002"],
  "config_overrides": {
    "epochs": 100,
    "sequence_length": 50
  }
}
```

**Response 202:**

```json
{
  "job_id": "train_20260311_100000",
  "status": "accepted",
  "message": "Training pipeline started"
}
```

---

### 4.2 Get Training Status

| Field              | Value                                    |
| ------------------ | ---------------------------------------- |
| **Endpoint**       | `GET /train/{job_id}`                    |
| **Description**    | Returns training job status and metrics. |
| **Authentication** | Admin API Key                            |

**Response 200:**

```json
{
  "job_id": "train_20260311_100000",
  "status": "completed",
  "started_at": "2026-03-11T10:00:00Z",
  "completed_at": "2026-03-11T11:30:00Z",
  "results": {
    "lstm": {
      "mae": 12.5,
      "rmse": 18.3,
      "r2": 0.87,
      "model_version": "v1.3.0"
    },
    "random_forest": { "accuracy": 0.94, "f1": 0.91, "model_version": "v1.2.0" }
  }
}
```

---

### 4.3 Check Drift

| Field              | Value                                          |
| ------------------ | ---------------------------------------------- |
| **Endpoint**       | `GET /drift`                                   |
| **Description**    | Returns current data and concept drift status. |
| **Authentication** | Admin API Key                                  |

**Response 200:**

```json
{
  "data_drift_detected": true,
  "concept_drift_detected": false,
  "drifted_features": ["temperature_3", "vibration"],
  "ks_test_results": {
    "temperature_3": { "statistic": 0.15, "p_value": 0.02 }
  },
  "recommendation": "retraining_recommended",
  "last_check": "2026-03-11T08:00:00Z"
}
```

---

### 4.4 Promote Model

| Field              | Value                                                            |
| ------------------ | ---------------------------------------------------------------- |
| **Endpoint**       | `POST /models/{model_name}/promote`                              |
| **Description**    | Promotes a model version to production stage in MLflow registry. |
| **Authentication** | Admin API Key                                                    |

**Request Body:**

```json
{
  "version": "v1.3.0",
  "reason": "Better MAE (12.5 vs 14.2)"
}
```

**Response 200:**

```json
{
  "model_name": "lstm_rul_predictor",
  "version": "v1.3.0",
  "stage": "Production",
  "promoted_at": "2026-03-11T12:00:00Z"
}
```

---

## 5. Observability API (port 9090)

### 5.1 Prometheus Metrics

| Field              | Value                                  |
| ------------------ | -------------------------------------- |
| **Endpoint**       | `GET /metrics`                         |
| **Description**    | Exposes Prometheus-compatible metrics. |
| **Authentication** | None                                   |

**Metrics exposed:**

```
# HELP inference_requests_total Total number of inference requests
# TYPE inference_requests_total counter
inference_requests_total{model="lstm_rul",status="success"} 15234

# HELP inference_latency_seconds Inference latency histogram
# TYPE inference_latency_seconds histogram
inference_latency_seconds_bucket{model="lstm_rul",le="0.1"} 14500

# HELP model_prediction_rul_hours Current RUL prediction
# TYPE model_prediction_rul_hours gauge
model_prediction_rul_hours{equipment_id="ENGINE_0001"} 72.5

# HELP alert_total Total alerts triggered
# TYPE alert_total counter
alert_total{severity="critical"} 12

# HELP kafka_consumer_lag Consumer group lag
# TYPE kafka_consumer_lag gauge
kafka_consumer_lag{group="stream-processor-group",topic="raw_sensor_data"} 15

# HELP feature_computation_seconds Feature computation time
# TYPE feature_computation_seconds histogram
feature_computation_seconds_bucket{le="0.05"} 9500
```

---

## 6. Kafka Topics (Message Schemas)

### 6.1 `raw_sensor_data`

```json
{
  "equipment_id": "ENGINE_0001",
  "equipment_type": "turbofan_engine",
  "timestamp": "2026-03-11T10:00:00Z",
  "cycle": 150,
  "data_source": "cmapss",
  "operational_settings": {
    "op_setting_1": -0.0007,
    "op_setting_2": -0.0004,
    "op_setting_3": 100.0
  },
  "sensor_readings": {
    "sensor_1": 518.67,
    "sensor_2": 643.5,
    "sensor_3": 1590.0
  },
  "metadata": {
    "location": "Factory_Floor_1",
    "model": "TURBOFAN_ENGINE_v1",
    "install_date": "2024-01-01T00:00:00Z"
  }
}
```

### 6.2 `processed_features`

```json
{
  "equipment_id": "ENGINE_0001",
  "timestamp": "2026-03-11T10:00:00Z",
  "features": {
    "temperature_2_rolling_60_mean": 643.2,
    "temperature_2_rolling_60_std": 1.8,
    "vibration_fft_dominant_freq": 0.15,
    "vibration_fft_spectral_energy": 1250.0,
    "sensors_cv": 0.45
  },
  "feature_type": "combined"
}
```

### 6.3 `failure_predictions`

```json
{
  "equipment_id": "ENGINE_0001",
  "timestamp": "2026-03-11T10:00:00Z",
  "rul_hours": 72.5,
  "rul_cycles": 145,
  "anomaly_score": 0.35,
  "health_status": "warning",
  "model_version": "v1.2.0",
  "confidence": 0.87
}
```

### 6.4 `maintenance_alerts`

```json
{
  "alert_id": "rul_critical_ENGINE_0001_1741694400",
  "equipment_id": "ENGINE_0001",
  "severity": "critical",
  "rule_id": "rul_critical",
  "message": "Equipment ENGINE_0001 has critical RUL: 8 cycles remaining",
  "timestamp": "2026-03-11T10:00:00Z",
  "data": { "rul": 8, "anomaly_score": 0.92 }
}
```

---

## 7. Cross-Cutting Concerns

### 7.1 Authentication (Production)

All public-facing APIs should use API Key authentication:

- Header: `X-API-Key: <key>`
- Admin endpoints: separate admin key
- Internal service-to-service: JWT or mutual TLS

### 7.2 Rate Limiting

| Endpoint Group | Rate Limit          |
| -------------- | ------------------- |
| `/predict/*`   | 100 requests/minute |
| `/alerts/*`    | 50 requests/minute  |
| `/equipment/*` | 200 requests/minute |
| `/train/*`     | 5 requests/hour     |

### 7.3 Error Response Schema (Standard)

All errors follow this format:

```json
{
  "error": "error_code",
  "message": "Human-readable message",
  "details": { ... },
  "timestamp": "2026-03-11T10:00:00Z",
  "request_id": "uuid-v4"
}
```

### 7.4 Pagination

List endpoints use cursor-based pagination:

```
GET /equipment?limit=50&cursor=eyJpZCI6MTAwfQ==
```

Response includes:

```json
{
  "data": [ ... ],
  "pagination": {
    "total": 500,
    "limit": 50,
    "next_cursor": "eyJpZCI6MTUwfQ==",
    "has_more": true
  }
}
```

---

## Assumptions

1. **No user authentication system in PRD** — API Key-based auth is assumed for production readiness.
2. **Equipment Data API (port 8002)** is inferred from the dashboard's need for equipment data; not explicitly called out in the system design.
3. **ML Pipeline API (port 8003)** is inferred from the need to trigger training and manage models programmatically.
4. **Observability API** is inferred from the system design requirement for Prometheus metrics.
5. Kafka message schemas are derived from the system design data flow descriptions and the data_generator's message format.
