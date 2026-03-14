# Expected Behaviour — Predictive Maintenance System

> **Derived from**: PRD + System Design (not existing code)
> **Version**: 1.0.0
> **Last Updated**: 2026-03-11

---

## 1. System Startup Behaviour

### 1.1 Infrastructure Bootstrap

1. **Kafka cluster** starts first (Zookeeper → Kafka → Kafdrop). Health verified via Kafdrop UI on port 9000.
2. **TimescaleDB** starts and executes `init-db` SQL scripts to create all tables, hypertables, indexes, retention policies, and continuous aggregates.
3. **MinIO** starts and creates the default bucket `predictive-maintenance` for model artifacts.
4. **Redis** starts and is available on port 6379.
5. All infrastructure services are healthy before application services start (`depends_on` with healthchecks).

### 1.2 Application Bootstrap

1. **Data Simulator / Data Loader** starts producing sensor data to `raw_sensor_data` Kafka topic.
2. **Stream Processor** connects to Kafka consumer group, begins consuming `raw_sensor_data`, extracts features, and writes to TimescaleDB (`sensor_readings` + `engineered_features` tables) and publishes to `processed_features` topic.
3. **Feature Store** (offline) can be triggered manually or on schedule to generate training datasets from TimescaleDB data.
4. **Inference Service** loads models from MLflow registry (production stage), starts FastAPI on port 8000, and reports healthy on `/health`.
5. **Alert Engine** connects to `failure_predictions` Kafka topic (or is called via API after each inference), loads alert rules from config, and begins evaluating.
6. **Dashboard** connects to the Equipment Data API and Inference API, starts Streamlit on port 8501 and Grafana on port 3000.
7. **MLflow** starts on port 5001 with PostgreSQL backend and MinIO artifact store.

### 1.3 First-Run Behaviour

On first deployment with no trained models:

1. System should log a clear warning: "No trained models found. Running initial training pipeline."
2. Data Loader streams C-MAPSS training data to Kafka.
3. Stream Processor processes data and populates TimescaleDB.
4. Feature Store generates labeled training dataset.
5. Training pipeline trains LSTM and Random Forest models.
6. Models are registered in MLflow and promoted to production stage.
7. Inference service auto-detects new production models and loads them.
8. System transitions to normal operation.

---

## 2. Data Ingestion Pipeline

### 2.1 Sensor Data Flow

**Input**: Raw sensor readings from equipment (IoT sensors or simulated data).

**Expected behaviour per reading:**

1. Equipment simulator generates a reading with: `equipment_id`, `timestamp`, `cycle`, `sensor_readings` (dict of sensor name → value), `operational_settings`.
2. Reading is published to Kafka topic `raw_sensor_data` as a JSON message. Kafka key = `equipment_id` (ensures ordering per equipment).
3. Stream Processor consumes the message within **< 100ms** (p99).
4. Data quality check runs:
   - Null/missing values → log warning, impute via forward-fill or drop.
   - Out-of-range values → flag as `outlier`, log anomaly.
   - Duplicate timestamps → deduplicate (keep latest).
5. Time-domain features computed (rolling mean, std, min, max, rate of change, cross-sensor correlations) over configurable windows (default: 30, 60, 120 timesteps).
6. Frequency-domain features computed (FFT dominant frequency, spectral energy, spectral entropy, cepstral coefficients).
7. Raw reading + computed features written to TimescaleDB.
8. Features published to `processed_features` Kafka topic.

**Throughput target**: 10,000 sensor readings/second across all equipment.

**Latency target**: End-to-end from sensor → features in DB < 500ms (p95).

### 2.2 Data Quality Validation

For every incoming reading, the following checks apply:

| Check                                  | Action on Failure                                  |
| -------------------------------------- | -------------------------------------------------- |
| Missing `equipment_id`                 | Reject message, log error                          |
| Missing `timestamp`                    | Reject message, log error                          |
| All sensor values null                 | Reject message, log error                          |
| > 30% sensors null                     | Flag as `missing`, impute and proceed with warning |
| Sensor value out of configured range   | Flag as `outlier`, include but annotate            |
| Duplicate timestamp for same equipment | Keep latest, discard duplicate                     |
| Timestamp > 5 minutes in future        | Reject, log clock drift warning                    |
| Timestamp > 24 hours in past           | Accept but flag as `backfill`                      |

### 2.3 C-MAPSS Data Loading

1. CMAPSSLoader reads `train_FD001.txt` through `train_FD004.txt`.
2. Columns are parsed: `unit_number`, `cycle`, 3 operational settings, 21 sensors.
3. RUL labels are computed using piecewise linear degradation (max RUL cap = 125 cycles).
4. Data is normalized per equipment using min-max scaling.
5. Sensor names are mapped from generic `sensor_1..21` to descriptive names (e.g., `fan_inlet_temperature`, `lpc_outlet_temperature`).
6. Data is streamed to Kafka in chronological order (sorted by unit + cycle).
7. Streaming respects configurable `delay_between_cycles` to simulate real-time ingestion.

---

## 3. Feature Engineering

### 3.1 Stream Processor Features (Real-Time)

Computed on every incoming reading using a sliding window buffer:

**Time-Domain Features:**

- Rolling statistics: mean, std, min, max, median (windows: 30, 60, 120)
- Rate of change (first derivative)
- Coefficient of variation
- Cross-sensor correlations (Pearson)
- Cumulative statistics: cumulative sum, cumulative max

**Frequency-Domain Features:**

- FFT dominant frequency
- FFT spectral energy
- FFT spectral entropy
- Cepstral coefficients (first 5)
- Short-time Fourier transform features (for longer windows)

**Output**: One feature vector per equipment per timestamp, containing ~50–100 features depending on sensor count.

### 3.2 Feature Store Features (Offline / Batch)

Computed periodically or on-demand for training:

**Additional features beyond stream processor:**

- Lag features: sensor values at t-1, t-5, t-10, t-20
- Exponential moving averages (spans: 10, 20, 50)
- Cumulative degradation indicators
- Interaction features (ratio of correlated sensors)

**Label Generation:**

- `rul` (float): Remaining useful life in cycles. Piecewise linear with max cap at 125.
- `failure_within_30` (bool): Binary label — will equipment fail in next 30 cycles?
- `health_status` (int): 0=healthy (RUL>60), 1=warning (30<RUL≤60), 2=critical (15<RUL≤30), 3=imminent_failure (RUL≤15)
- `degradation_rate` (float): Rate of RUL decline

**Dataset Splitting:**

- Equipment-based split (no data leakage across equipment)
- Default: 70% train, 15% validation, 15% test
- Temporal ordering preserved within each equipment

---

## 4. Model Training Pipeline

### 4.1 LSTM RUL Predictor

**Architecture:**

- Input: Sequences of length 50 (configurable), with N features per timestep
- 3 LSTM layers (128 → 64 → 32 units) with dropout (0.3)
- Custom Attention layer after final LSTM
- Dense layers: 64 → 32 → 1 (RUL output)
- Activation: ReLU (hidden), Linear (output)

**Training:**

- Optimizer: Adam (lr=0.001 with ReduceLROnPlateau)
- Loss: Huber loss (delta=1.0) — robust to RUL outliers
- Batch size: 64
- Epochs: 100 (with EarlyStopping patience=15)
- Callbacks: ModelCheckpoint (best val_loss), EarlyStopping, ReduceLROnPlateau

**Expected metrics (on C-MAPSS FD001):**

- MAE: < 15 cycles
- RMSE: < 20 cycles
- R²: > 0.85
- NASA Score: < 500

### 4.2 Random Forest Health Classifier

**Architecture:**

- 200 estimators (configurable)
- max_depth: 20
- class_weight: 'balanced' (handles imbalanced health classes)
- Features: same as LSTM input but flattened (latest timestep features + rolling window stats)

**Expected metrics:**

- Accuracy: > 90%
- F1 (weighted): > 0.88
- Per-class recall for `imminent_failure`: > 0.95 (critical — must not miss impending failures)

### 4.3 Hyperparameter Tuning

- **LSTM**: Manual random search over learning rate, LSTM units, dropout, sequence length, batch size. Limited by GPU time.
- **Random Forest**: RandomizedSearchCV with 5-fold time-series cross-validation. Search over n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features.

### 4.4 Model Registration

After training:

1. All parameters, metrics, and artifacts are logged to MLflow.
2. Model is registered in MLflow Model Registry under name `lstm_rul_predictor` or `rf_health_classifier`.
3. Model is initially set to `Staging` stage.
4. Model Comparator runs: compares new model against current Production model.
5. If new model is statistically significantly better (paired t-test, p < 0.05): promote to Production, archive old version.
6. If not better: log comparison, keep existing Production model.

---

## 5. Inference Service

### 5.1 Model Loading

On startup:

1. Connect to MLflow tracking server.
2. Query for models in `Production` stage: `lstm_rul_predictor` and `rf_health_classifier`.
3. Download model artifacts from MinIO.
4. Load models into memory (TensorFlow for LSTM, scikit-learn for RF).
5. Warm up models with a dummy prediction to initialize graphs.
6. Report ready on `/health` endpoint.

On model update:

1. Background process polls MLflow registry every 60 seconds.
2. If a new Production version is detected, load it in a shadow slot.
3. Run validation inference on a test batch.
4. If validation passes, hot-swap model atomically (no downtime).
5. Log model version change.

### 5.2 RUL Prediction Flow

1. Receive `POST /predict/rul` with equipment_id and sensor sequence.
2. **Validate**: sequence length ≥ 50, all required sensors present, values in valid range.
3. **Preprocess**: normalize sensor values using training-time scaler (stored with model artifact).
4. **Predict**: Run LSTM forward pass. Apply MC Dropout (N=10 forward passes) for uncertainty estimation.
5. **Post-process**:
   - Mean of MC Dropout predictions = point estimate
   - Std = uncertainty
   - Confidence interval = mean ± 2\*std
   - Clip RUL to [0, max_rul_cap]
6. **Classify health**: Map RUL to health_status using thresholds (healthy >60, warning 30-60, critical 15-30, imminent <15).
7. **Compute anomaly score**: Normalize prediction uncertainty and deviation from historical trend.
8. **Return** response with RUL, confidence, health_status, anomaly_score, recommendations.

**Latency target**: < 50ms per single prediction (p95).

### 5.3 Health Classification Flow

1. Receive `POST /predict/health` with equipment_id and feature vector.
2. **Validate**: all required features present.
3. **Predict**: Run Random Forest `.predict_proba()`.
4. **Return**: class label, probability distribution, anomaly score.

### 5.4 Batch Prediction

1. Receive `POST /predict/batch` with array of prediction requests.
2. Process sequentially or in parallel (configurable).
3. Return aggregated results.
4. Maximum batch size: 100 (configurable).

---

## 6. Alert Engine

### 6.1 Alert Evaluation

**Trigger**: After every prediction, the prediction result is evaluated against all active alert rules.

**Rule evaluation logic:**

1. Load active rules from config/database.
2. For each rule, evaluate condition expression against prediction data.
3. If condition is true AND rule is not in cooldown period for this equipment → trigger alert.
4. Alert severity inherited from rule.
5. Alert ID generated as: `{rule_id}_{equipment_id}_{timestamp_epoch}`.

**Built-in rules (from system design):**

| Rule ID                | Condition                             | Severity | Cooldown |
| ---------------------- | ------------------------------------- | -------- | -------- |
| `rul_critical`         | `rul < 10`                            | critical | 1 hour   |
| `rul_warning`          | `rul < 30`                            | warning  | 2 hours  |
| `rul_info`             | `rul < 50`                            | info     | 4 hours  |
| `anomaly_critical`     | `anomaly_score > 0.9`                 | critical | 30 min   |
| `anomaly_warning`      | `anomaly_score > 0.7`                 | warning  | 1 hour   |
| `health_imminent`      | `health_status == 'imminent_failure'` | critical | 30 min   |
| `health_critical`      | `health_status == 'critical'`         | warning  | 1 hour   |
| `temperature_high`     | `temperature > 95`                    | warning  | 2 hours  |
| `vibration_high`       | `vibration > 0.8`                     | warning  | 2 hours  |
| `rapid_degradation`    | `rul_change_rate > 5`                 | critical | 1 hour   |
| `multi_sensor_anomaly` | `anomaly_count >= 3`                  | critical | 30 min   |

### 6.2 Notification Channels

When an alert triggers:

1. **Database**: Write alert record to `alerts` table with status `triggered`.
2. **Email**: Send formatted HTML email to configured recipients based on severity routing.
3. **Slack**: Post to configured Slack channel via webhook with Block Kit formatting. Include equipment ID, severity, RUL, and link to dashboard.
4. **Webhook**: Send JSON payload to generic webhook URL for integration with external systems (PagerDuty, ServiceNow, etc.).

**Notification routing (by severity):**

- `critical` → Database + Email + Slack + Webhook
- `warning` → Database + Email + Slack
- `info` → Database only

### 6.3 Alert Lifecycle

```
TRIGGERED → ACKNOWLEDGED → RESOLVED
    ↓            ↓
 (suppressed) (auto-resolved if condition clears)
```

1. **Triggered**: Alert condition met. Notifications sent. Recorded in DB.
2. **Acknowledged**: Operator clicks "Acknowledge" in dashboard or calls API. Records who acknowledged and when.
3. **Resolved**: Maintenance is performed, or condition clears (auto-resolve if RUL recovers). Records resolution time.
4. **Suppressed**: Alert matches a suppression rule (e.g., equipment is in scheduled maintenance). No notifications sent, but recorded.

### 6.4 Alert Aggregation

- Multiple alerts for the same equipment within a configurable window (default: 5 minutes) are aggregated into a single notification.
- Aggregation groups by equipment_id and severity.
- Notification contains summary: "3 critical alerts for ENGINE_0001 in past 5 minutes".

---

## 7. Dashboard

### 7.1 Streamlit Dashboard Views

**Equipment Health Overview (Main Page):**

- Grid/card view of all equipment.
- Each card shows: equipment_id, health_status (color-coded), latest RUL, anomaly_score, last updated time.
- Color coding: Green (healthy), Yellow (warning), Orange (critical), Red (imminent_failure).
- Click on card → navigates to equipment detail view.

**Equipment Detail View:**

- Real-time sensor readings (line charts, auto-refreshing every 5 seconds).
- RUL trend over last 24h / 7d / 30d (line chart with confidence interval band).
- Health status history (timeline).
- Anomaly score trend.
- Active alerts for this equipment.
- Maintenance history.

**Alert Dashboard:**

- Table of active alerts (sortable by severity, time, equipment).
- Acknowledge/resolve buttons.
- Alert statistics: alerts per hour, mean time to acknowledge, severity distribution.

**Model Performance:**

- Latest model metrics (MAE, RMSE, R², F1).
- Prediction distribution histogram.
- Actual vs. predicted RUL scatter plot (if ground truth available).
- Drift status indicator.

**System Health:**

- Kafka consumer lag.
- Inference latency percentiles.
- Feature processing throughput.
- Service uptime.

### 7.2 Grafana Dashboard

Pre-configured panels:

- Sensor time-series per equipment (TimescaleDB data source).
- RUL predictions over time (TimescaleDB data source).
- Inference latency histogram (Prometheus data source).
- Kafka consumer lag (Prometheus data source).
- Alert rate (TimescaleDB data source).
- Equipment health status heatmap.

**Auto-refresh**: 10 seconds.

---

## 8. Retraining Pipeline

### 8.1 Trigger Conditions

Retraining is triggered by any of:

1. **Scheduled**: Weekly (configurable via cron or Airflow DAG).
2. **Data drift detected**: KS test p-value < 0.05 on any critical feature.
3. **Concept drift detected**: Prediction error increases > 20% compared to baseline.
4. **Manual trigger**: Operator calls `POST /train` API.

### 8.2 Retraining Flow

1. **Drift detection** runs on latest data window vs. training data distribution.
2. If drift detected, log to `drift_logs` table and trigger retraining.
3. **Feature Store** generates fresh training dataset from recent TimescaleDB data.
4. **Training pipeline** trains new LSTM and RF models.
5. **Model Comparator** compares new model vs. current Production model:
   - Run both models on the same test set.
   - Paired t-test: is the difference statistically significant (p < 0.05)?
   - Is the new model better on all primary metrics (MAE for RUL, F1 for health)?
6. If better: **Deployment Manager** promotes new model to Production in MLflow, archives old.
7. **Inference Service** detects new Production model, hot-swaps within 60 seconds.
8. If not better: Log comparison, keep existing model. Alert data team.

### 8.3 Rollback

If a newly deployed model causes issues:

1. Operator calls rollback API.
2. Previous Production model is restored from MLflow archive.
3. Inference service reloads previous model.
4. Incident logged.

---

## 9. Observability

### 9.1 Metrics (Prometheus)

Every service exposes `/metrics` endpoint with:

| Metric                        | Type      | Labels            |
| ----------------------------- | --------- | ----------------- |
| `inference_requests_total`    | Counter   | model, status     |
| `inference_latency_seconds`   | Histogram | model             |
| `prediction_rul_hours`        | Gauge     | equipment_id      |
| `alert_total`                 | Counter   | severity, rule_id |
| `kafka_consumer_lag`          | Gauge     | group, topic      |
| `feature_computation_seconds` | Histogram | feature_type      |
| `sensor_data_rate`            | Gauge     | equipment_id      |
| `model_reload_total`          | Counter   | model             |
| `data_quality_rejected_total` | Counter   | reason            |
| `db_write_latency_seconds`    | Histogram | table             |

### 9.2 Logging

Structured JSON logging with fields:

```json
{
  "timestamp": "2026-03-11T10:00:00Z",
  "level": "INFO",
  "service": "inference_service",
  "message": "RUL prediction completed",
  "equipment_id": "ENGINE_0001",
  "rul": 72.5,
  "latency_ms": 28,
  "model_version": "v1.2.0",
  "request_id": "uuid-v4"
}
```

Log levels:

- **ERROR**: Unrecoverable errors, model load failures, DB connection lost.
- **WARNING**: Data quality issues, approaching thresholds, retry attempts.
- **INFO**: Predictions, alerts, model loads, training progress.
- **DEBUG**: Feature values, intermediate computations, Kafka offsets.

### 9.3 Health Checks

Each service exposes `GET /health` returning:

```json
{
  "status": "healthy",
  "checks": {
    "kafka": "connected",
    "timescaledb": "connected",
    "redis": "connected",
    "model_loaded": true
  }
}
```

Health check is called by Docker/K8s every 30 seconds. Unhealthy service triggers restart after 3 consecutive failures.

---

## 10. Error Handling & Edge Cases

### 10.1 Kafka Unavailability

- **Stream Processor**: Retries connection with exponential backoff (1s, 2s, 4s, 8s, ..., max 60s). Buffers up to 1000 messages locally. If buffer full, drops oldest messages and logs warning.
- **Data Simulator**: Same retry strategy. Pauses simulation during disconnect.
- **Inference Service**: Continues serving from cached features. Logs Kafka disconnect. Resumes consuming on reconnect; processes backlog.

### 10.2 TimescaleDB Unavailability

- **Stream Processor**: Buffers features in memory queue (max 10,000 records). Retries writes with exponential backoff. If buffer full, writes to local disk fallback file.
- **Feature Store**: Fails gracefully. Training pipeline waits or uses cached training data.
- **Inference Service**: Uses Redis feature cache as fallback. Logs DB unavailability.

### 10.3 Model Load Failure

- If LSTM model fails to load: inference service returns `503` on RUL endpoints. Health endpoint reports `unhealthy`. Alert sent.
- If RF model fails to load: health classification endpoints return `503`. RUL still works.
- If MLflow is unreachable: use last-loaded model in memory. Log warning. Retry MLflow connection.

### 10.4 Invalid Sensor Data

- Missing sensor values (< 30% missing): Forward-fill imputation, flag as `interpolated`.
- Missing sensor values (≥ 30% missing): Reject reading, log error, increment `data_quality_rejected_total` metric.
- NaN / Inf: Replace with sensor-specific default. Log warning.
- Negative values for sensors that must be positive: Clamp to 0, log warning.

### 10.5 Equipment Not Found

- Prediction request for unknown equipment_id: Return `404` with message "Equipment not registered".
- New equipment can be auto-registered on first sensor reading (configurable).

### 10.6 Concurrent Model Reload

- Model reload uses read-write lock. Predictions in-flight complete with old model. New predictions use new model. No requests are dropped.

---

## 11. Performance Requirements

| Metric                        | Target                   | Measurement                  |
| ----------------------------- | ------------------------ | ---------------------------- |
| Sensor ingestion throughput   | ≥ 10,000 readings/sec    | Kafka consumer group metrics |
| Feature computation latency   | < 50ms per reading (p95) | Prometheus histogram         |
| RUL prediction latency        | < 50ms (p95)             | Prometheus histogram         |
| Health classification latency | < 10ms (p95)             | Prometheus histogram         |
| Batch prediction (100 items)  | < 2s                     | API response time            |
| Dashboard refresh             | Every 5 seconds          | Streamlit auto-refresh       |
| Alert notification latency    | < 5s from prediction     | End-to-end measurement       |
| Model reload (hot-swap)       | < 30s                    | Log timestamps               |
| TimescaleDB write throughput  | ≥ 5,000 rows/sec         | DB metrics                   |

---

## 12. Security Requirements

| Requirement                | Detail                                                |
| -------------------------- | ----------------------------------------------------- |
| API authentication         | API Key via `X-API-Key` header                        |
| Admin authentication       | Separate admin API key                                |
| Service-to-service auth    | JWT or mTLS                                           |
| Data encryption at rest    | TimescaleDB TDE or volume encryption                  |
| Data encryption in transit | TLS 1.2+ for all HTTP APIs                            |
| Kafka encryption           | SASL_SSL in production                                |
| Secret management          | Environment variables (minimum); Vault (recommended)  |
| Audit logging              | All API calls logged with request_id, user, timestamp |

---

## Assumptions

1. **Single-factory deployment** — no multi-tenancy or geo-distribution.
2. **Equipment count**: up to 1,000 monitored equipment units.
3. **Sensor frequency**: 1 reading per second per equipment (configurable).
4. **Internet connectivity**: not required for core function; only for external notifications (Slack, email).
5. **GPU**: optional for training (CPU training supported, slower). Inference runs on CPU.
6. **Time zone**: All timestamps in UTC (TIMESTAMPTZ).
