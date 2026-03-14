Predictive Maintenance System - Production Architecture

1. System Overview
   Real-world flow:
   Sensors → Edge Gateway → Message Queue → Feature Store → ML Pipeline → Inference API → Dashboard + Alerts

2. Architecture Diagram (Mental Model)
   [Factory Floor]
   ├─ IoT Sensors (vibration, temp, pressure, power)
   └─ Edge Device (data aggregation, initial filtering)
   ↓
   [Kafka Topic: raw_sensor_data]
   ↓
   [Stream Processor - Apache Flink / Python Consumer]
   ├─ Data validation
   ├─ Feature engineering (rolling stats, FFT)
   └─ Writes to → TimescaleDB (time-series) + S3 (raw backup)
   ↓
   [ML Pipeline]
   ├─ Offline Training (daily/weekly): LSTM + Random Forest
   ├─ Model Registry (MLflow)
   └─ Batch feature computation
   ↓
   [Real-time Inference Service - FastAPI]
   ├─ Loads latest model from MLflow
   ├─ Consumes from Kafka: processed_features
   └─ Predicts: RUL (Remaining Useful Life), Anomaly Score
   ↓
   [Alert System]
   ├─ Threshold-based triggers (RUL < 48hrs)
   └─ Sends → Email / Slack / SMS
   ↓
   [Dashboard - Streamlit/Grafana]
   └─ Real-time monitoring, historical trends, equipment health

3. Tech Stack (with justifications)
   ComponentToolWhy This?Message QueueApache KafkaIndustry standard for IoT streaming; handles 100k+ events/sec; fault-tolerant; you'll get asked about it in interviewsStream ProcessingPython (kafka-python)Simpler than Flink for demo; production-ready; easier to show ML integrationTime-series DBTimescaleDB (PostgreSQL extension)SQL familiarity + time-series optimizations; better than InfluxDB for relational context (equipment metadata)Object StorageMinIO (local S3)Production S3 patterns without AWS costs; stores raw data for retrainingML FrameworkPyTorch (LSTM) + scikit-learn (RF, Isolation Forest)LSTM for sequence modeling; RF for baseline; both interview-friendlyModel RegistryMLflowTracks experiments, versions models, serves via REST APIInference APIFastAPIFast, async, auto-generated docs; production standardMonitoringGrafana + StreamlitGrafana = ops teams love it; Streamlit = quick custom dashboardsOrchestrationAirflow (optional)Schedules retraining, feature backfills; shows you understand ML opsContainerizationDocker + Docker ComposeLocal dev = Compose; mention K8s for scale in README

4. Module Breakdown (Build Order)
   Phase 1: Data Foundation (Modules 1-3)

Data Simulator (data_generator/)

Simulates sensor data (vibration, temp, pressure)
Injects synthetic failures (degradation patterns)
Publishes to Kafka

Kafka Setup (infra/kafka/)

Docker Compose config
Topic creation scripts
Consumer group setup

Stream Processor (stream_processor/)

Consumes raw sensor data
Feature engineering (rolling mean, std, FFT)
Writes to TimescaleDB

Phase 2: ML Pipeline (Modules 4-6)

Feature Store (feature_engineering/)

Time-series features (lag, rolling windows)
Frequency domain features (vibration FFT peaks)
Label creation (RUL calculation from failure logs)

Training Pipeline (ml_pipeline/train/)

LSTM for RUL prediction
Random Forest for anomaly classification
Hyperparameter tuning, cross-validation
Saves to MLflow

Model Evaluation (ml_pipeline/evaluate/)

Precision/Recall for anomaly detection
MAE/RMSE for RUL prediction
Generates model card

Phase 3: Inference & Alerting (Modules 7-9)

Inference API (inference_service/)

FastAPI endpoint: /predict
Loads model from MLflow
Returns: {equipment_id, RUL_hours, anomaly_score, health_status}

Alert Engine (alerting/)

Rule-based triggers (RUL < threshold)
Sends notifications (email via SMTP, Slack webhook)
Logs alerts to DB

Real-time Dashboard (dashboard/)

Streamlit app: equipment health grid
Grafana: time-series visualizations
Shows: current RUL, anomaly trends, alert history

Phase 4: Ops & Deployment (Modules 10-12)

Monitoring & Logging (observability/)

Prometheus metrics (API latency, model inference time)
ELK stack logs (optional, or just structured logging)

Retraining Pipeline (ml_pipeline/retrain/)

Airflow DAG: weekly retraining
Drift detection (data + model performance)
A/B testing new models

Docker Compose Orchestration (docker-compose.yml)

All services defined
One command: docker-compose up
README with architecture diagram

5. Data Flow (Detailed)
   Ingestion:
   Sensor → Kafka (raw_sensor_data) → Stream Processor
   Feature Engineering:
   Stream Processor → TimescaleDB (features table) + S3 (raw backup)
   Training (Offline):
   TimescaleDB → Feature Engineering → Training Script → MLflow (model registry)
   Inference (Real-time):
   Kafka (processed_features) → Inference API (loads MLflow model) → Predictions → Alert Engine + Dashboard

6. Database Schema (TimescaleDB)
   Table: sensor_readings (hypertable, partitioned by time)
   sqlequipment_id, timestamp, vibration_x, vibration_y, vibration_z,
   temperature, pressure, power_consumption, rpm
   Table: engineered_features
   sqlequipment_id, timestamp, vibration_rms, vibration_kurtosis,
   temp_rolling_mean_1h, fft_peak_freq, ...
   Table: predictions
   sqlequipment_id, timestamp, rul_hours, anomaly_score, model_version
   Table: maintenance_logs
   sqlequipment_id, failure_timestamp, failure_type, downtime_hours

```

---

## 7. Key Interview Talking Points

1. **Why Kafka over RabbitMQ?**
   - Kafka = log-based, replayable, handles high throughput
   - Perfect for time-series data where you need to reprocess

2. **Why LSTM for RUL?**
   - Captures temporal dependencies in degradation patterns
   - Better than simple regression for sequential sensor data

3. **Why TimescaleDB over InfluxDB?**
   - Need SQL joins (sensor data + equipment metadata)
   - Better ecosystem for analytics (pg_stat_statements, PostGIS if adding location)

4. **How do you handle concept drift?**
   - Weekly retraining pipeline
   - Monitor prediction distribution vs. training distribution
   - A/B test new models before full deployment

5. **Scalability story:**
   - Current: Single machine (your laptop)
   - Next: Kafka cluster (3 brokers), TimescaleDB replica
   - Production: K8s deployment, autoscaling inference pods

---

## 8. File Structure (Final Project)
```

predictive-maintenance/
├── data_generator/ # Module 1
├── infra/
│ ├── kafka/ # Module 2
│ ├── timescaledb/
│ └── minio/
├── stream_processor/ # Module 3
├── feature_engineering/ # Module 4
├── ml_pipeline/
│ ├── train/ # Module 5
│ ├── evaluate/ # Module 6
│ └── retrain/ # Module 11
├── inference_service/ # Module 7
├── alerting/ # Module 8
├── dashboard/ # Module 9
├── observability/ # Module 10
├── docker-compose.yml # Module 12
├── README.md
└── docs/
└── ARCHITECTURE.md (this doc)
