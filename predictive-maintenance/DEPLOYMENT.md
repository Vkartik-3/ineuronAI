# Deployment Guide

Production deployment guide for the Predictive Maintenance System.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Infrastructure Setup](#infrastructure-setup)
- [Application Deployment](#application-deployment)
- [Configuration](#configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Backup & Recovery](#backup--recovery)
- [Scaling](#scaling)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum (Development):**

- 4 CPU cores
- 8GB RAM
- 50GB storage
- Docker 24.0+
- Docker Compose 2.0+

**Recommended (Production):**

- 8+ CPU cores
- 32GB RAM
- 200GB SSD storage
- Kubernetes 1.28+ (for production scale)
- Load balancer (Nginx/HAProxy)

### Software Dependencies

- Python 3.9+
- Docker & Docker Compose
- PostgreSQL client (psql)
- Git

## Infrastructure Setup

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd predictive-maintenance
```

### Step 2: Configure Environment

Create `.env` file in project root:

```bash
# Infrastructure
POSTGRES_USER=pmuser
POSTGRES_PASSWORD=<secure-password>
POSTGRES_DB=predictive_maintenance

REDIS_PASSWORD=<redis-password>

MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=<minio-password>

# Application
MLFLOW_TRACKING_URI=http://mlflow:5000
MODEL_NAME=predictive_maintenance_model
MODEL_STAGE=Production

# Alerts
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<email>
SMTP_PASSWORD=<app-password>
SLACK_WEBHOOK_URL=<slack-webhook>

# API Keys
INFERENCE_API_KEY=<secure-api-key>
```

### Step 3: Start Infrastructure Services

```bash
cd infra/kafka
docker-compose up -d

# Wait for services to be healthy
./scripts/health-check.sh
```

Services started:

- ✅ Kafka + Zookeeper
- ✅ TimescaleDB
- ✅ MinIO (S3)
- ✅ Redis
- ✅ Kafdrop (monitoring)

### Step 4: Initialize Database

```bash
# Connect to TimescaleDB
docker exec -it pm-timescaledb psql -U pmuser -d predictive_maintenance

# Verify schema
\dt

# Check hypertables
SELECT * FROM timescaledb_information.hypertables;
```

Expected tables:

- `sensor_readings` (hypertable)
- `sensor_features` (hypertable)
- `predictions`
- `alerts`
- `equipment`

### Step 5: Create MinIO Buckets

```bash
# Access MinIO console at http://localhost:9001
# Login: minioadmin / <minio-password>

# Create buckets:
# - raw-data (for raw sensor backups)
# - mlflow-artifacts (for model storage)
# - model-registry (for model versions)
```

Or via CLI:

```bash
docker exec pm-minio mc alias set local http://localhost:9000 minioadmin <minio-password>
docker exec pm-minio mc mb local/raw-data
docker exec pm-minio mc mb local/mlflow-artifacts
docker exec pm-minio mc mb local/model-registry
```

## Application Deployment

### Step 1: Build Docker Images

```bash
cd <project-root>

# Build inference service
docker build -t pm-inference-api:latest ./inference_service

# Build alert engine
docker build -t pm-alert-engine:latest ./alerting

# Build dashboard
docker build -t pm-dashboard:latest ./dashboard/streamlit_app
```

### Step 2: Start Application Services

```bash
# From project root
docker-compose up -d

# Check service status
docker-compose ps
```

Services:

- Inference API: http://localhost:8000
- Dashboard: http://localhost:8501
- Grafana: http://localhost:3000
- MLflow: http://localhost:5000

### Step 3: Run Data Generator

```bash
cd data_generator
pip install -r requirements.txt

# Start generating sensor data
python main.py --num-equipment 10 --equipment-type turbofan_engine
```

### Step 4: Start Stream Processor

```bash
cd stream_processor
pip install -r requirements.txt

# Process streaming data
python main.py
```

### Step 5: Train Initial Models

```bash
cd ml_pipeline/train
pip install -r requirements.txt

# Train models
python train_pipeline.py --config config/training_config.yaml

# Verify in MLflow
# http://localhost:5000
```

### Step 6: Promote Model to Production

```bash
# Via MLflow UI or Python:
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="predictive_maintenance_model",
    version="1",
    stage="Production"
)
```

## Configuration

### Inference API Configuration

Edit `inference_service/config/inference_config.yaml`:

```yaml
model:
  name: "predictive_maintenance_model"
  stage: "Production"
  update_interval_seconds: 300

prediction:
  max_batch_size: 100
  timeout_seconds: 30

thresholds:
  rul_warning: 72 # hours
  rul_critical: 24
  health_warning: 0.7
  health_critical: 0.4
```

### Alert Configuration

Edit `alerting/config/alert_config.yaml`:

```yaml
engine:
  check_interval_seconds: 60
  batch_size: 100

notifiers:
  email:
    enabled: true
    smtp_server: "smtp.gmail.com"
    smtp_port: 587

  slack:
    enabled: true
    webhook_url: "${SLACK_WEBHOOK_URL}"

  webhook:
    enabled: true
    url: "https://api.company.com/webhooks/alerts"
```

### Stream Processor Configuration

Edit `stream_processor/config/processor_config.yaml`:

```yaml
kafka:
  bootstrap_servers: "localhost:9092"
  consumer_group: "stream_processor"
  topics:
    - "raw_sensor_data"

processing:
  buffer_size: 1000
  write_interval_seconds: 10

feature_engineering:
  rolling_windows: [10, 30, 60] # seconds
  fft_enabled: true
```

## Monitoring & Observability

### Grafana Dashboards

Access: http://localhost:3000 (admin/admin)

Pre-configured dashboards:

1. **Equipment Health Overview**

   - Real-time health scores
   - RUL predictions
   - Alert status

2. **System Performance**

   - API latency
   - Kafka throughput
   - Database connections

3. **Model Performance**
   - Prediction accuracy
   - Drift metrics
   - Error rates

### MLflow Tracking

Access: http://localhost:5000

Features:

- Experiment tracking
- Model versioning
- Artifact storage
- Model registry

### Application Logs

```bash
# Inference API logs
docker-compose logs -f inference-api

# Alert engine logs
docker-compose logs -f alert-engine

# Dashboard logs
docker-compose logs -f dashboard

# All services
docker-compose logs -f
```

### Health Checks

```bash
# Infrastructure health
cd infra/kafka
./scripts/health-check.sh

# API health
curl http://localhost:8000/health

# Check all endpoints
curl http://localhost:8000/health/detailed
```

## Backup & Recovery

### Database Backup

```bash
# Backup TimescaleDB
docker exec pm-timescaledb pg_dump -U pmuser predictive_maintenance > backup_$(date +%Y%m%d).sql

# Automated daily backup
0 2 * * * docker exec pm-timescaledb pg_dump -U pmuser predictive_maintenance > /backups/pm_$(date +\%Y\%m\%d).sql
```

### Model Backup

```bash
# Models are stored in MinIO (mlflow-artifacts bucket)
# Configure S3 sync for redundancy

# Backup to external S3
aws s3 sync s3://mlflow-artifacts s3://backup-bucket/mlflow-artifacts --source-region local
```

### Configuration Backup

```bash
# Backup all configuration files
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  inference_service/config \
  alerting/config \
  stream_processor/config \
  ml_pipeline/train/config \
  ml_pipeline/retrain/config
```

### Recovery Procedure

```bash
# 1. Restore database
docker exec -i pm-timescaledb psql -U pmuser predictive_maintenance < backup_20240115.sql

# 2. Restart services
docker-compose restart

# 3. Verify model availability
curl http://localhost:8000/health/model

# 4. Check recent predictions
curl http://localhost:8000/predictions/recent
```

## Scaling

### Horizontal Scaling

#### Scale Inference API

```bash
# Using Docker Compose
docker-compose up -d --scale inference-api=3

# With load balancer (Nginx)
# See nginx.conf for configuration
```

#### Scale Alert Engine

```bash
# Multiple alert engine instances
docker-compose up -d --scale alert-engine=2
```

#### Kafka Partitioning

```bash
# Increase topic partitions
docker exec pm-kafka kafka-topics --bootstrap-server localhost:9092 \
  --alter --topic raw_sensor_data --partitions 10
```

### Vertical Scaling

Edit `docker-compose.yml`:

```yaml
inference-api:
  deploy:
    resources:
      limits:
        cpus: "2.0"
        memory: 4G
      reservations:
        cpus: "1.0"
        memory: 2G
```

### Kubernetes Deployment

For production scale, deploy to Kubernetes:

```bash
# Example deployment
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmaps.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/
kubectl apply -f k8s/ingress.yaml
```

## Security

### Network Security

```bash
# Use Docker networks for isolation
# Only expose necessary ports
# Configure firewall rules

# Example iptables rules
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

### Secrets Management

```bash
# Use Docker secrets (Swarm mode)
echo "my-secret-password" | docker secret create db_password -

# Or use HashiCorp Vault
vault kv put secret/predictive-maintenance \
  postgres_password=<password> \
  api_key=<key>
```

### API Authentication

Enable API key authentication in `inference_service/api/main.py`:

```python
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if api_key != os.getenv("INFERENCE_API_KEY"):
        return Response(status_code=403)
    return await call_next(request)
```

### SSL/TLS Configuration

```bash
# Generate certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem -out nginx/ssl/cert.pem

# Configure Nginx with SSL
# See nginx/nginx.conf
```

## Troubleshooting

### Common Issues

#### 1. Inference API Not Loading Model

**Symptoms:**

```
ERROR: Failed to load model from MLflow
```

**Solution:**

```bash
# Check MLflow connectivity
curl http://localhost:5000/api/2.0/mlflow/registered-models/get?name=predictive_maintenance_model

# Verify model exists in Production stage
python -c "
import mlflow
client = mlflow.tracking.MlflowClient()
versions = client.get_latest_versions('predictive_maintenance_model', stages=['Production'])
print(versions)
"

# Restart inference service
docker-compose restart inference-api
```

#### 2. TimescaleDB Connection Issues

**Symptoms:**

```
psycopg2.OperationalError: could not connect to server
```

**Solution:**

```bash
# Check TimescaleDB is running
docker ps | grep timescaledb

# Check logs
docker logs pm-timescaledb

# Verify connection
docker exec -it pm-timescaledb psql -U pmuser -d predictive_maintenance -c "SELECT version();"

# Restart database
docker-compose restart timescaledb
```

#### 3. Kafka Consumer Lag

**Symptoms:**

- Stream processor falling behind
- Increasing consumer lag

**Solution:**

```bash
# Check consumer lag
docker exec pm-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group stream_processor

# Scale stream processor
# Or increase partition count
docker exec pm-kafka kafka-topics --bootstrap-server localhost:9092 \
  --alter --topic raw_sensor_data --partitions 10
```

#### 4. Alert Notifications Not Sending

**Symptoms:**

- Alerts created but not delivered

**Solution:**

```bash
# Check alert engine logs
docker logs pm-alert-engine | grep ERROR

# Verify SMTP settings
python -c "
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('user@gmail.com', 'password')
server.quit()
print('SMTP OK')
"

# Test Slack webhook
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{"text":"Test alert"}'
```

### Performance Tuning

#### Database Optimization

```sql
-- Add indexes
CREATE INDEX idx_sensor_readings_time ON sensor_readings(timestamp DESC);
CREATE INDEX idx_predictions_equipment ON predictions(equipment_id, timestamp DESC);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM sensor_features WHERE timestamp > NOW() - INTERVAL '1 day';

-- Optimize queries
SET work_mem = '256MB';
```

#### Kafka Tuning

```yaml
# Increase buffer sizes
kafka:
  producer:
    buffer_memory: 67108864 # 64MB
    batch_size: 32768
    linger_ms: 10

  consumer:
    fetch_min_bytes: 10240
    fetch_max_wait_ms: 500
```

#### Model Inference Optimization

```python
# Use model caching
# Batch predictions
# Use async endpoints

# Example: Batch prediction optimization
@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    # Load model once
    model = model_manager.get_model()

    # Prepare batch
    features_batch = [req.features for req in requests]

    # Predict in batch (faster than individual predictions)
    predictions = model.predict(features_batch)

    return predictions
```

## Production Checklist

- [ ] All services running and healthy
- [ ] Database backups configured (daily)
- [ ] Model artifacts backed up to S3
- [ ] SSL/TLS certificates installed
- [ ] API authentication enabled
- [ ] Monitoring dashboards configured
- [ ] Alert notifications tested
- [ ] Log aggregation setup (e.g., ELK stack)
- [ ] Resource limits configured
- [ ] Auto-scaling policies defined
- [ ] Disaster recovery plan documented
- [ ] Security audit completed

## Support

For issues and questions:

- Check logs: `docker-compose logs -f`
- Review module READMEs
- Check GitHub issues
- Contact: ml-team@company.com
