# Database Schema — Predictive Maintenance System

> **Derived from**: PRD + System Design (not existing code)
> **Database**: TimescaleDB (PostgreSQL with time-series extensions)
> **Version**: 1.0.0
> **Last Updated**: 2026-03-11

---

## Overview

The data model supports the full predictive maintenance lifecycle:

1. **Raw sensor ingestion** — high-frequency time-series storage
2. **Feature engineering** — pre-computed features for ML
3. **Model management** — training runs, metrics, versions
4. **Predictions** — RUL and health classification results
5. **Alerts** — rule-based alert lifecycle
6. **Maintenance** — maintenance actions and outcomes
7. **Equipment registry** — metadata for all monitored equipment

---

## 1. Equipment Registry

```sql
CREATE TABLE equipment (
    equipment_id        VARCHAR(64) PRIMARY KEY,
    equipment_type      VARCHAR(64) NOT NULL,    -- 'turbofan_engine', 'pump', 'compressor'
    model               VARCHAR(128),
    manufacturer        VARCHAR(128),
    location            VARCHAR(128),
    install_date        TIMESTAMPTZ,
    last_maintenance    TIMESTAMPTZ,
    status              VARCHAR(32) DEFAULT 'active',  -- 'active', 'maintenance', 'decommissioned'
    metadata            JSONB,                  -- flexible extra fields
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equipment_type ON equipment(equipment_type);
CREATE INDEX idx_equipment_status ON equipment(status);
CREATE INDEX idx_equipment_location ON equipment(location);
```

---

## 2. Sensor Readings (Hypertable)

> Primary time-series table. Converted to a TimescaleDB hypertable for automatic time-based partitioning.

```sql
CREATE TABLE sensor_readings (
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    cycle               INTEGER,
    data_source         VARCHAR(32),             -- 'cmapss', 'synthetic', 'live'
    operational_settings JSONB,                   -- { op_setting_1, op_setting_2, op_setting_3 }
    sensor_readings     JSONB NOT NULL,           -- { sensor_1: 518.67, sensor_2: 643.5, ... }
    quality_flag        VARCHAR(16) DEFAULT 'ok', -- 'ok', 'missing', 'outlier', 'interpolated'
    metadata            JSONB,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('sensor_readings', 'time');

CREATE INDEX idx_sensor_equipment ON sensor_readings (equipment_id, time DESC);
CREATE INDEX idx_sensor_readings_gin ON sensor_readings USING GIN (sensor_readings);

-- Retention policy: keep raw data for 90 days
SELECT add_retention_policy('sensor_readings', INTERVAL '90 days');

-- Continuous aggregate: 1-minute averages
CREATE MATERIALIZED VIEW sensor_readings_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    equipment_id,
    AVG((sensor_readings->>'sensor_2')::float) AS avg_temperature_2,
    AVG((sensor_readings->>'sensor_3')::float) AS avg_temperature_3,
    AVG((sensor_readings->>'sensor_4')::float) AS avg_pressure_4,
    MAX((sensor_readings->>'sensor_2')::float) AS max_temperature_2,
    MIN((sensor_readings->>'sensor_2')::float) AS min_temperature_2,
    COUNT(*) AS sample_count
FROM sensor_readings
GROUP BY bucket, equipment_id;

-- Continuous aggregate: 1-hour averages
CREATE MATERIALIZED VIEW sensor_readings_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    equipment_id,
    AVG((sensor_readings->>'sensor_2')::float) AS avg_temperature_2,
    AVG((sensor_readings->>'sensor_3')::float) AS avg_temperature_3,
    MAX((sensor_readings->>'sensor_2')::float) AS max_temperature_2,
    COUNT(*) AS sample_count
FROM sensor_readings
GROUP BY bucket, equipment_id;
```

---

## 3. Engineered Features (Hypertable)

> Stores pre-computed time-domain and frequency-domain features produced by the stream processor and feature store.

```sql
CREATE TABLE engineered_features (
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    feature_set         VARCHAR(32) NOT NULL,    -- 'time_domain', 'frequency_domain', 'combined'
    features            JSONB NOT NULL,          -- key-value pairs of feature names to values
    window_size         INTEGER,                 -- rolling window size used
    computation_time_ms FLOAT,
    source              VARCHAR(32),             -- 'stream_processor', 'feature_store'
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('engineered_features', 'time');

CREATE INDEX idx_features_equipment ON engineered_features (equipment_id, time DESC);
CREATE INDEX idx_features_set ON engineered_features (feature_set);
CREATE INDEX idx_features_gin ON engineered_features USING GIN (features);
```

---

## 4. Feature Store (Training Data)

> Stores versioned feature snapshots used for model training and evaluation.

```sql
CREATE TABLE feature_store (
    id                  BIGSERIAL,
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    feature_version     VARCHAR(16) NOT NULL,    -- 'v1', 'v2', etc.
    features            JSONB NOT NULL,

    -- Labels (generated by LabelGenerator)
    rul                 FLOAT,                   -- remaining useful life (cycles)
    failure_within_30   BOOLEAN,                 -- binary: will fail in 30 cycles?
    health_status       INTEGER,                 -- 0=healthy, 1=warning, 2=critical, 3=imminent_failure
    degradation_rate    FLOAT,

    -- Split assignment
    split               VARCHAR(10),             -- 'train', 'val', 'test'

    created_at          TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (id)
);

SELECT create_hypertable('feature_store', 'time');

CREATE INDEX idx_fstore_equipment ON feature_store (equipment_id, time DESC);
CREATE INDEX idx_fstore_version ON feature_store (feature_version);
CREATE INDEX idx_fstore_split ON feature_store (split);
```

---

## 5. Predictions (Hypertable)

> Stores every prediction made by the inference service for auditing and trend analysis.

```sql
CREATE TABLE predictions (
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    prediction_type     VARCHAR(32) NOT NULL,    -- 'rul', 'health_classification'

    -- RUL fields
    rul_cycles          FLOAT,
    rul_hours           FLOAT,
    confidence          FLOAT,
    confidence_lower    FLOAT,
    confidence_upper    FLOAT,

    -- Health classification fields
    health_status       VARCHAR(32),             -- 'healthy', 'warning', 'critical', 'imminent_failure'
    health_probabilities JSONB,                  -- { healthy: 0.15, warning: 0.62, ... }

    -- Common fields
    anomaly_score       FLOAT,
    model_name          VARCHAR(64),
    model_version       VARCHAR(32),
    inference_time_ms   FLOAT,
    input_features      JSONB,                   -- snapshot of input features (optional — for debugging)
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('predictions', 'time');

CREATE INDEX idx_pred_equipment ON predictions (equipment_id, time DESC);
CREATE INDEX idx_pred_type ON predictions (prediction_type);
CREATE INDEX idx_pred_model ON predictions (model_name, model_version);
CREATE INDEX idx_pred_health ON predictions (health_status);
```

---

## 6. Alerts

> Stores all alerts generated by the alert engine throughout their lifecycle.

```sql
CREATE TABLE alerts (
    alert_id            VARCHAR(128) PRIMARY KEY,
    equipment_id        VARCHAR(64) NOT NULL,
    rule_id             VARCHAR(64) NOT NULL,
    severity            VARCHAR(16) NOT NULL,    -- 'info', 'warning', 'critical'
    message             TEXT NOT NULL,
    status              VARCHAR(16) NOT NULL DEFAULT 'triggered',
                                                 -- 'triggered', 'acknowledged', 'resolved'
    data                JSONB,                   -- prediction data that triggered the alert
    notifications_sent  JSONB,                   -- list of channels notified

    triggered_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_by     VARCHAR(128),
    acknowledged_at     TIMESTAMPTZ,
    resolved_at         TIMESTAMPTZ,

    suppressed          BOOLEAN DEFAULT FALSE,
    suppress_reason     VARCHAR(256)
);

CREATE INDEX idx_alerts_equipment ON alerts (equipment_id);
CREATE INDEX idx_alerts_severity ON alerts (severity);
CREATE INDEX idx_alerts_status ON alerts (status);
CREATE INDEX idx_alerts_triggered ON alerts (triggered_at DESC);
CREATE INDEX idx_alerts_rule ON alerts (rule_id);
```

---

## 7. Alert Rules Configuration

> Stores alert rule definitions (can also be loaded from YAML config).

```sql
CREATE TABLE alert_rules (
    rule_id             VARCHAR(64) PRIMARY KEY,
    name                VARCHAR(128) NOT NULL,
    description         TEXT,
    condition           TEXT NOT NULL,            -- e.g. "rul < 10"
    severity            VARCHAR(16) NOT NULL,
    notification_channels JSONB,                  -- ["email", "slack"]
    cooldown_seconds    INTEGER DEFAULT 300,
    enabled             BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 8. Maintenance Logs

> Records maintenance actions taken on equipment — links to alerts that prompted the maintenance.

```sql
CREATE TABLE maintenance_logs (
    log_id              BIGSERIAL PRIMARY KEY,
    equipment_id        VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id),
    maintenance_type    VARCHAR(32) NOT NULL,     -- 'preventive', 'corrective', 'emergency'
    description         TEXT,
    triggered_by        VARCHAR(128),             -- alert_id or 'scheduled' or 'manual'
    cost                DECIMAL(12,2),
    downtime_hours      FLOAT,
    parts_replaced      JSONB,                    -- [{ name, part_number, quantity }]
    performed_by        VARCHAR(128),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    outcome             VARCHAR(32),              -- 'resolved', 'partial', 'escalated'
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_maint_equipment ON maintenance_logs (equipment_id);
CREATE INDEX idx_maint_type ON maintenance_logs (maintenance_type);
CREATE INDEX idx_maint_time ON maintenance_logs (started_at DESC);
```

---

## 9. Model Registry (Supplementary to MLflow)

> Local tracking table for model versions deployed to production. MLflow is source of truth, but this allows DB-level queries.

```sql
CREATE TABLE model_registry (
    model_name          VARCHAR(64) NOT NULL,
    model_version       VARCHAR(32) NOT NULL,
    model_type          VARCHAR(32) NOT NULL,     -- 'lstm', 'random_forest'
    stage               VARCHAR(32) NOT NULL,     -- 'staging', 'production', 'archived'
    run_id              VARCHAR(128),             -- MLflow run ID
    artifact_uri        TEXT,                     -- MLflow artifact URI (S3/MinIO)
    metrics             JSONB,                    -- { mae: 12.5, rmse: 18.3 }
    parameters          JSONB,                    -- { epochs: 100, learning_rate: 0.001 }
    promoted_at         TIMESTAMPTZ,
    promoted_by         VARCHAR(128),
    created_at          TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (model_name, model_version)
);

CREATE INDEX idx_model_stage ON model_registry (stage);
```

---

## 10. Training Runs

> Tracks individual training pipeline runs (supplements MLflow).

```sql
CREATE TABLE training_runs (
    run_id              VARCHAR(128) PRIMARY KEY,
    model_name          VARCHAR(64) NOT NULL,
    model_version       VARCHAR(32),
    status              VARCHAR(16) NOT NULL,     -- 'running', 'completed', 'failed'
    trigger             VARCHAR(32),              -- 'manual', 'scheduled', 'drift_detected'
    config              JSONB,
    metrics             JSONB,
    training_data_stats JSONB,                    -- { num_samples, time_range, equipment_ids }
    started_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ,
    error_message       TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_training_model ON training_runs (model_name);
CREATE INDEX idx_training_status ON training_runs (status);
```

---

## 11. Drift Detection Logs

> Records drift detection results over time.

```sql
CREATE TABLE drift_logs (
    id                  BIGSERIAL PRIMARY KEY,
    check_time          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drift_type          VARCHAR(32) NOT NULL,     -- 'data_drift', 'concept_drift'
    detected            BOOLEAN NOT NULL,
    details             JSONB,                    -- { feature: "temp_3", ks_statistic: 0.15, p_value: 0.02 }
    action_taken        VARCHAR(64),              -- 'none', 'retraining_triggered', 'alert_sent'
    reference_window    TSTZRANGE,
    detection_window    TSTZRANGE
);

CREATE INDEX idx_drift_time ON drift_logs (check_time DESC);
CREATE INDEX idx_drift_type ON drift_logs (drift_type);
```

---

## Entity Relationship Diagram (Text)

```
equipment ──────────┬──< sensor_readings       (1:N, hypertable)
                    ├──< engineered_features   (1:N, hypertable)
                    ├──< feature_store         (1:N, hypertable)
                    ├──< predictions           (1:N, hypertable)
                    ├──< alerts                (1:N)
                    └──< maintenance_logs      (1:N)

alerts ─── triggered_by ──> maintenance_logs   (optional FK)

model_registry ──< training_runs              (1:N via model_name)
model_registry ──< predictions                (1:N via model_name + version)

drift_logs ──> training_runs                  (action_taken = retraining_triggered)
```

---

## TimescaleDB-Specific Configuration

### Hypertables

| Table                 | Partition Column | Chunk Interval |
| --------------------- | ---------------- | -------------- |
| `sensor_readings`     | `time`           | 1 day          |
| `engineered_features` | `time`           | 1 day          |
| `feature_store`       | `time`           | 7 days         |
| `predictions`         | `time`           | 1 day          |

### Retention Policies

| Table                 | Retention Period | Rationale                                          |
| --------------------- | ---------------- | -------------------------------------------------- |
| `sensor_readings`     | 90 days          | Raw data is high volume; older data archived to S3 |
| `engineered_features` | 180 days         | Features needed for retraining comparison          |
| `predictions`         | 365 days         | Needed for backtesting and trend analysis          |
| `alerts`              | 365 days         | Compliance and root cause analysis                 |

### Continuous Aggregates

| View                 | Source            | Bucket   | Purpose                  |
| -------------------- | ----------------- | -------- | ------------------------ |
| `sensor_readings_1m` | `sensor_readings` | 1 minute | Dashboard real-time view |
| `sensor_readings_1h` | `sensor_readings` | 1 hour   | Dashboard trend view     |

---

## Assumptions

1. **JSONB for sensor readings** — The number of sensors varies by equipment type (21 for turbofan, 5 for pump). JSONB provides flexibility. Alternatively, a normalized sensor table could be used but would increase join complexity.
2. **MLflow is the primary model registry** — The `model_registry` table is a supplementary mirror for SQL-level queries. MLflow (backed by MinIO/S3) is the authoritative source.
3. **Equipment-to-sensor mapping** is implicit in the sensor_readings JSONB. A separate `equipment_sensors` table could normalize this but is not required at current scale.
4. **No multi-tenancy** — Single-factory deployment assumed. Tenant isolation would require `tenant_id` columns.
5. **TimescaleDB chunk intervals** are initial recommendations; should be tuned based on actual data volume.
