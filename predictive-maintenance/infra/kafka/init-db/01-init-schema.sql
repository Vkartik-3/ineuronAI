-- TimescaleDB Initialization Script
-- Creates tables and hypertables for predictive maintenance system

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Sensor Readings Table (Time-series data)
CREATE TABLE IF NOT EXISTS sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    equipment_id VARCHAR(50) NOT NULL,
    equipment_type VARCHAR(50) NOT NULL,
    cycle INTEGER NOT NULL,

    -- Operational settings
    operational_setting_1 DOUBLE PRECISION,
    operational_setting_2 DOUBLE PRECISION,
    operational_setting_3 DOUBLE PRECISION,

    -- Sensor measurements (dynamic columns based on equipment type)
    sensor_data JSONB NOT NULL,

    -- Metadata
    location VARCHAR(100),
    model VARCHAR(50),
    failure_mode VARCHAR(100),
    rul_remaining INTEGER,
    degradation_stage VARCHAR(20),
    is_degraded BOOLEAN DEFAULT FALSE,

    PRIMARY KEY (time, equipment_id)
);

-- Convert to hypertable (time-series optimized)
SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_equipment_id ON sensor_readings (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_equipment_type ON sensor_readings (equipment_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_degradation ON sensor_readings (is_degraded, time DESC) WHERE is_degraded = TRUE;
CREATE INDEX IF NOT EXISTS idx_failure_mode ON sensor_readings (failure_mode, time DESC) WHERE failure_mode IS NOT NULL;

-- Processed Features Table
CREATE TABLE IF NOT EXISTS processed_features (
    time TIMESTAMPTZ NOT NULL,
    equipment_id VARCHAR(50) NOT NULL,
    cycle INTEGER NOT NULL,

    -- Time-domain features
    features JSONB NOT NULL,

    -- Frequency-domain features
    fft_features JSONB,

    -- Rolling statistics
    rolling_features JSONB,

    PRIMARY KEY (time, equipment_id)
);

SELECT create_hypertable('processed_features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_features_equipment ON processed_features (equipment_id, time DESC);

-- Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    time TIMESTAMPTZ NOT NULL,
    equipment_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,

    -- Predictions
    predicted_rul INTEGER,
    anomaly_score DOUBLE PRECISION,
    health_status VARCHAR(20),
    failure_probability DOUBLE PRECISION,

    -- Model info
    model_type VARCHAR(50),
    confidence DOUBLE PRECISION,

    PRIMARY KEY (time, equipment_id)
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_predictions_equipment ON predictions (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_health ON predictions (health_status, time DESC);

-- Maintenance Alerts Table
CREATE TABLE IF NOT EXISTS maintenance_alerts (
    alert_id SERIAL,
    time TIMESTAMPTZ NOT NULL,
    equipment_id VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT,
    rul_remaining INTEGER,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(100),
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,

    PRIMARY KEY (alert_id, time)
);

SELECT create_hypertable('maintenance_alerts', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_alerts_equipment ON maintenance_alerts (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unresolved ON maintenance_alerts (resolved, time DESC) WHERE resolved = FALSE;

-- Equipment Metadata Table (Regular table, not time-series)
CREATE TABLE IF NOT EXISTS equipment_metadata (
    equipment_id VARCHAR(50) PRIMARY KEY,
    equipment_type VARCHAR(50) NOT NULL,
    model VARCHAR(50),
    manufacturer VARCHAR(100),
    install_date DATE,
    location VARCHAR(100),
    status VARCHAR(20) DEFAULT 'active',
    last_maintenance DATE,
    total_cycles INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model Performance Metrics Table
CREATE TABLE IF NOT EXISTS model_metrics (
    time TIMESTAMPTZ NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    equipment_type VARCHAR(50),

    PRIMARY KEY (time, model_version, metric_name)
);

SELECT create_hypertable('model_metrics', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_metrics_model ON model_metrics (model_version, time DESC);

-- Continuous Aggregates for fast queries
-- 1. Hourly aggregates for sensor readings
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    equipment_id,
    equipment_type,
    COUNT(*) as reading_count,
    AVG((sensor_data->>'temperature')::DOUBLE PRECISION) as avg_temperature,
    AVG((sensor_data->>'vibration')::DOUBLE PRECISION) as avg_vibration,
    AVG((sensor_data->>'pressure')::DOUBLE PRECISION) as avg_pressure,
    MAX((sensor_data->>'temperature')::DOUBLE PRECISION) as max_temperature,
    MAX((sensor_data->>'vibration')::DOUBLE PRECISION) as max_vibration
FROM sensor_readings
GROUP BY bucket, equipment_id, equipment_type;

-- 2. Daily equipment health summary
CREATE MATERIALIZED VIEW IF NOT EXISTS equipment_health_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS bucket,
    equipment_id,
    COUNT(*) as predictions_count,
    AVG(predicted_rul) as avg_rul,
    MIN(predicted_rul) as min_rul,
    AVG(anomaly_score) as avg_anomaly_score,
    MAX(anomaly_score) as max_anomaly_score
FROM predictions
GROUP BY bucket, equipment_id;

-- Data retention policies (keep raw data for 30 days, aggregates for 1 year)
SELECT add_retention_policy('sensor_readings', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('processed_features', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('predictions', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('maintenance_alerts', INTERVAL '180 days', if_not_exists => TRUE);
SELECT add_retention_policy('model_metrics', INTERVAL '180 days', if_not_exists => TRUE);

-- Compression policies (compress data older than 7 days)
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id'
);

ALTER TABLE processed_features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id'
);

ALTER TABLE predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id'
);

SELECT add_compression_policy('sensor_readings', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('processed_features', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('predictions', INTERVAL '7 days', if_not_exists => TRUE);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pmuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pmuser;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'TimescaleDB schema initialized successfully!';
    RAISE NOTICE 'Tables created: sensor_readings, processed_features, predictions, maintenance_alerts';
    RAISE NOTICE 'Continuous aggregates: sensor_readings_hourly, equipment_health_daily';
    RAISE NOTICE 'Retention and compression policies applied';
END $$;
