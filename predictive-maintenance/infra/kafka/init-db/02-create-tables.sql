-- ============================================================================
-- Predictive Maintenance System — Database Initialization
-- Script 2/5: Tables & Hypertables
-- Matches db-schema.md exactly.
-- ============================================================================

-- 1. Equipment Registry
CREATE TABLE IF NOT EXISTS equipment (
    equipment_id        VARCHAR(64) PRIMARY KEY,
    equipment_type      VARCHAR(64) NOT NULL,
    model               VARCHAR(128),
    manufacturer        VARCHAR(128),
    location            VARCHAR(128),
    install_date        TIMESTAMPTZ,
    last_maintenance    TIMESTAMPTZ,
    status              VARCHAR(32) DEFAULT 'active',
    metadata            JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Sensor Readings (Hypertable)
CREATE TABLE IF NOT EXISTS sensor_readings (
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    cycle               INTEGER,
    data_source         VARCHAR(32),
    operational_settings JSONB,
    sensor_readings     JSONB NOT NULL,
    quality_flag        VARCHAR(16) DEFAULT 'ok',
    metadata            JSONB,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);

-- 3. Engineered Features (Hypertable)
CREATE TABLE IF NOT EXISTS engineered_features (
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    feature_set         VARCHAR(32) NOT NULL,
    features            JSONB NOT NULL,
    window_size         INTEGER,
    computation_time_ms FLOAT,
    source              VARCHAR(32),
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('engineered_features', 'time', if_not_exists => TRUE);

-- 4. Feature Store (Training Data — Hypertable)
CREATE TABLE IF NOT EXISTS feature_store (
    id                  BIGSERIAL,
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    feature_version     VARCHAR(16) NOT NULL,
    features            JSONB NOT NULL,
    rul                 FLOAT,
    failure_within_30   BOOLEAN,
    health_status       INTEGER,
    degradation_rate    FLOAT,
    split               VARCHAR(10),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (id, time)
);
SELECT create_hypertable('feature_store', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

-- 5. Predictions (Hypertable)
CREATE TABLE IF NOT EXISTS predictions (
    time                TIMESTAMPTZ NOT NULL,
    equipment_id        VARCHAR(64) NOT NULL,
    prediction_type     VARCHAR(32) NOT NULL,
    rul_cycles          FLOAT,
    rul_hours           FLOAT,
    confidence          FLOAT,
    confidence_lower    FLOAT,
    confidence_upper    FLOAT,
    health_status       VARCHAR(32),
    health_probabilities JSONB,
    anomaly_score       FLOAT,
    model_name          VARCHAR(64),
    model_version       VARCHAR(32),
    inference_time_ms   FLOAT,
    input_features      JSONB,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);

-- 6. Alerts
CREATE TABLE IF NOT EXISTS alerts (
    alert_id            VARCHAR(128) PRIMARY KEY,
    equipment_id        VARCHAR(64) NOT NULL,
    rule_id             VARCHAR(64) NOT NULL,
    severity            VARCHAR(16) NOT NULL,
    message             TEXT NOT NULL,
    status              VARCHAR(16) NOT NULL DEFAULT 'triggered',
    data                JSONB,
    notifications_sent  JSONB,
    triggered_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_by     VARCHAR(128),
    acknowledged_at     TIMESTAMPTZ,
    resolved_at         TIMESTAMPTZ,
    suppressed          BOOLEAN DEFAULT FALSE,
    suppress_reason     VARCHAR(256)
);

-- 7. Alert Rules Configuration
CREATE TABLE IF NOT EXISTS alert_rules (
    rule_id             VARCHAR(64) PRIMARY KEY,
    name                VARCHAR(128) NOT NULL,
    description         TEXT,
    condition           TEXT NOT NULL,
    severity            VARCHAR(16) NOT NULL,
    notification_channels JSONB,
    cooldown_seconds    INTEGER DEFAULT 300,
    enabled             BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 8. Maintenance Logs
CREATE TABLE IF NOT EXISTS maintenance_logs (
    log_id              BIGSERIAL PRIMARY KEY,
    equipment_id        VARCHAR(64) NOT NULL REFERENCES equipment(equipment_id),
    maintenance_type    VARCHAR(32) NOT NULL,
    description         TEXT,
    triggered_by        VARCHAR(128),
    cost                DECIMAL(12,2),
    downtime_hours      FLOAT,
    parts_replaced      JSONB,
    performed_by        VARCHAR(128),
    started_at          TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    outcome             VARCHAR(32),
    notes               TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 9. Model Registry (supplementary to MLflow)
CREATE TABLE IF NOT EXISTS model_registry (
    model_name          VARCHAR(64) NOT NULL,
    model_version       VARCHAR(32) NOT NULL,
    model_type          VARCHAR(32) NOT NULL,
    stage               VARCHAR(32) NOT NULL,
    run_id              VARCHAR(128),
    artifact_uri        TEXT,
    metrics             JSONB,
    parameters          JSONB,
    promoted_at         TIMESTAMPTZ,
    promoted_by         VARCHAR(128),
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (model_name, model_version)
);

-- 10. Training Runs
CREATE TABLE IF NOT EXISTS training_runs (
    run_id              VARCHAR(128) PRIMARY KEY,
    model_name          VARCHAR(64) NOT NULL,
    model_version       VARCHAR(32),
    status              VARCHAR(16) NOT NULL,
    trigger             VARCHAR(32),
    config              JSONB,
    metrics             JSONB,
    training_data_stats JSONB,
    started_at          TIMESTAMPTZ NOT NULL,
    completed_at        TIMESTAMPTZ,
    error_message       TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 11. Drift Detection Logs
CREATE TABLE IF NOT EXISTS drift_logs (
    id                  BIGSERIAL PRIMARY KEY,
    check_time          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drift_type          VARCHAR(32) NOT NULL,
    detected            BOOLEAN NOT NULL,
    details             JSONB,
    action_taken        VARCHAR(64),
    reference_window    TSTZRANGE,
    detection_window    TSTZRANGE
);

DO $$ BEGIN RAISE NOTICE '02-create-tables: All tables and hypertables created'; END $$;
