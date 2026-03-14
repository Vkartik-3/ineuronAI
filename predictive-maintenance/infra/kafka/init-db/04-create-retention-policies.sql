-- ============================================================================
-- Predictive Maintenance System — Database Initialization
-- Script 4/5: Retention Policies & Continuous Aggregates
-- ============================================================================

-- Retention policies
SELECT add_retention_policy('sensor_readings',     INTERVAL '90 days',  if_not_exists => TRUE);
SELECT add_retention_policy('engineered_features',  INTERVAL '180 days', if_not_exists => TRUE);
SELECT add_retention_policy('predictions',          INTERVAL '365 days', if_not_exists => TRUE);

-- Continuous aggregate: 1-minute sensor averages
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    equipment_id,
    AVG((sensor_readings->>'sensor_2')::float)  AS avg_temperature_2,
    AVG((sensor_readings->>'sensor_3')::float)  AS avg_temperature_3,
    AVG((sensor_readings->>'sensor_4')::float)  AS avg_pressure_4,
    MAX((sensor_readings->>'sensor_2')::float)  AS max_temperature_2,
    MIN((sensor_readings->>'sensor_2')::float)  AS min_temperature_2,
    COUNT(*) AS sample_count
FROM sensor_readings
GROUP BY bucket, equipment_id;

-- Continuous aggregate: 1-hour sensor averages
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_readings_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS bucket,
    equipment_id,
    AVG((sensor_readings->>'sensor_2')::float)  AS avg_temperature_2,
    AVG((sensor_readings->>'sensor_3')::float)  AS avg_temperature_3,
    MAX((sensor_readings->>'sensor_2')::float)  AS max_temperature_2,
    COUNT(*) AS sample_count
FROM sensor_readings
GROUP BY bucket, equipment_id;

-- Compression policies (compress chunks older than 7 days)
ALTER TABLE sensor_readings SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id'
);
SELECT add_compression_policy('sensor_readings', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE engineered_features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id'
);
SELECT add_compression_policy('engineered_features', INTERVAL '7 days', if_not_exists => TRUE);

ALTER TABLE predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'equipment_id'
);
SELECT add_compression_policy('predictions', INTERVAL '7 days', if_not_exists => TRUE);

DO $$ BEGIN RAISE NOTICE '04-retention-policies: Retention, continuous aggregates, and compression configured'; END $$;
