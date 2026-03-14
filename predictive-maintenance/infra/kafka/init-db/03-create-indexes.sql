-- ============================================================================
-- Predictive Maintenance System — Database Initialization
-- Script 3/5: Indexes
-- ============================================================================

-- Equipment
CREATE INDEX IF NOT EXISTS idx_equipment_type ON equipment(equipment_type);
CREATE INDEX IF NOT EXISTS idx_equipment_status ON equipment(status);
CREATE INDEX IF NOT EXISTS idx_equipment_location ON equipment(location);

-- Sensor Readings
CREATE INDEX IF NOT EXISTS idx_sensor_equipment ON sensor_readings (equipment_id, time DESC);

-- Engineered Features
CREATE INDEX IF NOT EXISTS idx_features_equipment ON engineered_features (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_features_set ON engineered_features (feature_set);

-- Feature Store
CREATE INDEX IF NOT EXISTS idx_fstore_equipment ON feature_store (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_fstore_version ON feature_store (feature_version);
CREATE INDEX IF NOT EXISTS idx_fstore_split ON feature_store (split);

-- Predictions
CREATE INDEX IF NOT EXISTS idx_pred_equipment ON predictions (equipment_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_pred_type ON predictions (prediction_type);
CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions (model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_pred_health ON predictions (health_status);

-- Alerts
CREATE INDEX IF NOT EXISTS idx_alerts_equipment ON alerts (equipment_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);
CREATE INDEX IF NOT EXISTS idx_alerts_triggered ON alerts (triggered_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_rule ON alerts (rule_id);

-- Maintenance Logs
CREATE INDEX IF NOT EXISTS idx_maint_equipment ON maintenance_logs (equipment_id);
CREATE INDEX IF NOT EXISTS idx_maint_type ON maintenance_logs (maintenance_type);
CREATE INDEX IF NOT EXISTS idx_maint_time ON maintenance_logs (started_at DESC);

-- Model Registry
CREATE INDEX IF NOT EXISTS idx_model_stage ON model_registry (stage);

-- Training Runs
CREATE INDEX IF NOT EXISTS idx_training_model ON training_runs (model_name);
CREATE INDEX IF NOT EXISTS idx_training_status ON training_runs (status);

-- Drift Logs
CREATE INDEX IF NOT EXISTS idx_drift_time ON drift_logs (check_time DESC);
CREATE INDEX IF NOT EXISTS idx_drift_type ON drift_logs (drift_type);

DO $$ BEGIN RAISE NOTICE '03-create-indexes: All indexes created'; END $$;
