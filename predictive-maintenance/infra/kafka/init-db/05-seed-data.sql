-- ============================================================================
-- Predictive Maintenance System — Database Initialization
-- Script 5/5: Seed Data
-- ============================================================================

-- Default alert rules (matching alert_config.yaml)
INSERT INTO alert_rules (rule_id, name, description, condition, severity, notification_channels, cooldown_seconds)
VALUES
    ('rul_critical',       'RUL Critical',          'Equipment RUL below critical threshold',            'rul < 10',                          'critical', '["database","email","slack","webhook"]', 3600),
    ('rul_warning',        'RUL Warning',           'Equipment RUL below warning threshold',             'rul < 30',                          'warning',  '["database","email","slack"]',          7200),
    ('rul_info',           'RUL Info',              'Equipment RUL below informational threshold',        'rul < 50',                          'info',     '["database"]',                          14400),
    ('anomaly_critical',   'Anomaly Critical',      'Anomaly score exceeds critical threshold',           'anomaly_score > 0.9',               'critical', '["database","email","slack","webhook"]', 1800),
    ('anomaly_warning',    'Anomaly Warning',       'Anomaly score exceeds warning threshold',            'anomaly_score > 0.7',               'warning',  '["database","email","slack"]',          3600),
    ('health_imminent',    'Health Imminent Failure','Equipment health classified as imminent failure',    'health_status == ''imminent_failure''','critical','["database","email","slack","webhook"]', 1800),
    ('health_critical',    'Health Critical',       'Equipment health classified as critical',             'health_status == ''critical''',     'warning',  '["database","email","slack"]',          3600),
    ('temperature_high',   'High Temperature',      'Equipment temperature exceeds threshold',             'temperature > 95',                  'warning',  '["database","email","slack"]',          7200),
    ('vibration_high',     'High Vibration',        'Equipment vibration exceeds threshold',               'vibration > 0.8',                   'warning',  '["database","email","slack"]',          7200),
    ('rapid_degradation',  'Rapid Degradation',     'RUL declining rapidly',                               'rul_change_rate > 5',               'critical', '["database","email","slack","webhook"]', 3600),
    ('multi_sensor_anomaly','Multi Sensor Anomaly', 'Multiple sensors showing anomalous readings',         'anomaly_count >= 3',                'critical', '["database","email","slack","webhook"]', 1800)
ON CONFLICT (rule_id) DO NOTHING;

-- Seed equipment records for C-MAPSS dataset (FD001: 100 engines)
DO $$
BEGIN
    FOR i IN 1..100 LOOP
        INSERT INTO equipment (equipment_id, equipment_type, model, manufacturer, location, install_date, status, metadata)
        VALUES (
            'ENGINE_' || LPAD(i::text, 4, '0'),
            'turbofan_engine',
            'TURBOFAN_V1',
            'NASA_CMAPSS',
            'Factory_Floor_1',
            NOW() - INTERVAL '2 years',
            'active',
            jsonb_build_object('dataset', 'FD001', 'unit_number', i)
        )
        ON CONFLICT (equipment_id) DO NOTHING;
    END LOOP;
END $$;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pmuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pmuser;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO pmuser;

DO $$ BEGIN RAISE NOTICE '05-seed-data: Alert rules and equipment records seeded'; END $$;
