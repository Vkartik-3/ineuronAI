-- ============================================================================
-- Predictive Maintenance System — Database Initialization
-- Script 1/5: Extensions
-- ============================================================================
-- Enables required PostgreSQL extensions for time-series functionality.

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Success indicator
DO $$ BEGIN RAISE NOTICE '01-create-extensions: TimescaleDB extension enabled'; END $$;
