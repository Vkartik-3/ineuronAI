@echo off
REM Topic Creation Script for Predictive Maintenance System (Windows)
REM Creates all required Kafka topics with appropriate configurations

setlocal enabledelayedexpansion

set KAFKA_CONTAINER=pm-kafka
set BOOTSTRAP_SERVER=localhost:9092

echo ==========================================
echo Kafka Topic Creation Script
echo ==========================================

echo Waiting for Kafka to be ready...
set /a MAX_WAIT=60
set /a WAITED=0

:wait_loop
docker exec %KAFKA_CONTAINER% kafka-broker-api-versions --bootstrap-server localhost:9092 >nul 2>&1
if %ERRORLEVEL% EQU 0 goto kafka_ready

timeout /t 2 /nobreak >nul
set /a WAITED+=2
if %WAITED% GEQ %MAX_WAIT% (
    echo ERROR: Kafka failed to start within %MAX_WAIT% seconds
    exit /b 1
)
goto wait_loop

:kafka_ready
echo √ Kafka is ready
echo.

REM Create topics using Docker exec

echo Creating topic: raw_sensor_data
docker exec %KAFKA_CONTAINER% kafka-topics --create --if-not-exists --bootstrap-server %BOOTSTRAP_SERVER% --topic raw_sensor_data --partitions 10 --replication-factor 1 --config retention.ms=604800000 --config cleanup.policy=delete --config compression.type=gzip --config min.insync.replicas=1
echo √ Topic 'raw_sensor_data' created
echo.

echo Creating topic: processed_features
docker exec %KAFKA_CONTAINER% kafka-topics --create --if-not-exists --bootstrap-server %BOOTSTRAP_SERVER% --topic processed_features --partitions 6 --replication-factor 1 --config retention.ms=1209600000 --config cleanup.policy=delete --config compression.type=gzip --config min.insync.replicas=1
echo √ Topic 'processed_features' created
echo.

echo Creating topic: failure_predictions
docker exec %KAFKA_CONTAINER% kafka-topics --create --if-not-exists --bootstrap-server %BOOTSTRAP_SERVER% --topic failure_predictions --partitions 3 --replication-factor 1 --config retention.ms=1209600000 --config cleanup.policy=delete --config compression.type=gzip --config min.insync.replicas=1
echo √ Topic 'failure_predictions' created
echo.

echo Creating topic: maintenance_alerts
docker exec %KAFKA_CONTAINER% kafka-topics --create --if-not-exists --bootstrap-server %BOOTSTRAP_SERVER% --topic maintenance_alerts --partitions 3 --replication-factor 1 --config retention.ms=2592000000 --config cleanup.policy=compact --config compression.type=gzip --config min.insync.replicas=1
echo √ Topic 'maintenance_alerts' created
echo.

echo Creating topic: equipment_metadata
docker exec %KAFKA_CONTAINER% kafka-topics --create --if-not-exists --bootstrap-server %BOOTSTRAP_SERVER% --topic equipment_metadata --partitions 1 --replication-factor 1 --config retention.ms=7776000000 --config cleanup.policy=compact --config compression.type=gzip --config min.insync.replicas=1
echo √ Topic 'equipment_metadata' created
echo.

echo Creating topic: model_metrics
docker exec %KAFKA_CONTAINER% kafka-topics --create --if-not-exists --bootstrap-server %BOOTSTRAP_SERVER% --topic model_metrics --partitions 2 --replication-factor 1 --config retention.ms=1209600000 --config cleanup.policy=delete --config compression.type=gzip --config min.insync.replicas=1
echo √ Topic 'model_metrics' created
echo.

echo Creating topic: dlq_failed_messages
docker exec %KAFKA_CONTAINER% kafka-topics --create --if-not-exists --bootstrap-server %BOOTSTRAP_SERVER% --topic dlq_failed_messages --partitions 2 --replication-factor 1 --config retention.ms=2592000000 --config cleanup.policy=delete --config compression.type=gzip --config min.insync.replicas=1
echo √ Topic 'dlq_failed_messages' created
echo.

echo ==========================================
echo Topic Creation Complete
echo ==========================================
echo.

echo Listing all topics:
docker exec %KAFKA_CONTAINER% kafka-topics --list --bootstrap-server %BOOTSTRAP_SERVER%
echo.

echo Describing topics:
docker exec %KAFKA_CONTAINER% kafka-topics --describe --bootstrap-server %BOOTSTRAP_SERVER%
echo.

echo √ All topics created successfully!
echo.
echo Access Kafdrop UI at: http://localhost:9000
echo.

endlocal
