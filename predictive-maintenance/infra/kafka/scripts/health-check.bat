@echo off
REM Health Check Script for Kafka Infrastructure (Windows)

setlocal enabledelayedexpansion

set KAFKA_CONTAINER=pm-kafka
set BOOTSTRAP_SERVER=localhost:9092

echo ==========================================
echo Kafka Infrastructure Health Check
echo ==========================================
echo.

echo Checking Docker daemon...
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo X Docker is not running
    exit /b 1
)
echo √ Docker is running
echo.

echo Core Services:
echo ----------------------------------------

REM Check Zookeeper
docker ps --format "{{.Names}}" | findstr /C:"pm-zookeeper" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ Zookeeper - Running
) else (
    echo X Zookeeper - Not running
)

REM Check Kafka
docker ps --format "{{.Names}}" | findstr /C:"pm-kafka" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ Kafka - Running
) else (
    echo X Kafka - Not running
)

REM Check Kafdrop
docker ps --format "{{.Names}}" | findstr /C:"pm-kafdrop" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ Kafdrop - Running
) else (
    echo X Kafdrop - Not running
)
echo.

echo Supporting Services:
echo ----------------------------------------

REM Check TimescaleDB
docker ps --format "{{.Names}}" | findstr /C:"pm-timescaledb" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ TimescaleDB - Running
) else (
    echo X TimescaleDB - Not running
)

REM Check MinIO
docker ps --format "{{.Names}}" | findstr /C:"pm-minio" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ MinIO - Running
) else (
    echo X MinIO - Not running
)

REM Check Redis
docker ps --format "{{.Names}}" | findstr /C:"pm-redis" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ Redis - Running
) else (
    echo X Redis - Not running
)

REM Check Schema Registry (optional)
echo Checking Schema Registry (optional)...
docker ps --format "{{.Names}}" | findstr /C:"pm-schema-registry" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ Schema Registry - Running
) else (
    echo ⚠ Schema Registry - Not enabled (optional service)
)
echo.

echo Kafka Broker Connectivity:
echo ----------------------------------------
docker exec %KAFKA_CONTAINER% kafka-broker-api-versions --bootstrap-server localhost:9092 >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ Kafka broker is accessible
) else (
    echo X Cannot connect to Kafka broker
)
echo.

echo Kafka Topics:
echo ----------------------------------------
docker exec %KAFKA_CONTAINER% kafka-topics --list --bootstrap-server %BOOTSTRAP_SERVER% 2>nul
echo.

echo Consumer Groups:
echo ----------------------------------------
docker exec %KAFKA_CONTAINER% kafka-consumer-groups --bootstrap-server %BOOTSTRAP_SERVER% --list 2>nul
echo.

echo TimescaleDB Connection:
echo ----------------------------------------
docker exec pm-timescaledb pg_isready -U pmuser >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ PostgreSQL/TimescaleDB is ready
) else (
    echo X Cannot connect to TimescaleDB
)
echo.

echo Redis Connection:
echo ----------------------------------------
docker exec pm-redis redis-cli ping >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo √ Redis is responding
) else (
    echo X Cannot connect to Redis
)
echo.

echo ==========================================
echo Health Check Complete
echo ==========================================
echo.
echo Access Points:
echo   Kafka: localhost:9092
echo   Kafdrop UI: http://localhost:9000
echo   MinIO Console: http://localhost:9001
echo   TimescaleDB: localhost:5432
echo   Redis: localhost:6379
echo.

endlocal
