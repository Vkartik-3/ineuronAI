@echo off
REM Startup Script for Predictive Maintenance Infrastructure (Windows)

setlocal enabledelayedexpansion

echo ==========================================
echo Starting Predictive Maintenance Infrastructure
echo ==========================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not running. Please start Docker first.
    exit /b 1
)

REM Get script directory
set SCRIPT_DIR=%~dp0
set INFRA_DIR=%SCRIPT_DIR%..

cd /d "%INFRA_DIR%"

REM Check if docker-compose.yml exists
if not exist "docker-compose.yml" (
    echo ERROR: docker-compose.yml not found in %INFRA_DIR%
    exit /b 1
)

REM Pull latest images
echo Pulling latest Docker images...
docker-compose pull

REM Start services
echo.
echo Starting infrastructure services...
docker-compose up -d

echo.
echo Waiting for services to be ready...
timeout /t 10 /nobreak >nul

REM Wait for Kafka to be ready
echo Waiting for Kafka broker...
set /a MAX_WAIT=60
set /a WAITED=0

:wait_kafka
docker exec pm-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >nul 2>&1
if %ERRORLEVEL% EQU 0 goto kafka_ready

timeout /t 2 /nobreak >nul
set /a WAITED+=2
if %WAITED% GEQ %MAX_WAIT% (
    echo ERROR: Kafka failed to start within %MAX_WAIT% seconds
    echo Check logs with: docker-compose logs kafka
    exit /b 1
)
goto wait_kafka

:kafka_ready
echo √ Kafka is ready

REM Wait for TimescaleDB to be ready
echo Waiting for TimescaleDB...
set /a MAX_WAIT=30
set /a WAITED=0

:wait_db
docker exec pm-timescaledb pg_isready -U pmuser >nul 2>&1
if %ERRORLEVEL% EQU 0 goto db_ready

timeout /t 2 /nobreak >nul
set /a WAITED+=2
if %WAITED% GEQ %MAX_WAIT% (
    echo WARNING: TimescaleDB not ready within %MAX_WAIT% seconds
    goto continue
)
goto wait_db

:db_ready
echo √ TimescaleDB is ready

:continue
echo.
echo ==========================================
echo Infrastructure Started Successfully
echo ==========================================
echo.
echo Services Status:
docker-compose ps
echo.
echo Access Points:
echo   • Kafka: localhost:9092
echo   • Kafdrop UI: http://localhost:9000
echo   • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
echo   • TimescaleDB: localhost:5432 (pmuser/pmpassword)
echo   • Redis: localhost:6379
echo.
echo Next Steps:
echo   1. Create Kafka topics: scripts\create-topics.bat
echo   2. Run health check: scripts\health-check.bat
echo   3. Start data generator: cd ..\data_generator ^&^& python main.py
echo.

endlocal
