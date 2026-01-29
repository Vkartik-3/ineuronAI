@echo off
REM Reset Script for Predictive Maintenance Infrastructure (Windows)

setlocal enabledelayedexpansion

echo ==========================================
echo RESET Predictive Maintenance Infrastructure
echo ==========================================
echo.
echo WARNING: This will DELETE all data including:
echo   - All Kafka topics and messages
echo   - All TimescaleDB data
echo   - All MinIO objects
echo   - All Redis cache
echo.

set /p confirm="Are you sure you want to continue? (yes/no): "
if /i not "%confirm%"=="yes" (
    echo Reset cancelled
    exit /b 0
)

echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set INFRA_DIR=%SCRIPT_DIR%..

cd /d "%INFRA_DIR%"

REM Stop and remove everything
echo Stopping all containers...
docker-compose down

echo Removing volumes...
docker-compose down -v

echo Removing orphaned containers...
docker-compose down --remove-orphans

echo.
echo ==========================================
echo Infrastructure Reset Complete
echo ==========================================
echo.
echo To start fresh, run: scripts\start-infra.bat
echo.

endlocal
