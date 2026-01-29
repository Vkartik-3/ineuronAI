@echo off
REM Stop Script for Predictive Maintenance Infrastructure (Windows)

echo ==========================================
echo Stopping Predictive Maintenance Infrastructure
echo ==========================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set INFRA_DIR=%SCRIPT_DIR%..

cd /d "%INFRA_DIR%"

REM Stop services
echo Stopping services...
docker-compose stop

echo.
echo √ All services stopped
echo.
echo To remove containers: docker-compose down
echo To remove volumes: docker-compose down -v
echo.
