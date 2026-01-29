#!/bin/bash

# Startup Script for Predictive Maintenance Infrastructure
# Starts all required services in correct order

set -e

echo "=========================================="
echo "Starting Predictive Maintenance Infrastructure"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"

cd "$INFRA_DIR"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "ERROR: docker-compose.yml not found in $INFRA_DIR"
    exit 1
fi

# Pull latest images
echo "Pulling latest Docker images..."
docker-compose pull

# Start services
echo ""
echo "Starting infrastructure services..."
docker-compose up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

# Wait for Kafka to be ready
echo "Waiting for Kafka broker..."
MAX_WAIT=60
WAITED=0

until docker exec pm-kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Kafka failed to start within ${MAX_WAIT} seconds"
        echo "Check logs with: docker-compose logs kafka"
        exit 1
    fi
done

echo "✓ Kafka is ready"

# Wait for TimescaleDB to be ready
echo "Waiting for TimescaleDB..."
MAX_WAIT=30
WAITED=0

until docker exec pm-timescaledb pg_isready -U pmuser > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "WARNING: TimescaleDB not ready within ${MAX_WAIT} seconds"
        break
    fi
done

if docker exec pm-timescaledb pg_isready -U pmuser > /dev/null 2>&1; then
    echo "✓ TimescaleDB is ready"
else
    echo "⚠ TimescaleDB may not be ready yet"
fi

echo ""
echo "=========================================="
echo "Infrastructure Started Successfully"
echo "=========================================="
echo ""
echo "Services Status:"
docker-compose ps
echo ""
echo "Access Points:"
echo "  • Kafka: localhost:9092"
echo "  • Kafdrop UI: http://localhost:9000"
echo "  • MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo "  • TimescaleDB: localhost:5432 (pmuser/pmpassword)"
echo "  • Redis: localhost:6379"
echo ""
echo "Next Steps:"
echo "  1. Create Kafka topics: ./scripts/create-topics.sh"
echo "  2. Run health check: ./scripts/health-check.sh"
echo "  3. Start data generator: cd ../data_generator && python main.py"
echo ""
