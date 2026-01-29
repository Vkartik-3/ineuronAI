#!/bin/bash

# Health Check Script for Kafka Infrastructure
# Verifies all services are running and topics are accessible

set -e

KAFKA_CONTAINER="pm-kafka"
BOOTSTRAP_SERVER="localhost:9092"

echo "=========================================="
echo "Kafka Infrastructure Health Check"
echo "=========================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local container_name=$2

    echo -n "Checking $service_name... "

    if docker ps --format '{{.Names}}' | grep -q "^${container_name}$"; then
        local health=$(docker inspect --format='{{.State.Health.Status}}' $container_name 2>/dev/null || echo "no-health-check")

        if [ "$health" == "healthy" ] || [ "$health" == "no-health-check" ]; then
            echo -e "${GREEN}✓ Running${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠ Running but unhealthy (status: $health)${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Not running${NC}"
        return 1
    fi
}

# Check Docker daemon
echo -n "Checking Docker daemon... "
if docker info > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Docker is not running${NC}"
    exit 1
fi
echo ""

# Check core services
echo "Core Services:"
echo "----------------------------------------"
check_service "Zookeeper" "pm-zookeeper"
check_service "Kafka" "pm-kafka"
check_service "Kafdrop" "pm-kafdrop"
echo ""

# Check supporting services
echo "Supporting Services:"
echo "----------------------------------------"
check_service "TimescaleDB" "pm-timescaledb"
check_service "MinIO" "pm-minio"
check_service "Redis" "pm-redis"
echo -n "Checking Schema Registry (optional)... "
if docker ps --format '{{.Names}}' | grep -q "^pm-schema-registry$"; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${YELLOW}⚠ Not enabled (optional service)${NC}"
fi
echo ""

# Check Kafka broker connectivity
echo "Kafka Broker Connectivity:"
echo "----------------------------------------"
if docker exec $KAFKA_CONTAINER kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Kafka broker is accessible${NC}"
else
    echo -e "${RED}✗ Cannot connect to Kafka broker${NC}"
    exit 1
fi
echo ""

# List topics
echo "Kafka Topics:"
echo "----------------------------------------"
TOPICS=$(docker exec $KAFKA_CONTAINER kafka-topics --list --bootstrap-server $BOOTSTRAP_SERVER 2>/dev/null)

if [ -z "$TOPICS" ]; then
    echo -e "${YELLOW}⚠ No topics found (run create-topics script)${NC}"
else
    echo "$TOPICS" | while read topic; do
        echo -e "${GREEN}✓${NC} $topic"
    done
fi
echo ""

# Check consumer groups
echo "Consumer Groups:"
echo "----------------------------------------"
GROUPS=$(docker exec $KAFKA_CONTAINER kafka-consumer-groups --bootstrap-server $BOOTSTRAP_SERVER --list 2>/dev/null)

if [ -z "$GROUPS" ]; then
    echo -e "${YELLOW}⚠ No consumer groups found${NC}"
else
    echo "$GROUPS" | while read group; do
        echo -e "${GREEN}✓${NC} $group"
    done
fi
echo ""

# Check TimescaleDB connectivity
echo "TimescaleDB Connection:"
echo "----------------------------------------"
if docker exec pm-timescaledb pg_isready -U pmuser > /dev/null 2>&1; then
    echo -e "${GREEN}✓ PostgreSQL/TimescaleDB is ready${NC}"

    # Check if database exists
    DB_EXISTS=$(docker exec pm-timescaledb psql -U pmuser -lqt 2>/dev/null | cut -d \| -f 1 | grep -w predictive_maintenance | wc -l)
    if [ "$DB_EXISTS" -eq 1 ]; then
        echo -e "${GREEN}✓ Database 'predictive_maintenance' exists${NC}"
    else
        echo -e "${YELLOW}⚠ Database 'predictive_maintenance' not found${NC}"
    fi
else
    echo -e "${RED}✗ Cannot connect to TimescaleDB${NC}"
fi
echo ""

# Check MinIO connectivity
echo "MinIO (S3) Connection:"
echo "----------------------------------------"
if curl -f http://localhost:9002/minio/health/live > /dev/null 2>&1; then
    echo -e "${GREEN}✓ MinIO is accessible${NC}"
    echo "   Console: http://localhost:9001"
    echo "   API: http://localhost:9002"
else
    echo -e "${RED}✗ Cannot connect to MinIO${NC}"
fi
echo ""

# Check Redis connectivity
echo "Redis Connection:"
echo "----------------------------------------"
if docker exec pm-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis is responding${NC}"
else
    echo -e "${RED}✗ Cannot connect to Redis${NC}"
fi
echo ""

# Summary
echo "=========================================="
echo "Health Check Complete"
echo "=========================================="
echo ""
echo "Access Points:"
echo "  Kafka: localhost:9092"
echo "  Kafdrop UI: http://localhost:9000"
echo "  MinIO Console: http://localhost:9001"
echo "  TimescaleDB: localhost:5432"
echo "  Redis: localhost:6379"
echo ""
