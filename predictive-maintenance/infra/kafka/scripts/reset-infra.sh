#!/bin/bash

# Reset Script for Predictive Maintenance Infrastructure
# Completely removes all containers, volumes, and networks

set -e

echo "=========================================="
echo "RESET Predictive Maintenance Infrastructure"
echo "=========================================="
echo ""
echo "WARNING: This will DELETE all data including:"
echo "  - All Kafka topics and messages"
echo "  - All TimescaleDB data"
echo "  - All MinIO objects"
echo "  - All Redis cache"
echo ""

read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Reset cancelled"
    exit 0
fi

echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"

cd "$INFRA_DIR"

# Stop and remove everything
echo "Stopping all containers..."
docker-compose down

echo "Removing volumes..."
docker-compose down -v

echo "Removing orphaned containers..."
docker-compose down --remove-orphans

echo ""
echo "=========================================="
echo "Infrastructure Reset Complete"
echo "=========================================="
echo ""
echo "To start fresh, run: ./scripts/start-infra.sh"
echo ""
