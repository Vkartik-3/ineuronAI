#!/bin/bash

# Stop Script for Predictive Maintenance Infrastructure
# Gracefully stops all services

set -e

echo "=========================================="
echo "Stopping Predictive Maintenance Infrastructure"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"

cd "$INFRA_DIR"

# Stop services
echo "Stopping services..."
docker-compose stop

echo ""
echo "✓ All services stopped"
echo ""
echo "To remove containers: docker-compose down"
echo "To remove volumes: docker-compose down -v"
echo ""
