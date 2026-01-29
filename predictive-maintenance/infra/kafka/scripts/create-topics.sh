#!/bin/bash

# Topic Creation Script for Predictive Maintenance System
# Creates all required Kafka topics with appropriate configurations

set -e

KAFKA_CONTAINER="pm-kafka"
BOOTSTRAP_SERVER="localhost:9092"

echo "=========================================="
echo "Kafka Topic Creation Script"
echo "=========================================="

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
MAX_WAIT=60
WAITED=0
until docker exec $KAFKA_CONTAINER kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: Kafka failed to start within ${MAX_WAIT} seconds"
        exit 1
    fi
done
echo "✓ Kafka is ready"

# Function to create topic
create_topic() {
    local topic_name=$1
    local partitions=$2
    local replication_factor=$3
    local retention_ms=$4
    local cleanup_policy=$5
    local compression=$6

    echo ""
    echo "Creating topic: $topic_name"
    echo "  Partitions: $partitions"
    echo "  Replication Factor: $replication_factor"
    echo "  Retention: $retention_ms ms"
    echo "  Cleanup Policy: $cleanup_policy"
    echo "  Compression: $compression"

    docker exec $KAFKA_CONTAINER kafka-topics \
        --create \
        --if-not-exists \
        --bootstrap-server $BOOTSTRAP_SERVER \
        --topic $topic_name \
        --partitions $partitions \
        --replication-factor $replication_factor \
        --config retention.ms=$retention_ms \
        --config cleanup.policy=$cleanup_policy \
        --config compression.type=$compression \
        --config min.insync.replicas=1

    if [ $? -eq 0 ]; then
        echo "✓ Topic '$topic_name' created successfully"
    else
        echo "✗ Failed to create topic '$topic_name'"
    fi
}

# Topic configurations based on data flow architecture

# 1. Raw Sensor Data Topic
# - High throughput sensor readings from data generator
# - Short retention (7 days) since data is archived to S3/TimescaleDB
# - Multiple partitions for parallel processing
create_topic "raw_sensor_data" 10 1 604800000 "delete" "gzip"

# 2. Processed Features Topic
# - Engineered features from stream processor
# - Medium retention (14 days)
# - Fewer partitions since data is aggregated
create_topic "processed_features" 6 1 1209600000 "delete" "gzip"

# 3. Failure Predictions Topic
# - ML model predictions (RUL, anomaly scores)
# - Medium retention (14 days)
# - Low volume, few partitions
create_topic "failure_predictions" 3 1 1209600000 "delete" "gzip"

# 4. Maintenance Alerts Topic
# - Critical alerts for maintenance teams
# - Long retention (30 days) for audit trail
# - Compact policy to keep latest state per equipment
create_topic "maintenance_alerts" 3 1 2592000000 "compact" "gzip"

# 5. Equipment Metadata Topic
# - Equipment specifications and configurations
# - Very long retention (90 days)
# - Compacted to keep latest metadata per equipment
create_topic "equipment_metadata" 1 1 7776000000 "compact" "gzip"

# 6. Model Performance Metrics Topic
# - ML model performance tracking
# - Medium retention (14 days)
create_topic "model_metrics" 2 1 1209600000 "delete" "gzip"

# 7. Dead Letter Queue
# - Failed messages for debugging
# - Long retention (30 days)
create_topic "dlq_failed_messages" 2 1 2592000000 "delete" "gzip"

echo ""
echo "=========================================="
echo "Topic Creation Complete"
echo "=========================================="
echo ""

# List all topics
echo "Listing all topics:"
docker exec $KAFKA_CONTAINER kafka-topics \
    --list \
    --bootstrap-server $BOOTSTRAP_SERVER

echo ""
echo "Describing topics:"
docker exec $KAFKA_CONTAINER kafka-topics \
    --describe \
    --bootstrap-server $BOOTSTRAP_SERVER

echo ""
echo "✓ All topics created successfully!"
echo ""
echo "Access Kafdrop UI at: http://localhost:9000"
echo ""
