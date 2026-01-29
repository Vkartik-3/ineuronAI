# Kafka Infrastructure Module

## Overview

Production-ready Kafka streaming infrastructure for the Predictive Maintenance System. Includes Kafka, Zookeeper, TimescaleDB, MinIO (S3), Redis, and monitoring tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network                          │
│                 predictive-maintenance-network              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────────┐         │
│  │Zookeeper │◄───┤  Kafka   │◄───┤  Kafdrop UI  │         │
│  │  :2181   │    │  :9092   │    │    :9000     │         │
│  └──────────┘    └────┬─────┘    └──────────────┘         │
│                       │                                     │
│        ┌──────────────┼──────────────┬──────────┐          │
│        │              │              │          │          │
│  ┌─────▼─────┐  ┌────▼─────┐  ┌────▼────┐  ┌──▼───────┐  │
│  │TimescaleDB│  │  MinIO   │  │  Redis  │  │ Schema   │  │
│  │   :5432   │  │:9001/9002│  │  :6379  │  │ Registry │  │
│  │(Time-     │  │  (S3)    │  │(Cache)  │  │  :8081   │  │
│  │series DB) │  │          │  │         │  │          │  │
│  └───────────┘  └──────────┘  └─────────┘  └──────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Services

### Core Services

- **Kafka** (port 9092): Message broker for streaming sensor data
- **Zookeeper** (port 2181): Kafka coordination service
- **Kafdrop** (port 9000): Web UI for Kafka monitoring

### Storage Services

- **TimescaleDB** (port 5432): PostgreSQL + time-series optimization
  - User: `pmuser`, Password: `pmpassword`, DB: `predictive_maintenance`
- **MinIO** (ports 9001/9002): S3-compatible object storage
  - Console: http://localhost:9001 (minioadmin/minioadmin)
- **Redis** (port 6379): In-memory cache

### Optional Services

### Optional Services

### Optional Services (Disabled by Default)

- **Schema Registry** (port 8082): Avro schema management - Uncomment in docker-compose.yml to enable

## Quick Start

### 1. Start Infrastructure

**Linux/Mac:**

```bash
cd infra/kafka
chmod +x scripts/*.sh
./scripts/start-infra.sh
```

**Windows:**

```cmd
cd infra\kafka
scripts\start-infra.bat
```

### 2. Create Kafka Topics

**Linux/Mac:**

```bash
./scripts/create-topics.sh
```

**Windows:**

```cmd
scripts\create-topics.bat
```

### 3. Verify Health

**Linux/Mac:**

```bash
./scripts/health-check.sh
```

**Windows:**

```cmd
scripts\health-check.bat
```

### 4. Access Services

- **Kafdrop UI**: http://localhost:9000
- **MinIO Console**: http://localhost:9001
- **Kafka**: localhost:9092
- **TimescaleDB**: localhost:5432
- **Schema Registry**: http://localhost:8082

## Kafka Topics

| Topic                 | Partitions | Retention | Description                               |
| --------------------- | ---------- | --------- | ----------------------------------------- |
| `raw_sensor_data`     | 10         | 7 days    | Raw sensor readings from equipment        |
| `processed_features`  | 6          | 14 days   | Engineered features from stream processor |
| `failure_predictions` | 3          | 14 days   | ML model predictions (RUL, anomaly)       |
| `maintenance_alerts`  | 3          | 30 days   | Critical maintenance alerts               |
| `equipment_metadata`  | 1          | 90 days   | Equipment specifications                  |
| `model_metrics`       | 2          | 14 days   | Model performance metrics                 |
| `dlq_failed_messages` | 2          | 30 days   | Dead letter queue for failures            |

## Consumer Groups

Defined in `config/consumer-groups.yaml`:

- **stream-processor-group**: Processes raw sensor data
- **ml-inference-group**: Generates predictions
- **alert-engine-group**: Triggers maintenance alerts
- **dashboard-group**: Powers real-time dashboards
- **data-archiver-group**: Archives to S3/TimescaleDB
- **monitoring-group**: Collects metrics

## TimescaleDB Schema

### Tables

- `sensor_readings`: Hypertable for raw sensor data
- `processed_features`: Engineered features
- `predictions`: ML model outputs
- `maintenance_alerts`: Alert history
- `equipment_metadata`: Equipment master data
- `model_metrics`: Model performance tracking

### Continuous Aggregates

- `sensor_readings_hourly`: Hourly sensor aggregates
- `equipment_health_daily`: Daily health summaries

### Policies

- **Retention**: 30 days for raw data, 90-180 days for aggregates
- **Compression**: Data older than 7 days compressed
- **Indexes**: Optimized for equipment_id and time queries

## Management Scripts

### Start/Stop

```bash
./scripts/start-infra.sh   # Start all services
./scripts/stop-infra.sh    # Stop services (keep data)
./scripts/reset-infra.sh   # Delete everything (WARNING!)
```

### Monitoring

```bash
./scripts/health-check.sh  # Check service health
docker-compose logs -f     # View logs
docker-compose ps          # Service status
```

### Kafka Operations

```bash
# List topics
docker exec pm-kafka kafka-topics --list --bootstrap-server localhost:9092

# Describe topic
docker exec pm-kafka kafka-topics --describe --topic raw_sensor_data --bootstrap-server localhost:9092

# Consumer groups
docker exec pm-kafka kafka-consumer-groups --list --bootstrap-server localhost:9092

# Consumer group details
docker exec pm-kafka kafka-consumer-groups --describe --group stream-processor-group --bootstrap-server localhost:9092

# Produce test message
echo "test message" | docker exec -i pm-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic raw_sensor_data

# Consume messages
docker exec pm-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic raw_sensor_data --from-beginning --max-messages 10
```

### TimescaleDB Operations

```bash
# Connect to database
docker exec -it pm-timescaledb psql -U pmuser -d predictive_maintenance

# Useful queries
SELECT * FROM sensor_readings ORDER BY time DESC LIMIT 10;
SELECT * FROM equipment_health_daily ORDER BY bucket DESC;
SHOW timescaledb.license;
```

## Configuration

### Docker Compose

Edit `docker-compose.yml` to customize:

- Resource limits (CPU, memory)
- Port mappings
- Environment variables
- Volume locations

### Kafka Settings

Key configurations in `docker-compose.yml`:

```yaml
KAFKA_LOG_RETENTION_HOURS: 168 # 7 days
KAFKA_LOG_SEGMENT_BYTES: 1073741824 # 1GB segments
KAFKA_COMPRESSION_TYPE: gzip
KAFKA_HEAP_OPTS: "-Xmx1G -Xms1G"
```

### TimescaleDB

Connection string:

```
postgresql://pmuser:pmpassword@localhost:5432/predictive_maintenance
```

## Troubleshooting

### Kafka Not Starting

```bash
# Check Zookeeper first
docker logs pm-zookeeper

# Check Kafka logs
docker logs pm-kafka

# Verify Zookeeper connection
docker exec pm-kafka nc -zv zookeeper 2181
```

### Out of Disk Space

```bash
# Check Docker disk usage
docker system df

# Clean up old images/containers
docker system prune -a

# Check Kafka retention
docker exec pm-kafka kafka-configs --describe --entity-type topics --bootstrap-server localhost:9092
```

### Topics Not Created

```bash
# Re-run topic creation
./scripts/create-topics.sh

# Verify topics
docker exec pm-kafka kafka-topics --list --bootstrap-server localhost:9092
```

### TimescaleDB Connection Issues

```bash
# Check if ready
docker exec pm-timescaledb pg_isready -U pmuser

# View logs
docker logs pm-timescaledb

# Connect manually
docker exec -it pm-timescaledb psql -U pmuser -d predictive_maintenance
```

### Performance Tuning

**For high throughput (>10k msg/sec):**

```yaml
# In docker-compose.yml
KAFKA_HEAP_OPTS: "-Xmx4G -Xms4G"
KAFKA_NUM_NETWORK_THREADS: 8
KAFKA_NUM_IO_THREADS: 16
```

**For low latency:**

```yaml
KAFKA_SOCKET_SEND_BUFFER_BYTES: 102400
KAFKA_SOCKET_RECEIVE_BUFFER_BYTES: 102400
```

## Production Considerations

### Scaling

- **Kafka**: Add brokers with `KAFKA_BROKER_ID: 2, 3...`
- **Partitions**: Increase for parallel processing
- **Replication**: Set replication factor > 1 for HA
- **Kubernetes**: Use Strimzi operator for K8s deployment

### Security

- Enable SASL/SSL authentication
- Set up ACLs for topics
- Use secrets for passwords
- Network policies for isolation

### Monitoring

- Enable JMX metrics
- Use Prometheus + Grafana
- Monitor consumer lag
- Set up alerting

### Backup

```bash
# TimescaleDB backup
docker exec pm-timescaledb pg_dump -U pmuser predictive_maintenance > backup.sql

# MinIO backup (use mc client)
mc mirror minio/predictive-maintenance /backup/minio
```

## Resource Requirements

### Minimum

- RAM: 8GB
- CPU: 4 cores
- Disk: 50GB

### Recommended

- RAM: 16GB
- CPU: 8 cores
- Disk: 200GB SSD

## Integration

### With Data Generator

```bash
# Start infrastructure first
./scripts/start-infra.sh
./scripts/create-topics.sh

# Then start data generator
cd ../../data_generator
python main.py --num-equipment 10
```

### With Stream Processor (Module 3)

```python
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'raw_sensor_data',
    bootstrap_servers='localhost:9092',
    group_id='stream-processor-group'
)
```

## Next Steps

1. ✅ Infrastructure is ready
2. ⏭️ Implement Stream Processor (Module 3)
3. ⏭️ Build Feature Store (Module 4)
4. ⏭️ Train ML Models (Module 5-6)

## References

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [TimescaleDB Docs](https://docs.timescale.com/)
- [Kafdrop GitHub](https://github.com/obsidiandynamics/kafdrop)
- [MinIO Documentation](https://min.io/docs/)

---

**Part of the Predictive Maintenance System**
