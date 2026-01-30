# Docker Infrastructure Troubleshooting Guide

## Fixed Issues ✅

### 1. Docker Compose Dependency Error

**Issue:** Services depend on undefined infrastructure services (kafka, timescaledb, etc.)

**Fix Applied:** Removed cross-compose-file `depends_on` references. Infrastructure and application services are now in separate compose files.

**Solution:**

1. Start infrastructure first: `cd infra/kafka && docker-compose up -d`
2. Wait 30 seconds for services to be healthy
3. Start application services: `cd ../.. && docker-compose up -d`

---

## Current Issues ⚠️

### 2. TimescaleDB Container Restarting

**Symptoms:**

- Container restarts continuously
- Logs show initialization errors or connection issues

**Diagnostic Steps:**

```bash
# Check TimescaleDB logs
docker logs pm-timescaledb

# Check container status
docker ps -a | grep timescaledb

# Check if port is in use
netstat -an | grep 5432

# Inspect container
docker inspect pm-timescaledb
```

**Common Causes & Fixes:**

#### A. Port 5432 Already in Use

```bash
# Check what's using port 5432
netstat -ano | findstr :5432  # Windows
lsof -i :5432                  # Linux/Mac

# Option 1: Stop conflicting service
# Stop PostgreSQL service if running

# Option 2: Change port in docker-compose.yml
# Change "5432:5432" to "5433:5432"
```

#### B. Volume/Permission Issues

```bash
# Remove volume and recreate
docker-compose down
docker volume rm kafka_timescale-data
docker-compose up -d timescaledb

# Check volume
docker volume inspect kafka_timescale-data
```

#### C. Memory/Resource Constraints

```bash
# Check Docker resource limits
docker stats pm-timescaledb

# Increase Docker Desktop memory allocation:
# Docker Desktop → Settings → Resources → Memory (set to 4GB+)
```

#### D. Database Initialization Failure

```bash
# Check init scripts
ls -la infra/kafka/init-db/

# Remove container and volume, start fresh
docker-compose down -v
docker-compose up -d timescaledb

# Watch initialization
docker logs -f pm-timescaledb
```

**Manual Fix:**

```bash
cd infra/kafka

# Stop and remove TimescaleDB
docker-compose stop timescaledb
docker-compose rm -f timescaledb
docker volume rm kafka_timescale-data

# Restart just TimescaleDB
docker-compose up -d timescaledb

# Monitor logs
docker logs -f pm-timescaledb
```

**Expected Healthy Output:**

```
PostgreSQL init process complete; ready for start up.
database system is ready to accept connections
```

---

### 3. Kafdrop Unhealthy (Minor Issue)

**Status:** Service is running but shows unhealthy (no health check defined)

**Impact:** Cosmetic only - service works fine

**Fix (Optional):** Add health check to `infra/kafka/docker-compose.yml`:

```yaml
kafdrop:
  # ... existing config ...
  healthcheck:
    test:
      [
        "CMD",
        "wget",
        "--quiet",
        "--tries=1",
        "--spider",
        "http://localhost:9000",
      ]
    interval: 30s
    timeout: 10s
    retries: 3
```

---

## Verification Commands

### Check All Infrastructure Status

```bash
cd infra/kafka
docker-compose ps
```

Expected output - all should be "Up" or "Up (healthy)":

- ✅ pm-zookeeper
- ✅ pm-kafka
- ✅ pm-timescaledb (if healthy)
- ✅ pm-redis
- ✅ pm-minio
- ⚠️ pm-kafdrop (Up but may show unhealthy)

### Test Services Individually

**Kafka:**

```bash
# List topics
docker exec pm-kafka kafka-topics --bootstrap-server localhost:9092 --list

# Access Kafdrop UI
# Open: http://localhost:9000
```

**TimescaleDB:**

```bash
# Connect to database
docker exec -it pm-timescaledb psql -U pmuser -d predictive_maintenance

# Run test query
\dt
SELECT version();
\q
```

**Redis:**

```bash
# Test Redis connection
docker exec -it pm-redis redis-cli ping
# Should return: PONG
```

**MinIO:**

```bash
# Access MinIO Console
# Open: http://localhost:9001
# Login: minioadmin / minioadmin
```

---

## Complete Restart Sequence

If multiple issues persist:

```bash
# 1. Stop everything
cd /d/data-science-projects/predictive-maintenance
docker-compose down
cd infra/kafka
docker-compose down

# 2. Remove problematic volumes (optional - will delete data)
docker volume rm kafka_timescale-data kafka_kafka-data

# 3. Start infrastructure
docker-compose up -d

# 4. Wait and verify (30-60 seconds)
watch docker-compose ps
docker logs -f pm-timescaledb

# 5. Once healthy, start applications
cd ../..
docker-compose up -d

# 6. Check all services
docker ps
docker-compose ps
```

---

## Access Points (When All Healthy)

### Infrastructure

- 🌐 Kafka UI (Kafdrop): http://localhost:9000
- 🗄️ TimescaleDB: `localhost:5432` (pmuser/pmpassword)
- 💾 Redis: `localhost:6379`
- 📦 MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

### Applications

- 🔮 Inference API: http://localhost:8000/docs
- 📊 Streamlit Dashboard: http://localhost:8501
- 📈 Grafana: http://localhost:3000
- 🔬 MLflow: http://localhost:5000

---

## Quick Diagnostic Script

Save as `check_services.sh`:

```bash
#!/bin/bash
echo "=== Checking Infrastructure Services ==="
echo ""

services=("pm-zookeeper" "pm-kafka" "pm-timescaledb" "pm-redis" "pm-minio" "pm-kafdrop")

for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Status}}' $service 2>/dev/null || echo "not found")
    health=$(docker inspect -f '{{.State.Health.Status}}' $service 2>/dev/null || echo "n/a")
    echo "$service: $status (health: $health)"
done

echo ""
echo "=== Kafka Topics ==="
docker exec pm-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null || echo "Kafka not ready"

echo ""
echo "=== TimescaleDB Connection ==="
docker exec pm-timescaledb psql -U pmuser -d predictive_maintenance -c "SELECT version();" 2>/dev/null | head -3 || echo "TimescaleDB not ready"
```

Run: `bash check_services.sh`
