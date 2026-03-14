"""
Prometheus metrics for the Alert Engine.

Exposes metrics via the FastAPI app (mounted by api.py) and provides
counters and gauges for alert tracking.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)
from fastapi import APIRouter, Response

router = APIRouter()

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
ALERTS_TRIGGERED_TOTAL = Counter(
    "alerts_triggered_total",
    "Total number of alerts triggered",
    ["severity", "rule_id"],
)

NOTIFICATIONS_SENT_TOTAL = Counter(
    "notifications_sent_total",
    "Total number of notifications sent",
    ["channel"],  # "email", "slack", "webhook", "database"
)

NOTIFICATIONS_FAILED_TOTAL = Counter(
    "notifications_failed_total",
    "Total number of failed notification attempts",
    ["channel"],
)

ALERTS_ACKNOWLEDGED_TOTAL = Counter(
    "alerts_acknowledged_total",
    "Total number of alerts acknowledged",
)

ALERTS_RESOLVED_TOTAL = Counter(
    "alerts_resolved_total",
    "Total number of alerts resolved",
)

KAFKA_MESSAGES_CONSUMED_TOTAL = Counter(
    "alert_kafka_messages_consumed_total",
    "Total number of Kafka messages consumed by alert engine",
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
ALERT_EVALUATION_SECONDS = Histogram(
    "alert_evaluation_seconds",
    "Time to evaluate all alert rules for one prediction",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
ACTIVE_ALERTS = Gauge(
    "active_alerts",
    "Number of currently active (un-resolved) alerts",
)

KAFKA_CONSUMER_RUNNING = Gauge(
    "alert_kafka_consumer_running",
    "Whether the Kafka alert consumer is running (1=yes, 0=no)",
)


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------
@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Expose Prometheus metrics."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
