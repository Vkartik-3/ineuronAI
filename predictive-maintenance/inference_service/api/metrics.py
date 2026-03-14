"""
Prometheus metrics for the Inference Service.

Exposes the /metrics endpoint and provides Counter / Histogram / Gauge
instruments for tracking prediction requests, latency, and RUL values.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)
from fastapi import APIRouter, Response

router = APIRouter()

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["model", "status"],
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
INFERENCE_LATENCY_SECONDS = Histogram(
    "inference_latency_seconds",
    "Inference request latency in seconds",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
PREDICTION_RUL_HOURS = Gauge(
    "prediction_rul_hours",
    "Latest predicted RUL in hours",
    ["equipment_id"],
)

MODELS_LOADED = Gauge(
    "models_loaded",
    "Number of models currently loaded",
)

KAFKA_PIPELINE_RUNNING = Gauge(
    "kafka_pipeline_running",
    "Whether the Kafka prediction pipeline is running (1=yes, 0=no)",
)

SERVICE_UPTIME_SECONDS = Gauge(
    "service_uptime_seconds",
    "Time since service started in seconds",
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
