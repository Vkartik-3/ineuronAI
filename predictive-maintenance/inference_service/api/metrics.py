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
# Idempotent instrument factory
# ---------------------------------------------------------------------------
# metrics.py can legitimately be imported under more than one module identity
# (e.g. "inference_service.api.metrics" and "api.metrics" via different
# sys.path roots) or re-executed by a reloading worker. Constructing a
# prometheus_client collector registers it on the global REGISTRY, and a second
# construction with the same name raises "Duplicated timeseries in
# CollectorRegistry". To keep import order-independent and reload-safe, reuse
# the already-registered collector instead of crashing the process.
def _get_or_create(cls, name, documentation, labelnames=(), **kwargs):
    try:
        return cls(name, documentation, labelnames=labelnames, **kwargs)
    except ValueError:
        existing = REGISTRY._names_to_collectors.get(name)
        if existing is not None:
            return existing
        raise


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
INFERENCE_REQUESTS_TOTAL = _get_or_create(
    Counter,
    "inference_requests_total",
    "Total number of inference requests",
    ["model", "status"],
)

CACHE_HITS_TOTAL = _get_or_create(
    Counter,
    "prediction_cache_hits_total",
    "Total number of Redis prediction-cache hits",
)

CACHE_MISSES_TOTAL = _get_or_create(
    Counter,
    "prediction_cache_misses_total",
    "Total number of Redis prediction-cache misses",
)

RETRAINING_TRIGGERED_TOTAL = _get_or_create(
    Counter,
    "retraining_triggered_total",
    "Total number of retraining runs triggered (manual or automatic)",
    ["reason"],
)

FEEDBACK_RECORDS_TOTAL = _get_or_create(
    Counter,
    "feedback_records_total",
    "Total number of ground-truth feedback records recorded via POST /feedback",
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
INFERENCE_LATENCY_SECONDS = _get_or_create(
    Histogram,
    "inference_latency_seconds",
    "Inference request latency in seconds",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
PREDICTION_RUL_HOURS = _get_or_create(
    Gauge,
    "prediction_rul_hours",
    "Latest predicted RUL in hours",
    ["equipment_id"],
)

MODELS_LOADED = _get_or_create(
    Gauge,
    "models_loaded",
    "Number of models currently loaded",
)

KAFKA_PIPELINE_RUNNING = _get_or_create(
    Gauge,
    "kafka_pipeline_running",
    "Whether the Kafka prediction pipeline is running (1=yes, 0=no)",
)

SERVICE_UPTIME_SECONDS = _get_or_create(
    Gauge,
    "service_uptime_seconds",
    "Time since service started in seconds",
)

CURRENT_ACCURACY = _get_or_create(
    Gauge,
    "model_rolling_accuracy",
    "Rolling within-tolerance accuracy from PerformanceMonitor (0-1) — "
    "the same value compared against the 80% retraining-trigger threshold",
    ["model"],
)

MODEL_VERSION_INFO = _get_or_create(
    Gauge,
    "model_version_info",
    "Currently loaded model version (value=1 marks the active version)",
    ["model", "version"],
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
